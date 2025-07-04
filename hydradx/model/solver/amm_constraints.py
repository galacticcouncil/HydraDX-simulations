import copy, math, numpy as np, clarabel as cb
from abc import abstractmethod

from hydradx.model.amm.exchange import Exchange
from hydradx.model.amm.xyk_amm import ConstantProductPoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState


class AmmIndexObject:
    def __init__(self, amm, offset: int):
        n_amm = len(amm.asset_list) + 1
        self.shares_net = offset  # X_0
        self.shares_out = offset + n_amm  # L_0
        self.asset_net = []
        self.asset_out = []
        for i in range(1, n_amm):
            self.asset_net.append(offset + i)  # X_i
            self.asset_out.append(offset + n_amm + i)  # L_i
        self.aux = []
        if isinstance(amm, StableSwapPoolState):
            for i in range(n_amm):
                self.aux.append(offset + 2 * n_amm + i)
        elif isinstance(amm, ConstantProductPoolState):
            pass  # no auxiliary variables for XYK AMM
        else:  # TODO generalize auxiliary handling beyond stableswap
            raise ValueError("Only stableswap implemented for now")
        self.num_vars = 2 * n_amm + len(self.aux)  # total number of variables for this AMM
        self.net_is = slice(self.shares_net, self.shares_out)  # returns X_0, ..., X_n indices
        self.out_is = slice(self.shares_out, self.shares_out + n_amm)  # returns L_0, ..., L_n indices
        self.aux_is = slice(self.shares_out + n_amm, self.shares_out + n_amm + len(self.aux))  # returns aux indices


class AmmConstraints:
    def __init__(self, amm: Exchange):
        self.asset_list = amm.asset_list
        self.tkn_share = amm.unique_id
        self.amm_i = AmmIndexObject(amm, 0)

    def get_amm_limits_A(self, amm_directions: list, last_amm_deltas: list, trading_tkns: list):
        """Uses AMM structures and directional information to bound AMM variables"""
        if last_amm_deltas is None:
            last_amm_deltas = []
        cones_limits = []
        cones_sizes = []

        if len(amm_directions) > 0 and len(last_amm_deltas) > 0:
            delta_pct = last_amm_deltas[0] / self.shares  # possibly round to zero
        else:
            delta_pct = 1  # to avoid causing any rounding
        if self.tkn_share in trading_tkns and abs(delta_pct) > 1e-11:
            A_limits = np.zeros((2, self.k))
            A_limits[0, self.amm_i.shares_out] = -1  # L0 >= 0
            A_limits[1, self.amm_i.shares_net] = -1  # X_0 + L_0 >= 0
            A_limits[1, self.amm_i.shares_out] = -1  # X_0 + L_0 >= 0
            cones_limits.append(cb.NonnegativeConeT(2))
            cones_sizes.append(2)
            if len(amm_directions) > 0:
                if (dir := amm_directions[0]) in ["buy", "sell"]:
                    A_limits_dir = np.zeros((1, self.k))
                    A_limits_dir[0, self.amm_i.shares_net] = -1 if dir == "buy" else 1
                    A_limits = np.vstack([A_limits, A_limits_dir])
                    cones_limits.append(cb.NonnegativeConeT(1))
                    cones_sizes.append(1)
        else:
            A_limits = np.zeros((2, self.k))  # TODO delete variables instead of forcing to zero
            A_limits[0, self.amm_i.shares_net] = 1
            A_limits[1, self.amm_i.shares_out] = 1
            cones_limits.append(cb.ZeroConeT(2))
            cones_sizes.append(2)
        for j, tkn in enumerate(self.asset_list):
            if len(amm_directions) > 0 and len(last_amm_deltas) > 0:
                delta_pct = last_amm_deltas[j + 1] / self.liquidity[tkn]  # possibly round to zero
            else:
                delta_pct = 1  # to avoid causing any rounding
            if tkn in trading_tkns and abs(delta_pct) > 1e-11:
                A_limits_j = np.zeros((2, self.k))
                A_limits_j[0, self.amm_i.asset_out[j]] = -1  # Lj >= 0
                A_limits_j[1, self.amm_i.asset_net[j]] = -1  # Xj + Lj >= 0
                A_limits_j[1, self.amm_i.asset_out[j]] = -1
                cones_limits.append(cb.NonnegativeConeT(2))
                cones_sizes.append(2)
                if len(amm_directions) > 0:
                    if (dir := amm_directions[j + 1]) in ["buy", "sell"]:
                        A_limits_j_dir = np.zeros((1, self.k))
                        A_limits_j_dir[0, self.amm_i.asset_net[j]] = -1 if dir == "buy" else 1
                        A_limits_j = np.vstack([A_limits_j, A_limits_j_dir])
                        cones_limits.append(cb.NonnegativeConeT(1))
                        cones_sizes.append(1)
            else:
                A_limits_j = np.zeros((2, self.k))  # TODO delete variables instead of forcing to zero
                A_limits_j[0, self.amm_i.asset_net[j]] = 1
                A_limits_j[1, self.amm_i.asset_out[j]] = 1
                cones_limits.append(cb.ZeroConeT(2))
                cones_sizes.append(2)
            A_limits = np.vstack([A_limits, A_limits_j])
        b_limits = np.zeros(A_limits.shape[0])
        return A_limits, b_limits, cones_limits, cones_sizes

    def get_indicator_matrices(self, global_asset_list: list[str]):
        asset_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        share_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        for i, asset in enumerate(global_asset_list):
            if asset == self.tkn_share:
                share_mat[i, 0] = 1
            if asset in self.asset_list:
                asset_mat[i, self.asset_list.index(asset) + 1] = 1
        return share_mat, asset_mat

    def get_profit_coefs(self, global_asset_list, buffer_fee: float = 0.0, fee_match: float = 0.0):
        share_mat, asset_mat = self.get_indicator_matrices(global_asset_list)
        fees = [self.trade_fee + buffer_fee - fee_match] * (len(self.asset_list) + 1)
        X_coefs = share_mat - asset_mat
        L_coefs = asset_mat @ np.diag(-np.array(fees))
        a_coefs = np.zeros((len(global_asset_list), self.auxiliary_ct))
        return np.hstack((X_coefs, L_coefs, a_coefs))


class XykConstraints(AmmConstraints):
    def __init__(self, amm: ConstantProductPoolState):
        super().__init__(amm)
        self.trade_fee = amm.trade_fee(amm.asset_list[0], 0)  # assuming fixed uniform fee
        self.auxiliary_ct = 0
        self.k = 2*(len(amm.asset_list) + 1)
        self.shares = amm.shares
        self.liquidity = {tkn: amm.liquidity[tkn] for tkn in amm.asset_list}


class StableswapConstraints(AmmConstraints):
    def __init__(self, amm: Exchange):
        super().__init__(amm)
        self.trade_fee = amm.trade_fee
        self.auxiliary_ct = len(self.asset_list) + 1
        self.k = 2*(len(amm.asset_list) + 1) + self.auxiliary_ct
        self.shares = amm.shares
        self.liquidity = {tkn: amm.liquidity[tkn] for tkn in amm.asset_list}
