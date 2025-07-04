import copy, math, numpy as np
from abc import abstractmethod

from hydradx.model.amm.exchange import Exchange
from hydradx.model.amm.xyk_amm import ConstantProductPoolState


class AmmConstraints:
    def __init__(self, amm: Exchange):
        self.asset_list = amm.asset_list
        self.tkn_shares = amm.unique_id

    def get_indicator_matrices(self, global_asset_list: list[str]):
        asset_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        share_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        for i, asset in enumerate(global_asset_list):
            if asset == self.tkn_shares:
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


class StableswapConstraints(AmmConstraints):
    def __init__(self, amm: Exchange):
        super().__init__(amm)
        self.trade_fee = amm.trade_fee
        self.auxiliary_ct = len(self.asset_list) + 1
