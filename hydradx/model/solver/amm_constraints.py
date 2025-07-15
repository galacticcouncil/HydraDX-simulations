import copy, math, numpy as np, clarabel as cb
from abc import abstractmethod

import highspy

from hydradx.model.amm.exchange import Exchange
from hydradx.model.amm.xyk_amm import ConstantProductPoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

INF = highspy.kHighsInf

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
        self.asset_list = [tkn for tkn in amm.asset_list]
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

    @abstractmethod
    def get_amm_bounds(self, approx: str, scaling: dict) -> tuple:
        pass

    def get_amm_constraint_matrix(self, approx: str, scaling: dict, amm_directions: list, last_amm_deltas: list,
                                  trading_tkns: list):
        A1, b1, cones1, cone_sizes1 = self.get_amm_limits_A(amm_directions, last_amm_deltas, trading_tkns)
        A2, b2, cones2, cone_sizes2 = self.get_amm_bounds(approx, scaling)
        A = np.vstack([A1, A2])
        b = np.concatenate([b1, b2])
        cones = cones1 + cones2
        cone_sizes = cone_sizes1 + cone_sizes2
        return A, b, cones, cone_sizes

    @abstractmethod
    def get_boundary_values(self, global_asset_list: list, scaling_mat) -> tuple:
        pass

    @abstractmethod
    def get_linearized_amm_constraints(self, x_list, global_asset_list: list, scaling_mat):
        pass

    def get_delta_limits(self):  # inequality constraints: X_j + L_j >= 0
        A = np.zeros((len(self.asset_list) + 1, self.k))
        for i in range(len(self.asset_list) + 1):
            A[i, self.amm_i.shares_net + i] = 1  # X_j
            A[i, self.amm_i.shares_out + i] = 1  # L_j
        A_upper = np.array([INF] * A.shape[0])
        A_lower = np.zeros(A.shape[0])
        return A, A_lower, A_upper

    def get_milp_constraints(self, x_list, global_asset_list, scaling_mat):
        A1, A1_lower, A1_upper = self.get_linearized_amm_constraints(x_list, global_asset_list, scaling_mat)
        A2, A2_lower, A2_upper = self.get_delta_limits()
        A = np.vstack([A1, A2])
        A_lower = np.concatenate([A1_lower, A2_lower])
        A_upper = np.concatenate([A1_upper, A2_upper])
        return A, A_lower, A_upper

    @abstractmethod
    def get_approx(self, deltas: list):
        pass

    @abstractmethod
    def upgrade_approx(self, deltas: list, current_approx):
        pass

    @abstractmethod
    def get_linear_approx(self):
        pass


class XykConstraints(AmmConstraints):
    def __init__(self, amm: ConstantProductPoolState):
        super().__init__(amm)
        self.trade_fee = amm.trade_fee(amm.asset_list[0], 0)  # assuming fixed uniform fee
        self.auxiliary_ct = 0
        self.k = 2*(len(amm.asset_list) + 1)
        self.shares = amm.shares
        self.liquidity = {tkn: amm.liquidity[tkn] for tkn in amm.asset_list}

    def get_amm_bounds(self, approx: str, scaling: dict) -> tuple:
        # TODO implement linear, quadratic approximations
        amm_i = self.amm_i
        coef = [scaling[self.tkn_share] / self.shares] + [scaling[tkn] / self.liquidity[tkn] for tkn in self.asset_list]
        # if approx == "linear":  # linearize the AMM constraint
        #     c1 = 1 / (1 + epsilon_tkn[tkn])
        #     c2 = 1 / (1 - epsilon_tkn[tkn]) if epsilon_tkn[tkn] < 1 else 1e15
        #     A5j2 = np.zeros((2, k))
        #     b5j2 = np.zeros(2)
        #     A5j2[0, amm_i.asset_net[0]] = -B[1]
        #     A5j2[0, amm_i.asset_net[1]] = -B[2] * c1
        #     A5j2[1, amm_i.asset_net[0]] = -B[1]
        #     A5j2[1, amm_i.asset_net[1]] = -B[2] * c2
        #     cones5j.append(cb.NonnegativeConeT(2))
        #     cones_count5j.append(2)
        # else:  # full constraint
        #     A5j2 = np.zeros((3, k))
        #     b5j2 = np.ones(3)
        #     A5j2[0, amm_i.asset_net[0]] = -B[1]
        #     A5j2[1, amm_i.asset_net[1]] = -B[2]
        #     cones5j.append(cb.PowerConeT(0.5))
        #     cones_count5j.append(3)
        A = np.zeros((3, self.k))
        b = np.ones(3)
        A[0, amm_i.asset_net[0]] = -coef[1]
        A[1, amm_i.asset_net[1]] = -coef[2]
        A[2, amm_i.shares_net] = -coef[0]
        cones = [cb.PowerConeT(0.5)]
        cone_sizes = [3]

        return A, b, cones, cone_sizes

    def get_boundary_values(self, global_asset_list: list, scaling_mat):
        rho, psi = self.get_indicator_matrices(global_asset_list)
        C = rho.T @ scaling_mat
        B = psi.T @ scaling_mat
        max_L = np.array([0] + [self.liquidity[tkn] for tkn in self.asset_list]) / (B + C)
        max_X = [0] + [INF] * len(self.asset_list)
        min_X = [-x for x in max_L]
        min_L = np.zeros(len(self.asset_list) + 1)
        max_vals = np.concatenate([max_X, max_L])
        min_vals = np.concatenate([min_X, min_L])
        return min_vals, max_vals

    def get_linearized_amm_constraints(self, x_list, global_asset_list: list, scaling_mat):
        S = np.zeros((0, self.k))
        S_upper = np.zeros(0)
        amm_i = self.amm_i
        _, psi = self.get_indicator_matrices(global_asset_list)
        B = psi.T @ scaling_mat
        for x in x_list:
            S_row = np.zeros((1, self.k))
            S_row_upper = np.zeros(1)
            # lrna_c = p.get_omnipool_lrna_coefs()
            # asset_c = p.get_omnipool_asset_coefs()
            i = amm_i.asset_net[0]
            j = amm_i.asset_net[1]
            c0 = B[1] / self.liquidity[self.asset_list[0]]
            c1 = B[2] / self.liquidity[self.asset_list[1]]
            grads_x0 = -c0 - c0 * c1 * x[j]
            grads_x1 = -c1 - c0 * c1 * x[i]
            # TODO generalize this process for any AMM type
            S_row[0, i] = grads_x0
            S_row[0, j] = grads_x1
            grad_dot_x = grads_x0 * x[i] + grads_x1 * x[j]
            g_neg = c0 * x[i] + c1 * x[j] + c0 * c1 * x[i] * x[j]
            S_row_upper[0] = grad_dot_x + g_neg
            S = np.vstack([S, S_row])
            S_upper = np.concatenate([S_upper, S_row_upper])
        S_lower = np.array([-INF] * len(S_upper))  # no lower bounds for linearized constraints
        return S, S_lower, S_upper

    def get_approx(self, deltas: list) -> str:
        pcts = [abs(deltas[0]) / self.shares]  # first shares size constraint, delta_s / s_0 <= epsilon
        pcts.extend([abs(deltas[j + 1]) / self.liquidity[tkn] for j, tkn in enumerate(self.asset_list)])
        approx = "linear" if max(pcts) <= 1e-5 else "full"
        return approx

    def upgrade_approx(self, deltas: list, current_approx: str) -> str:
        if current_approx == "full":
            return "full"  # we don't want to downgrade full constraints to linear approximations
        return self.get_approx(deltas)

    def get_linear_approx(self):  # generate a linear approximation object
        return "linear"


class StableswapConstraints(AmmConstraints):
    def __init__(self, amm: Exchange):
        super().__init__(amm)
        self.trade_fee = amm.trade_fee
        self.auxiliary_ct = len(self.asset_list) + 1
        self.k = 2*(len(amm.asset_list) + 1) + self.auxiliary_ct
        self.shares = amm.shares
        self.liquidity = {tkn: amm.liquidity[tkn] for tkn in amm.asset_list}
        self.ann = amm.ann
        self.d = amm.d

    def get_amm_bounds(self, approx: str, scaling: dict):
        B = [0] + [scaling[tkn] for tkn in self.asset_list]
        C = [scaling[self.tkn_share]]
        ann = self.ann
        s0 = self.shares
        D0 = self.d
        amm_i = self.amm_i
        n_amm = len(self.asset_list) + 1
        sum_assets = sum([self.liquidity[tkn] for tkn in self.asset_list])
        # D0' = D_0 * (1 - 1/ann)
        D0_prime = D0 * (1 - 1 / ann)
        # a0 ~= -delta_s/s0 + [1 / (sum x_i^0 - D0') * sum delta_x_i - (D0'/s0) / (sum x_i^0 - D0') * delta_s]
        denom = sum_assets - D0_prime
        if approx[0] == "linear":
            A = np.zeros((1, self.k))
            A[0, amm_i.aux[0]] = 1  # a_0 coefficient
            A[0, amm_i.shares_net] = (1 + D0_prime / denom) * C[0] / s0  # X_0 coefficient
            for t in range(1, n_amm):
                A[0, amm_i.asset_net[t - 1]] = -B[t] / denom  # X_t coefficient
            b = np.array([0])
            cones = [cb.ZeroConeT(1)]
            cone_sizes = [1]
        else:
            A = np.zeros((3, self.k))
            b = np.array([0, 0, 0])
            # x = a_0
            A[0, amm_i.aux[0]] = -1  # a_0 coefficient
            # y = 1 + C * X_0 / S
            A[1, amm_i.shares_net] = -C[0] / s0  # X_0 coefficient
            b[1] = 1
            # z = An^n / D_0 sum(x_i^0 + B_i X_i) + (1 - An^n)(1 + C_jS_j / s_0)
            A[2, amm_i.shares_net] = D0_prime * C[0] / denom / s0
            for t in range(1, n_amm):
                A[2, amm_i.asset_net[t - 1]] = -B[t] / denom
            b[2] = 1
            cones = [cb.ExponentialConeT()]
            cone_sizes = [3]

        for t in range(1, n_amm):
            x0 = self.liquidity[self.asset_list[t - 1]]
            if approx[t] == "linear":
                A_t = np.zeros((1, self.k))
                A_t[0, amm_i.aux[t]] = 1  # a_t coefficient
                A_t[0, amm_i.shares_net] = C[0] / s0  # X_0 coefficient
                A_t[0, amm_i.asset_net[t - 1]] = -B[t] / x0  # X_i coefficient
                b_t = np.array([0])
                cone_t = cb.ZeroConeT(1)
                cone_size_t = 1
            else:
                A_t = np.zeros((3, self.k))
                b_t = np.zeros(3)
                # x = a_t
                A_t[0, amm_i.aux[t]] = -1
                # y = 1 + C * X_0 / S
                A_t[1, amm_i.shares_net] = -C[0] / s0
                b_t[1] = 1
                # z = 1 + B_t X_t / R_t
                A_t[2, amm_i.asset_net[t - 1]] = -B[t] / x0
                b_t[2] = 1
                cone_t = cb.ExponentialConeT()
                cone_size_t = 3
            cones.append(cone_t)
            cone_sizes.append(cone_size_t)
            A = np.vstack([A, A_t])
            b = np.append(b, np.array(b_t))

        A_final = np.zeros((1, self.k))
        for t in range(n_amm):
            A_final[0, amm_i.aux[t]] = -1
        b_final = np.array([0])
        A = np.vstack([A, A_final])
        b = np.append(b, b_final)
        cones.append(cb.NonnegativeConeT(1))
        cone_sizes.append(1)

        return A, b, cones, cone_sizes

    def get_boundary_values(self, global_asset_list: list, scaling_mat):
        inf = highspy.kHighsInf
        rho, psi = self.get_indicator_matrices(global_asset_list)
        C = rho.T @ scaling_mat
        B = psi.T @ scaling_mat
        max_L = np.array([self.shares] + [self.liquidity[tkn] for tkn in self.asset_list]) / (B + C)
        max_X = [inf] * (len(self.asset_list) + 1)
        min_X = [-x for x in max_L]
        min_L = np.zeros(len(self.asset_list) + 1)
        min_a = [-inf] * self.auxiliary_ct
        max_a = [inf] * self.auxiliary_ct
        max_vals = np.concatenate([max_X, max_L, max_a])
        min_vals = np.concatenate([min_X, min_L, min_a])
        return min_vals, max_vals

    def get_linearized_amm_constraints(self, x_list, global_asset_list: list, scaling_mat):
        S = np.zeros((0, self.k))
        S_upper = np.zeros(0)
        amm_i = self.amm_i
        rho, psi = self.get_indicator_matrices(global_asset_list)
        B = psi.T @ scaling_mat
        C = rho.T @ scaling_mat
        for x in x_list:
            D0_prime = self.d - self.d / self.ann
            s0 = self.shares
            c = C[0]
            sum_assets = sum([self.liquidity[tkn] for tkn in self.asset_list])
            denom = sum_assets - D0_prime
            a0 = x[amm_i.aux[0]]
            X0 = x[amm_i.shares_net]
            exp = np.exp(float(a0 / (1 + (c * X0 / s0))))
            term = (c / s0 - c * a0 / (s0 + c)) * exp
            # linearization of shares constraint, i.e. a0
            S_row = np.zeros((1, self.k))
            grad_s = term + c * D0_prime / (s0 * denom)
            grads_i = [- B[l] / denom for l in range(1, len(self.asset_list) + 1)]
            grad_a = exp
            S_row[0, amm_i.shares_net] = grad_s
            S_row[0, amm_i.aux[0]] = grad_a
            for l, tkn in enumerate(self.asset_list):
                S_row[0, amm_i.asset_net[l]] = grads_i[l]
            grad_dot_x = grad_s * X0 + grad_a * a0 + sum(
                [grads_i[l] * x[amm_i.asset_net[l]] for l in range(len(self.asset_list))])
            sum_deltas = sum([B[l + 1] * x[amm_i.asset_net[l]] for l in range(len(self.asset_list))])
            g_neg = (1 + c * X0 / s0) * exp - sum_deltas / denom + D0_prime * c * X0 / (denom * s0) - 1
            S_row_upper = np.array([grad_dot_x + g_neg])
            S = np.vstack([S, S_row])
            S_upper = np.concatenate([S_upper, S_row_upper])

            # linearization of asset constraints, i.e. a1, a2, ...
            for l, tkn in enumerate(self.asset_list):
                S_row = np.zeros((1, self.k))
                grad_s = term
                grad_a = exp
                grad_x = -B[l + 1] / self.liquidity[tkn]
                S_row[0, amm_i.shares_net] = grad_s
                S_row[0, amm_i.aux[l + 1]] = grad_a
                S_row[0, amm_i.asset_net[l]] = grad_x
                ai = x[amm_i.aux[l + 1]]
                grad_dot_x = grad_s * X0 + grad_a * ai + grad_x * x[amm_i.asset_net[l]]
                g_neg = (1 + c * X0 / s0) * exp - B[l + 1] * x[amm_i.asset_net[l]] / self.liquidity[tkn] - 1
                S_row_upper = np.array([grad_dot_x + g_neg])
                S = np.vstack([S, S_row])
                S_upper = np.concatenate([S_upper, S_row_upper])
        S_lower = np.array([-INF] * len(S_upper))  # no lower bounds for linearized constraints
        return S, S_lower, S_upper

    def get_overall_constraint(self):
        A = np.zeros((1, self.k))
        for aux_i in self.amm_i.aux:
            A[0, aux_i] = 1
        A_upper = np.array([INF])
        A_lower = np.zeros(1)
        return A, A_lower, A_upper

    def get_milp_constraints(self, x_list, global_asset_list, scaling_mat):
        A1, A1_lower, A1_upper = super().get_milp_constraints(x_list, global_asset_list, scaling_mat)
        A2, A2_lower, A2_upper = self.get_overall_constraint()
        A = np.vstack([A1, A2])
        A_lower = np.concatenate([A1_lower, A2_lower])
        A_upper = np.concatenate([A1_upper, A2_upper])
        return A, A_lower, A_upper

    def get_approx(self, deltas: list) -> list[str]:
        pcts = [abs(deltas[0]) / self.shares]  # first shares size constraint, delta_s / s_0 <= epsilon
        sum_delta_x = sum([abs(deltas[j + 1]) for j in range(len(self.asset_list))])
        pcts.append(sum_delta_x / self.d)
        pcts.extend([abs(deltas[j + 1]) / self.liquidity[tkn] for j, tkn in enumerate(self.asset_list)])
        approx = ["linear" if max(pcts[0], pcts[1]) <= 1e-5 else "full"]
        for pct in pcts[2:]:
            approx.append("linear" if pct <= 1e-5 else "full")
        return approx

    def upgrade_approx(self, deltas: list, current_approx: list[str]) -> list[str]:
        new_approx = self.get_approx(deltas)
        for i, approx in enumerate(current_approx):
            if approx == "full" and new_approx[i] == "linear":
                new_approx[i] = "full"  # we don't want to downgrade full constraints to linear approximations
        return new_approx

    def get_linear_approx(self):  # generate a linear approximation object
        return ["linear"] * (len(self.asset_list) + 1)