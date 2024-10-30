import copy
import math, bisect, mpmath

import clarabel
import numpy as np
import cvxpy as cp
import clarabel as cb
import highspy
from scipy import sparse

from hydradx.model.amm.omnipool_amm import OmnipoolState


class ICEProblem:
    def __init__(self,
                 omnipool: OmnipoolState,
                 intents: list,
                 tkn_profit: str = "HDX",
                 min_partial: float = 1,
                 apply_min_partial: bool = True
                 ):
        self.omnipool = omnipool
        self.min_partial = min_partial
        self.intents = intents
        assert tkn_profit in omnipool.asset_list
        self.tkn_profit = tkn_profit
        temp_intents = []
        for intent in self.intents:
            temp_intents.append(copy.deepcopy(intent))
            buy_amt_lrna_value = intent['buy_quantity'] * self.omnipool.price(self.omnipool, intent['tkn_buy'])
            selL_amt_lrna_value = intent['sell_quantity'] * self.omnipool.price(self.omnipool, intent['tkn_sell'])
            if buy_amt_lrna_value < self.min_partial and selL_amt_lrna_value < self.min_partial and apply_min_partial:
                temp_intents[-1]['partial'] = False

        self.partial_intent_indices = [i for i in range(len(temp_intents)) if temp_intents[i]['partial']]
        self.full_intent_indices = [i for i in range(len(temp_intents)) if not temp_intents[i]['partial']]
        self.partial_intents = [temp_intents[i] for i in self.partial_intent_indices]
        self.full_intents = [temp_intents[i] for i in self.full_intent_indices]

        self.m = len(self.partial_intents)
        self.r = len(self.full_intents)

        self.asset_list = [self.tkn_profit]
        for intent in self.intents:
            if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in self.asset_list:
                self.asset_list.append(intent['tkn_sell'])
            if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in self.asset_list:
                self.asset_list.append(intent['tkn_buy'])
        self.n = len(self.asset_list)
        self.fee_match = 0.0005
        assert self.fee_match <= min([self.omnipool.last_fee[tkn] for tkn in self.asset_list])
        self.I = None
        self._directional_flags = None
        self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]

        self._known_flow = None
        self._scaling = None
        self.omnipool_directions = None  # whether Omnipool is buying or selling each asset
        self._tau = None
        self._phi = None
        self._max_in = None
        self._max_out = None
        self._amm_lrna_coefs = None
        self._amm_asset_coefs = None
        self._q = None  # self._q @ x_descaled is the LRNA profit of the solver
        self._profit_A = None  # self._profit_A @ x_descaled is the leftover in each asset

    def clear(self):
        self.I = None
        self._directional_flags = None
        self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]

        self._known_flow = None
        self._scaling = None
        self.omnipool_directions = None  # whether Omnipool is buying or selling each asset
        self._tau = None
        self._phi = None
        self._max_in = None
        self._max_out = None
        self._amm_lrna_coefs = None
        self._amm_asset_coefs = None
        self._q = None
        self._profit_A = None

    def _set_known_flow(self):
        self._known_flow = {tkn: {'in': 0, 'out': 0} for tkn in ["LRNA"] + self.asset_list}
        # if self.I is not None:  # full intent executions are known
        assert len(self.I) == len(self.full_intents)
        for i, intent in enumerate(self.full_intents):
            if self.I[i] > 0.5:
                self._known_flow[intent['tkn_sell']]['in'] += intent["sell_quantity"]
                self._known_flow[intent['tkn_buy']]['out'] += intent["buy_quantity"]

    # note that max out is not enforced in Omnipool, it's used to scale variables and use good estimates for AMMs
    # in particular, the max_out for tkn_profit does not reflect that the solver will buy it with any leftover
    def _set_max_in_out(self):
        self._max_in = {tkn: 0 for tkn in self.asset_list + ['LRNA']}
        self._max_out = {tkn: 0 for tkn in self.asset_list + ['LRNA']}
        for i, intent in enumerate(self.partial_intents):
            self._max_in[intent['tkn_sell']] += self.partial_sell_maxs[i]
            buy_amt = intent['buy_quantity'] / intent['sell_quantity'] * self.partial_sell_maxs[i]
            self._max_out[intent['tkn_buy']] += math.nextafter(buy_amt, math.inf) if buy_amt != 0 else 0
        if self.I is None:
            for intent in self.full_intents:
                self._max_in[intent['tkn_sell']] += intent['sell_quantity']
                self._max_out[intent['tkn_buy']] += intent['buy_quantity']
        for tkn in self._known_flow:
            self._max_in[tkn] += self._known_flow[tkn]['in'] - self._known_flow[tkn]['out']
            self._max_out[tkn] -= self._known_flow[tkn]['in'] - self._known_flow[tkn]['out']
        fees = {tkn: self.omnipool.last_fee[tkn] for tkn in self.asset_list}
        for tkn in self.asset_list:
            self._max_in[tkn] = max(self._max_in[tkn], 0)
            self._max_out[tkn] = max(self._max_out[tkn] / (1 - fees[tkn]), 0)
        self._max_out["LRNA"] = 0
        self._max_in["LRNA"] = max(self._max_in["LRNA"], 0)

    def _set_scaling(self):
        self._scaling = {tkn: 0 for tkn in self.asset_list}
        self._scaling["LRNA"] = 0
        for tkn in self.asset_list:
            self._scaling[tkn] = max(self._max_in[tkn], self._max_out[tkn])
            if self._scaling[tkn] == 0:
                self._scaling[tkn] = 1
            else:
                self._scaling[tkn] = min(self._scaling[tkn], self.omnipool.liquidity[tkn])
            # set scaling for LRNA equal to scaling for asset, adjusted by spot price
            scalar = self._scaling[tkn] * self.omnipool.lrna[tkn] / self.omnipool.liquidity[tkn]
            self._scaling["LRNA"] = max(self._scaling["LRNA"], scalar)

    def _set_omnipool_directions(self):
        known_intent_directions = {self.tkn_profit: 'buy'}  # solver collects profits in tkn_profit
        for j, intent in enumerate(self.partial_intents):
            if self.partial_sell_maxs[j] > 0:
                if intent['tkn_sell'] not in known_intent_directions:
                    known_intent_directions[intent['tkn_sell']] = "sell"
                elif known_intent_directions[intent['tkn_sell']] == "buy":
                    known_intent_directions[intent['tkn_sell']] = "both"
                if intent['tkn_buy'] not in known_intent_directions:
                    known_intent_directions[intent['tkn_buy']] = "buy"
                elif known_intent_directions[intent['tkn_buy']] == "sell":
                    known_intent_directions[intent['tkn_buy']] = "both"

        for tkn in self.asset_list:
            if self._known_flow[tkn]['in'] > self._known_flow[tkn]['out']:  # net agent is selling tkn
                if tkn not in known_intent_directions:
                    known_intent_directions[tkn] = "sell"
                elif known_intent_directions[tkn] == "buy":
                    known_intent_directions[tkn] = "both"
            elif self._known_flow[tkn]['in'] < self._known_flow[tkn]['out']:  # net agent is buying tkn
                if tkn not in known_intent_directions:
                    known_intent_directions[tkn] = "buy"
                elif known_intent_directions[tkn] == "sell":
                    known_intent_directions[tkn] = "both"
            elif self._known_flow[tkn]['in'] > 0:  # known flow matches, will need fees from Omnipool
                if tkn not in known_intent_directions:
                    known_intent_directions[tkn] = "buy"
                elif known_intent_directions[tkn] == "sell":
                    known_intent_directions[tkn] = "both"

        self._omnipool_directions = {}
        for tkn in self.asset_list:
            if tkn in known_intent_directions:
                if known_intent_directions[tkn] == "sell":
                    self._omnipool_directions[tkn] = "buy"
                elif known_intent_directions[tkn] == "buy":
                    self._omnipool_directions[tkn] = "sell"
            else:  # no trades in the asset
                self._omnipool_directions[tkn] = "neither"


    def _set_tau_phi(self):
        tau1 = sparse.csc_matrix((self.n + 1, self.m))
        phi1 = sparse.csc_matrix((self.n + 1, self.m))
        tau2 = sparse.csc_matrix((self.n + 1, self.r))
        phi2 = sparse.csc_matrix((self.n + 1, self.r))
        tkn_list = ["LRNA"] + self.asset_list
        for j, intent in enumerate(self.partial_intents):
            sell_i = tkn_list.index(intent['tkn_sell'])
            buy_i = tkn_list.index(intent['tkn_buy'])
            tau1[sell_i, j] = 1
            phi1[buy_i, j] = 1
        for l, intent in enumerate(self.full_intents):
            sell_i = tkn_list.index(intent['tkn_sell'])
            buy_i = tkn_list.index(intent['tkn_buy'])
            tau2[sell_i, l] = 1
            phi2[buy_i, l] = 1

        self._tau = sparse.hstack([tau1, tau2])
        self._phi = sparse.hstack([phi1, phi2])

    def _set_amm_coefs(self):
        self._amm_lrna_coefs = {tkn: self._scaling["LRNA"] / self.omnipool.lrna[tkn] for tkn in self.asset_list}
        self._amm_asset_coefs = {tkn: self._scaling[tkn] / self.omnipool.liquidity[tkn] for tkn in self.asset_list}

    def _set_coefficients(self):
        # profit calculations
        # variables are y_i, x_i, lrna_lambda_i, lambda_i, d_j, I_l
        # y_i are net LRNA into Omnipool
        profit_lrna_y_coefs = -np.ones(self.n)
        # x_i are net assets into Omnipool
        profit_lrna_x_coefs = np.zeros(self.n)
        # lrna_lambda_i are LRNA amounts coming out of Omnipool
        lrna_fees = [self.omnipool.last_lrna_fee[tkn] for tkn in self.asset_list]
        profit_lrna_lrna_lambda_coefs = -np.array(lrna_fees)
        profit_lrna_lambda_coefs = np.zeros(self.n)
        profit_lrna_d_coefs = np.array([self._tau[0, j] for j in range(self.m)])
        profit_lrna_I_coefs = np.array([self._tau[0, self.m + l] * self.full_intents[l]['sell_quantity'] / self._scaling["LRNA"] for l in range(self.r)])
        profit_lrna_coefs = np.concatenate([profit_lrna_y_coefs, profit_lrna_x_coefs, profit_lrna_lrna_lambda_coefs, profit_lrna_lambda_coefs, profit_lrna_d_coefs, profit_lrna_I_coefs])

        # leftover must be higher than required fees
        # other assets
        tkn_list = ["LRNA"] + self.asset_list
        fees = [self.omnipool.last_fee[tkn] for tkn in self.asset_list]
        partial_intent_prices = self.get_partial_intent_prices()
        profit_y_coefs = sparse.csc_matrix((self.n, self.n))
        profit_x_coefs = -sparse.identity(self.n, format='csc')
        profit_lrna_lambda_coefs = sparse.csc_matrix((self.n, self.n))
        profit_lambda_coefs = -sparse.diags(np.array(fees).astype(float) - self.fee_match, format='csc')
        profit_d_coefs = -sparse.csc_matrix([[1 / (1 - self.fee_match) * self._phi[i, j] * float(
            partial_intent_prices[j] * self._scaling[self.partial_intents[j]['tkn_sell']] / self._scaling[
                self.partial_intents[j]['tkn_buy']]) - self._tau[i, j] for j in range(self.m)] for i in range(1, self.n + 1)])
        I_coefs = -sparse.csc_matrix([[float((1 / (1 - self.fee_match) * self._phi[i, self.m + l] * self.full_intents[l]['buy_quantity'] -
                                             self._tau[i, self.m + l] * self.full_intents[l]['sell_quantity']) / self._scaling[tkn_list[i]])
                                      for l in range(self.r)] for i in range(1, self.n + 1)])
        profit_A_LRNA = sparse.csc_matrix(profit_lrna_coefs.astype(float))
        profit_A_assets = sparse.hstack([profit_y_coefs, profit_x_coefs, profit_lrna_lambda_coefs, profit_lambda_coefs, profit_d_coefs, I_coefs])
        self._profit_A = sparse.vstack([profit_A_LRNA, profit_A_assets], format='csc')

        profit_i = self.asset_list.index(self.tkn_profit)
        self._q = self._profit_A[profit_i, :].toarray().flatten()

    def _recalculate(self):
        self._set_known_flow()
        self._set_max_in_out()
        self._set_scaling()
        self._set_omnipool_directions()
        self._set_tau_phi()
        self._set_amm_coefs()
        self._set_coefficients()

    def set_up_problem(self, I: list, flags: dict = None, sell_maxes: list = None, clear_sell_maxes: bool = True):
        assert len(I) == len(self.full_intents)
        self.I = I
        if sell_maxes is not None:
            self.partial_sell_maxs = sell_maxes
        elif clear_sell_maxes:
            self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]
        if flags is not None:
            self._directional_flags = flags
        self._recalculate()

    def get_amm_lrna_coefs(self):
        return {k: v for k, v in self._amm_lrna_coefs.items()}

    def get_amm_asset_coefs(self):
        return {k: v for k, v in self._amm_asset_coefs.items()}

    def get_q(self):
        return np.array([v for v in self._q])

    def get_profit_A(self):
        return sparse.csc_matrix(self._profit_A)

    def get_omnipool_directions(self):
        return {k: v for k, v in self._omnipool_directions.items()}

    def get_epsilon_tkn(self):
        return {t: max([abs(self._max_in[t]), abs(self._max_out[t])]) / self.omnipool.liquidity[t] for t in self.asset_list}

    def get_fees(self):
        return [self.omnipool.last_fee[tkn] for tkn in self.asset_list]

    def get_lrna_fees(self):
        return [self.omnipool.last_lrna_fee[tkn] for tkn in self.asset_list]

    def get_partial_intent_prices(self):
        partial_intent_prices = [intent['buy_quantity'] / intent['sell_quantity'] if intent['sell_quantity'] > 0 else 0
                                 for intent in self.partial_intents]
        return partial_intent_prices

    def get_partial_sell_maxs_scaled(self):
        return [self.partial_sell_maxs[j] / self._scaling[intent['tkn_sell']] for j, intent in enumerate(self.partial_intents)]

    def scale_LRNA_amt(self, amt):
        return amt * self._scaling["LRNA"]

    def get_real_x(self, x):
        '''
        Get the real asset quantities from the scaled x.
        x has the stucture [y_i, x_i, lrna_lambda_i, lambda_i, d_j, I_l],
        although it may or may not have the I_l values.
        The y_i and lrna_lambda_i are scaled to with scaling["LRNA"],
        while the x_i and lambda_i are scaled with scaling[tkn].
        The d_i are scaled to scaling[sell_tkn], and the I_l are in {0,1}.
        '''
        n, m, r = self.n, self.m, self.r
        assert len(x) in [4 * n + m, 4 * n + m + r]
        scaled_yi = [x[i] * self._scaling["LRNA"] for i in range(n)]
        scaled_xi = [x[n + i] * self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
        scaled_lrna_lambda = [x[2*n + i] * self._scaling["LRNA"] for i in range(n)]
        scaled_lambda = [x[3 * n + i] * self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
        scaled_d = [x[4 * n + j] * self._scaling[intent['tkn_sell']] for j, intent in enumerate(self.partial_intents)]
        scaled_x = np.concatenate([scaled_yi, scaled_xi, scaled_lrna_lambda, scaled_lambda, scaled_d])
        if len(x) == 4 * n + m + r:
            scaled_I = [x[4 * n + m + l] for l in range(r)]
            scaled_x = np.concatenate([scaled_x, scaled_I])
        return scaled_x


def _calculate_tau_phi(intents: list, tkn_list: list, scaling: dict) -> tuple:
    n = len(tkn_list)
    m = len(intents)
    tau = sparse.csc_matrix((n, m))
    phi = sparse.csc_matrix((n, m))
    for j, intent in enumerate(intents):
        sell_i = tkn_list.index(intent['tkn_sell'])
        buy_i = tkn_list.index(intent['tkn_buy'])
        tau[sell_i, j] = 1
        phi[buy_i, j] = 1
    return tau, phi


def _calculate_scaling(partial_intents: list, full_intents: list, I: list, state: OmnipoolState, asset_list: list):
    scaling = {tkn: 0 for tkn in asset_list}
    max_in = {tkn: 0 for tkn in asset_list + ['LRNA']}
    max_out = {tkn: 0 for tkn in asset_list + ['LRNA']}
    scaling["LRNA"] = 0
    for intent in partial_intents:
        max_in[intent['tkn_sell']] += intent['sell_quantity']
        max_out[intent['tkn_buy']] += intent['buy_quantity']
    for i, intent in enumerate(full_intents):
        if I[i] > 0.5:
            max_in[intent['tkn_sell']] += intent['sell_quantity']
            max_out[intent['tkn_sell']] -= intent['sell_quantity']
            max_out[intent['tkn_buy']] += intent['buy_quantity']
            max_in[intent['tkn_buy']] -= intent['buy_quantity']
    for tkn in asset_list:
        scaling[tkn] = max(max_in[tkn], max_out[tkn])
        if scaling[tkn] == 0:
            scaling[tkn] = 1
        else:
            scaling[tkn] = min(scaling[tkn], state.liquidity[tkn])
        # set scaling for LRNA equal to scaling for asset, adjusted by spot price
        scalar = scaling[tkn] * state.lrna[tkn] / state.liquidity[tkn]
        scaling["LRNA"] = max(scaling["LRNA"], scalar)
    return scaling, max_in, max_out


def scale_down_partial_intents(p, trade_pcts):
    zero_ct = 0
    intent_sell_maxs = []
    for i, m in enumerate(p.partial_sell_maxs):
        # we allow new solution to find trade size up to 10x old solution
        new_sell_quantity = min([m * trade_pcts[i] * 10, m])
        tkn = p.partial_intents[i]['tkn_sell']
        sell_amt_lrna_value = new_sell_quantity * p.omnipool.price(p.omnipool, tkn)
        # if we are scaling lower than min_partial, we eliminate the intent from execution
        if sell_amt_lrna_value < p.min_partial:
            new_sell_quantity = 0
            zero_ct += 1  # we count the number of intents that are eliminated
        intent_sell_maxs.append(new_sell_quantity)
    return intent_sell_maxs, zero_ct


def _find_solution_unrounded(
        p: ICEProblem
) -> (dict, list):

    if p.I is None:
        raise
    I = p.I
    if sum(I) + sum(p.partial_sell_maxs) == 0:  # nothing to execute
        return {tkn: 0 for tkn in p.asset_list}, [0] * len(p.partial_intents), np.zeros(4 * p.n + p.m), 0, 0, 'Solved'

    full_intents, partial_intents, state = p.full_intents, p.partial_intents, p.omnipool

    # assets involved trades for which execution is being solved
    asset_list = p.asset_list
    n, m, r = p.n, p.m, p.r
    if len(partial_intents) + sum(I) == 0:  # nothing for solver to do
        return {tkn: 0 for tkn in asset_list}, [], np.zeros(4*n), 0, 0, 'Solved'  # TODO enable solver with m=0

    directions = p.get_omnipool_directions()
    k = 4 * n + m

    indices_to_keep = list(range(k))
    for tkn in directions:
        if directions[tkn] in ["sell", "neither"]:
            indices_to_keep.remove(2 * n + asset_list.index(tkn))  # lrna_lambda_i is zero
        if directions[tkn] in ["buy", "neither"]:
            indices_to_keep.remove(3 * n + asset_list.index(tkn))  # lambda_i is zero
        if directions[tkn] == "neither":
            indices_to_keep.remove(asset_list.index(tkn))  # y_i is zero
            indices_to_keep.remove(n + asset_list.index(tkn))  # x_i is zero

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    k_real = len(indices_to_keep)
    P_trimmed = sparse.csc_matrix((k_real, k_real))
    q_all = p.get_q()
    objective_I_coefs = -q_all[4*n+m:]
    q = -q_all[:4*n+m]
    q_trimmed = np.array([q[i] for i in indices_to_keep])

    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#

    diff_coefs = sparse.csc_matrix((2*n + m,2*n))
    nonzero_coefs = -sparse.identity(2 * n + m, format='csc')
    A1 = sparse.hstack([diff_coefs, nonzero_coefs])
    rows_to_keep = [i for i in range(2*n+m) if 2*n+i in indices_to_keep]
    A1_trimmed = A1[:, indices_to_keep][rows_to_keep, :]
    b1 = np.zeros(A1_trimmed.shape[0])
    cone1 = cb.NonnegativeConeT(A1_trimmed.shape[0])

    # intent variables are constrained from above
    amm_coefs = sparse.csc_matrix((m, 4*n))
    d_coefs = sparse.identity(m, format='csc')
    A2 = sparse.hstack([amm_coefs, d_coefs], format='csc')
    b2 = np.array(p.get_partial_sell_maxs_scaled())
    A2_trimmed = A2[:, indices_to_keep]
    cone2 = cb.NonnegativeConeT(m)

    # # leftover must be higher than required fees
    profit_A = p.get_profit_A()
    A3 = -profit_A[:, :4 * n + m]
    I_coefs = -profit_A[:, 4 * n + m:]
    A3_trimmed = A3[:, indices_to_keep]
    if r == 0:
        b3 = np.zeros(n+1)
    else:
        b3 = -I_coefs @ I
    cone3 = cb.NonnegativeConeT(n + 1)

    # AMM invariants must not go down
    amm_lrna_coefs = p.get_amm_lrna_coefs()
    amm_asset_coefs = p.get_amm_asset_coefs()
    A4 = sparse.csc_matrix((0, k))
    b4 = np.array([])
    cones4 = []
    epsilon_tkn = p.get_epsilon_tkn()
    for i in range(n):
        tkn = asset_list[i]
        if epsilon_tkn[tkn] <= 1e-6:  # linearize the AMM constraint
            if tkn not in directions:
                c1 = 1 / (1 + epsilon_tkn[tkn])
                c2 = 1 / (1 - epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((2, k))
                b4i = np.zeros(2)
                A4i[0, i] = -amm_lrna_coefs[tkn]
                A4i[0, n + i] = -amm_asset_coefs[tkn] * c1
                A4i[1, i] = -amm_lrna_coefs[tkn]
                A4i[1, n + i] = -amm_asset_coefs[tkn] * c2
                cones4.append(cb.NonnegativeConeT(2))
            else:
                if directions[tkn] == "sell":
                    c = 1 / (1 - epsilon_tkn[tkn])
                else:
                    c = 1 / (1 + epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((1, k))
                b4i = np.zeros(1)
                A4i[0, i] = -amm_lrna_coefs[tkn]
                A4i[0, n+i] = -amm_asset_coefs[tkn] * c
                cones4.append(cb.ZeroConeT(1))
        elif epsilon_tkn[tkn] <= 1e-3:  # quadratic approximation to in-given-out function
            A4i = sparse.csc_matrix((3, k))
            A4i[1,i] = -amm_lrna_coefs[tkn]
            A4i[1,n+i] = -amm_asset_coefs[tkn]
            A4i[2,n+i] = -amm_asset_coefs[tkn]
            b4i = np.array([1, 0, 0])
            cones4.append(cb.PowerConeT(0.5))
        else:  # full AMM constraint
            A4i = sparse.csc_matrix((3, k))
            b4i = np.ones(3)
            A4i[0, i] = -amm_lrna_coefs[tkn]
            A4i[1, n + i] = -amm_asset_coefs[tkn]
            cones4.append(cb.PowerConeT(0.5))
        A4 = sparse.vstack([A4, A4i])
        b4 = np.append(b4, b4i)
    A4_trimmed = A4[:, indices_to_keep]

    # A5: inequality constraints on comparison of lrna_lambda to yi, lambda to xi
    A5 = sparse.csc_matrix((0, k))
    # A6: inequality constraints on xi, yi
    A6 = sparse.csc_matrix((0, k))
    # A7: equality constraints on lrna_lambda to yi, lambda to xi, if known
    A7 = sparse.csc_matrix((0, k))
    for i in range(n):
        tkn = asset_list[i]
        if tkn not in directions:
            A5i = sparse.csc_matrix((2, k))
            A5i[0, i] = -1  # lrna_lambda + yi >= 0
            A5i[0, 2*n+i] = -1  # lrna_lambda + yi >= 0
            A5i[1, n+i] = -1  # lambda + xi >= 0
            A5i[1, 3*n+i] = -1  # lambda + xi >= 0
            A5 = sparse.vstack([A5, A5i])
        else:
            A6i = sparse.csc_matrix((2, k))
            A7i = sparse.csc_matrix((1, k))
            if directions[tkn] == "sell":
                A6i[0, i] = -1  # yi >= 0
                A6i[1, n+i] = 1  # xi <= 0
                A7i[0, n+i] = 1  # xi + lambda = 0
                A7i[0, 3*n+i] = 1  # xi + lambda = 0
            else:
                A6i[0, i] = 1  # yi <= 0
                A6i[1, n+i] = -1  # xi >= 0
                A7i[0, i] = 1  # yi + lrna_lambda = 0
                A7i[0, 2*n+i] = 1  # yi + lrna_lambda = 0
            A6 = sparse.vstack([A6, A6i])
            A7 = sparse.vstack([A7, A7i])


    A5_trimmed = A5[:, indices_to_keep]
    A6_trimmed = A6[:, indices_to_keep]
    A7_trimmed = A7[:, indices_to_keep]

    b5 = np.zeros(A5.shape[0])
    b6 = np.zeros(A6.shape[0])
    b7 = np.zeros(A7.shape[0])
    cone5 = cb.NonnegativeConeT(A5.shape[0])
    cone6 = cb.NonnegativeConeT(A6.shape[0])
    cone7 = cb.ZeroConeT(A7.shape[0])

    A = sparse.vstack([A1_trimmed, A2_trimmed, A3_trimmed, A4_trimmed, A5_trimmed, A6_trimmed, A7_trimmed], format='csc')
    b = np.concatenate([b1, b2, b3, b4, b5, b6, b7])
    cones = [cone1, cone2, cone3] + cones4 + [cone5, cone6, cone7]

    # solve
    settings = clarabel.DefaultSettings()
    settings.max_step_fraction = 0.95
    solver = clarabel.DefaultSolver(P_trimmed, q_trimmed, A, b, cones, settings)
    solution = solver.solve()
    x = solution.x
    z = solution.z
    s = solution.s

    new_amm_deltas = {}
    exec_intent_deltas = [None] * len(partial_intents)
    x_expanded = [0] * k
    for i in range(k):
        if i in indices_to_keep:
            x_expanded[i] = x[indices_to_keep.index(i)]
    x_scaled = p.get_real_x(x_expanded)
    for i in range(n):
        tkn = asset_list[i]
        new_amm_deltas[tkn] = x_scaled[n+i]

    for j in range(len(partial_intents)):
        exec_intent_deltas[j] = -x_scaled[4 * n + j]

    fixed_profit = objective_I_coefs @ I if I is not None else 0
    return (new_amm_deltas, exec_intent_deltas, x_expanded, p.scale_LRNA_amt(solution.obj_val + fixed_profit),
            p.scale_LRNA_amt(solution.obj_val_dual + fixed_profit), str(solution.status))


def _solve_inclusion_problem(
        p: ICEProblem,
        x: np.array = None,  # NLP solution
        upper_bound: float = None,
        lower_bound: float = None,
        old_A = None,
        old_A_upper = None,
        old_A_lower = None
):
    state = p.omnipool
    tkn_profit = p.tkn_profit
    asset_list = p.asset_list
    tkn_list = ["LRNA"] + asset_list

    partial_intents = p.partial_intents
    full_intents = p.full_intents
    n, m, r = p.n, p.m, p.r
    k = 4 * n + m + r

    fees = [float(state.last_fee[tkn]) for tkn in asset_list]  # f_i
    lrna_fees = [float(state.last_lrna_fee[tkn]) for tkn in asset_list]  # l_i
    fee_match = 0.0005
    assert fee_match <= min(fees)  # breaks otherwise

    partial_intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) for intent in partial_intents]
    full_intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) for intent in full_intents]

    # calculate tau, phi
    scaling, _, _ = _calculate_scaling(partial_intents + full_intents, [], [], state, asset_list)
    tau1, phi1 = _calculate_tau_phi(partial_intents, tkn_list, scaling)
    tau2, phi2 = _calculate_tau_phi(full_intents, tkn_list, scaling)
    tau = sparse.hstack([tau1, tau2]).toarray()
    phi = sparse.hstack([phi1, phi2]).toarray()

    # we start with the 4n + m variables from the initial problem
    # then we add r indicator variables for the r non-partial intents
    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    y_coefs = np.ones(n)
    x_coefs = np.zeros(n)
    lrna_lambda_coefs = np.array(lrna_fees)
    lambda_coefs = np.zeros(n)
    d_coefs = np.array([-tau[0,j] for j in range(m)])
    I_coefs = np.array([-tau[0,m+l]*full_intents[l]['sell_quantity']/scaling["LRNA"] for l in range(r)])
    c = np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs, I_coefs])

    # bounds on variables
    # y, x are unbounded
    # lrna_lambda, lambda >= 0
    # 0 <= d <= max
    # 0 <= I <= 1

    inf = highspy.kHighsInf

    if upper_bound is None:
        upper_bound = inf
    if lower_bound is None:
        lower_bound = -inf

    lower = np.array([-inf] * 2 * n + [0] * (2 * n + m + r))
    partial_intent_sell_amts = [i['sell_quantity']/scaling[i['tkn_sell']] for i in partial_intents]
    upper = np.array([inf] * 4 * n + partial_intent_sell_amts + [1] * r)

    # we will temporarily assume a 0 solution is latest, and linearize g() around that.
    S = np.zeros((n, k))
    S_upper = np.zeros(n)
    S_lower = np.array([-inf]*n)
    grads = np.zeros(2*n)
    for i in range(n):
        grads[i] = -scaling["LRNA"] * state.liquidity[asset_list[i]] - scaling["LRNA"] * scaling[asset_list[i]] * x[n+i]
        grads[n+i] = -scaling[asset_list[i]] * state.lrna[asset_list[i]] - scaling["LRNA"] * scaling[asset_list[i]] * x[i]
        S[i, i] = grads[i]
        S[i, n+i] = grads[n+i]
        S_upper[i] = (grads[i] * x[n+i] + grads[n+i] * x[n+i] + scaling["LRNA"] * state.liquidity[asset_list[i]] * x[i]
                + scaling[asset_list[i]] * state.lrna[asset_list[i]] * x[n+i]
                + scaling["LRNA"] * scaling[asset_list[i]] * x[i] * x[n+i])

    # asset leftover must be above zero
    A3 = np.zeros((n+1, k))
    A3[0, :] = c
    for i in range(n):
        A3[i+1, n+i] = 1
        A3[i+1, 3*n+i] = fees[i] - fee_match
        for j in range(m):
            A3[i+1, 4*n+j] = 1/(1-fee_match)*phi[i+1, j] *scaling[partial_intents[j]['tkn_sell']]/scaling[partial_intents[j]['tkn_buy']] * partial_intent_prices[j] - tau[i+1, j]
        for l in range(r):
            buy_amt = 1 / (1 - fee_match) * phi[i+1, m+l] * full_intents[l]['buy_quantity']
            sell_amt = tau[i + 1, m+l] * full_intents[l]['sell_quantity']
            A3[i+1, 4*n+m+l] = (buy_amt - sell_amt)/scaling[asset_list[i]]
    A3_upper = np.zeros(n+1)
    A3_lower = np.array([-inf]*(n+1))

    # asset leftover in tkn_profit is actual objective
    profit_i = asset_list.index(tkn_profit)
    row_profit = A3[profit_i+1, :]

    # sum of lrna_lambda and y_i should be non-negative
    # sum of lambda and x_i should be non-negative
    A5 = np.zeros((2 * n, k))
    for i in range(n):
        A5[i, i] = 1
        A5[i, 2 * n + i] = 1
        A5[n + i, n + i] = 1
        A5[n + i, 3 * n + i] = 1
    A5_upper = np.array([inf] * 2 * n)
    A5_lower = np.zeros(2 * n)

    # optimized value must be lower than best we have so far, higher than lower bound
    A8 = np.zeros((1, k))
    A8[0, :] = c
    A8_upper = np.array([upper_bound])
    A8_lower = np.array([lower_bound])

    if old_A is None:
        old_A = np.zeros((0, k))
    if old_A_upper is None:
        old_A_upper = np.array([])
    if old_A_lower is None:
        old_A_lower = np.array([])
    assert len(old_A_upper) == len(old_A_lower) == old_A.shape[0]
    A = np.vstack([old_A, S, A3, A5, A8])
    A_upper = np.concatenate([old_A_upper, S_upper, A3_upper, A5_upper, A8_upper])
    A_lower = np.concatenate([old_A_lower, S_lower, A3_lower, A5_lower, A8_lower])

    nonzeros = []
    start = [0]
    a = []
    for i in range(A.shape[0]):
        row_nonzeros = np.where(A[i, :] != 0)[0]
        nonzeros.extend(row_nonzeros)
        start.append(len(nonzeros))
        a.extend(A[i, row_nonzeros])
    h = highspy.Highs()
    lp = highspy.HighsLp()

    lp.num_col_ = k
    lp.num_row_ = A.shape[0]

    lp.col_cost_ = row_profit
    lp.col_lower_ = lower
    lp.col_upper_ = upper
    lp.row_lower_ = A_lower
    lp.row_upper_ = A_upper

    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = nonzeros
    lp.a_matrix_.value_ = a

    lp.integrality_ = np.array([highspy.HighsVarType.kContinuous] * (4*n + m) + [highspy.HighsVarType.kInteger] * r)

    h.passModel(lp)
    h.run()
    solution = h.getSolution()
    info = h.getInfo()
    basis = h.getBasis()

    x_expanded = solution.col_value

    new_amm_deltas = {}
    exec_partial_intent_deltas = [None] * len(partial_intents)
    exec_full_intent_flags = [None] * len(full_intents)

    for i in range(n):
        tkn = tkn_list[i+1]
        new_amm_deltas[tkn] = x_expanded[n+i] * scaling[tkn]

    for i in range(m):
        exec_partial_intent_deltas[i] = -x_expanded[4 * n + i] * scaling[partial_intents[i]['tkn_sell']]

    exec_full_intent_flags = [1 if x_expanded[4 * n + m + i] > 0.5 else 0 for i in range(r)]

    save_A = np.vstack([old_A, S])
    save_A_upper = np.concatenate([old_A_upper, S_upper])
    save_A_lower = np.concatenate([old_A_lower, S_lower])

    return new_amm_deltas, exec_partial_intent_deltas, exec_full_intent_flags, save_A, save_A_upper, save_A_lower, c @ x_expanded * scaling["LRNA"], solution.value_valid


def round_solution(intents, intent_deltas, tolerance=0.0001):
    deltas = []
    for i in range(len(intent_deltas)):
        # don't leave dust in intent due to rounding error
        if intents[i]['sell_quantity'] + intent_deltas[i] < tolerance * intents[i]['sell_quantity']:
            deltas.append(-intents[i]['sell_quantity'])
        # don't trade dust amount due to rounding error
        elif -intent_deltas[i] <= tolerance * intents[i]['sell_quantity']:
            deltas.append(0)
        else:
            deltas.append(intent_deltas[i])
    return deltas


def add_buy_deltas(intents, sell_deltas) -> list:
    deltas = []
    for i in range(len(intents)):
        if intents[i]['sell_quantity'] == 0:
            deltas.append([0, 0])
        else:
            deltas.append([sell_deltas[i], -sell_deltas[i] * intents[i]['buy_quantity'] / intents[i]['sell_quantity']])
    return deltas


def get_directional_flags(amm_deltas: dict) -> dict:
    flags = {}
    for tkn in amm_deltas:
        delta = amm_deltas[tkn]
        if delta > 0:
            flags[tkn] = 1
        elif delta < 0:
            flags[tkn] = -1
        else:
            flags[tkn] = 0
    return flags


def find_solution(state: OmnipoolState, intents: list) -> list:
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents)
    flags = get_directional_flags(amm_deltas)
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents, flags = flags)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)


def find_solution_outer_approx(state: OmnipoolState, init_intents: list, min_partial: float = 1) -> list:
    if len(init_intents) == 0:
        return []

    p = ICEProblem(state, init_intents, min_partial=min_partial)

    m, r, n = p.m, p.r, p.n
    inf = highspy.kHighsInf
    k_milp = 4 * n + m + r
    # get initial I values
    indicators = [0]*r
    # set Z_L = -inf, Z_U = inf
    Z_L = -inf
    Z_U = inf
    best_status = "Not Solved"
    y_best = indicators
    best_amm_deltas = [0]*n
    best_intent_deltas = [0]*m
    milp_obj = -inf
    new_A, new_A_upper, new_A_lower = np.zeros((0, k_milp)), np.array([]), np.array([])
    # loop until MILP has no solution:
    for _i in range(50):
        # - update I^(K+1), Z_L
        Z_L = max(Z_L, milp_obj)
        # - do NLP solve given I values, update x^K
        p.set_up_problem(I=indicators)
        amm_deltas, intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded(p)
        if obj < Z_U and dual_obj < 0:  # - update Z_U, y*, x*
            Z_U = obj
            x_best = x
            y_best = indicators
            best_amm_deltas = amm_deltas
            best_intent_deltas = intent_deltas
            best_status = status
        # - get new cone constraint from I^K
        BK = np.where(np.array(indicators) == 1)[0] + 4 * n + m
        NK = np.where(np.array(indicators) == 0)[0] + 4 * n + m
        IC_A = np.zeros((1, k_milp))
        IC_A[0, BK] = 1
        IC_A[0, NK] = -1
        IC_upper = np.array([len(BK) - 1])
        IC_lower = np.array([-inf])

        # - add cone constraint to A, A_upper, A_lower
        A = np.vstack([new_A, IC_A])
        A_upper = np.concatenate([new_A_upper, IC_upper])
        A_lower = np.concatenate([new_A_lower, IC_lower])

        # - do MILP solve
        amm_deltas, partial_intent_deltas, indicators, new_A, new_A_upper, new_A_lower, milp_obj, valid = _solve_inclusion_problem(p, x, Z_U, Z_L, A, A_upper, A_lower)
        if not valid:
            break

    if valid == True:  # this means we did not get to a solution
        return [[0,0]]*(m + r)

    trade_pcts = [-best_intent_deltas[i] / m for i, m in enumerate(p.partial_sell_maxs)]

    # if solution is not good yet, try scaling down partial intent sizes, to get scaling better
    while len(p.partial_intents) > 0 and (best_status != "Solved" or Z_U > 0) and min(trade_pcts) < 0.05:
        new_maxes, zero_ct = scale_down_partial_intents(p, trade_pcts)
        p.set_up_problem(I=y_best, sell_maxes=new_maxes)
        if zero_ct == m:
            break  # all partial intents have been eliminated from execution

        amm_deltas, intent_deltas, x, obj, dual_obj, temp_status = _find_solution_unrounded(p)
        if temp_status in ['PrimalInfeasible', 'DualInfeasible']:
            # the better scaling revealed that there is no actual solution
            return [[0,0]]*(m + r)
        if obj < Z_U:
            best_amm_deltas = amm_deltas
            best_intent_deltas = intent_deltas
            best_x = x
            Z_U = obj
            status = temp_status
        else:
            break  # break if no improvement in solution
        trade_pcts = [-best_intent_deltas[i] / m if m > 0 else 0 for i, m in enumerate(p.partial_sell_maxs)]


    flags = get_directional_flags(best_amm_deltas)
    p.set_up_problem(I=y_best, flags=flags, clear_sell_maxes=False)
    best_amm_deltas, best_intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded(p)
    # linearize = []
    # _, max_in, max_out = _calculate_scaling(new_partial_intents, full_intents, y_best, state, asset_list)
    # epsilon_tkn_ls = [(max([abs(max_in[t]), abs(max_out[t])]) / state.liquidity[t], t) for t in asset_list]
    # epsilon_tkn_ls.sort()
    # loc = bisect.bisect_right([x[0] for x in epsilon_tkn_ls], epsilon)
    # while status != "Solved" and loc < len(epsilon_tkn_ls) and epsilon_tkn_ls[loc][0] < 0:
    #     # force linearization of asset with smallest epsilon
    #     linearize.append(epsilon_tkn_ls[loc][1])
    #     loc += 1
    #     best_amm_deltas, best_intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded3(state, new_partial_intents,
    #                                                                                     full_intents, I=y_best,
    #                                                                                     flags=flags, epsilon=epsilon,
    #                                                                                     force_linear = linearize)
    if status not in ["Solved", "AlmostSolved"]:
        if obj > 0:
            return [[0,0]]*(m+r)  # no solution found
        else:
            raise
    sell_deltas = round_solution(p.partial_intents, best_intent_deltas)
    partial_deltas_with_buys = add_buy_deltas(p.partial_intents, sell_deltas)
    full_deltas_with_buys = [[-p.full_intents[l]['sell_quantity'], p.full_intents[l]['buy_quantity']] if y_best[l] == 1 else [0,0] for l in range(r)]
    deltas = [None] * (m + r)
    for i in range(len(p.partial_intent_indices)):
        deltas[p.partial_intent_indices[i]] = partial_deltas_with_buys[i]
    for i in range(len(p.full_intent_indices)):
        deltas[p.full_intent_indices[i]] = full_deltas_with_buys[i]
    return deltas
