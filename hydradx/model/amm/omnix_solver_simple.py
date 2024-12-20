import copy
import math, bisect, mpmath

import clarabel
import numpy as np
import clarabel as cb
import highspy
from scipy import sparse

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.stableswap_amm import StableSwapPoolState


class ICEProblem:
    def __init__(self,
                 omnipool: OmnipoolState,
                 intents: list,
                 amm_list: list = None,
                 tkn_profit: str = "HDX",
                 init_i: list = None,
                 min_partial: float = 1,
                 apply_min_partial: bool = True,
                 trading_tkns: list = None,
                 ):
        self.omnipool = omnipool
        if amm_list is None:
            self.amm_list = []
        else:
            self.amm_list = amm_list
        self.min_partial = min_partial
        self.intents = intents
        assert tkn_profit in omnipool.asset_list
        self.tkn_profit = tkn_profit
        temp_intents = []
        for intent in self.intents:
            new_intent = {k: v for k, v in intent.items() if k != 'agent'}
            new_intent['agent'] = intent['agent'].copy()
            temp_intents.append(new_intent)
            if intent['tkn_buy'] in omnipool.asset_list and intent['tkn_sell'] in omnipool.asset_list:
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
        self.s = len(self.amm_list)

        if init_i is not None:
            init_i_full = [i for i in range(self.r) if self.full_intent_indices[i] in init_i]
            self.I = [0]*self.r
            for i in init_i_full:
                self.I[i] = 1
        else:
            self.I = None

        self.sigmas = []
        self.auxiliaries = []
        self.asset_list = [tkn for tkn in omnipool.asset_list]
        for amm in self.amm_list:  # add assets from other amms
            share_tkn = amm.unique_id
            if share_tkn not in self.asset_list:
                self.asset_list.append(share_tkn)
            self.sigmas.append(len(amm.asset_list)+1)
            if isinstance(amm, StableSwapPoolState):
                self.auxiliaries.append(len(amm.asset_list) + 1)
            else:
                self.auxiliaries.append(0)
            for tkn in amm.asset_list:
                if tkn not in self.asset_list:
                    self.asset_list.append(tkn)
        self.sigma = sum(self.sigmas)
        self.u = sum(self.auxiliaries)
        for intent in self.intents:  # add assets from intents
            if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in self.asset_list:
                self.asset_list.append(intent['tkn_sell'])
            if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in self.asset_list:
                self.asset_list.append(intent['tkn_buy'])
        self.n = len(omnipool.asset_list)
        self.N = len(self.asset_list)
        self.fee_match = 0.0000
        # assert self.fee_match <= min([self.omnipool.last_fee[tkn] for tkn in self.asset_list])
        self._omnipool_directional_flags = None
        self._amm_directional_flags = None
        self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]

        self._known_flow = None
        self._scaling = None
        self.omnipool_directions = None  # whether Omnipool is buying or selling each asset
        self._tau = None
        self._phi = None
        self._o = None
        self._rho = None
        self._phi = None
        self._max_in = None
        self._max_out = None
        self._omnipool_lrna_coefs = None
        self._omnipool_asset_coefs = None
        self._q = None  # self._q @ x_descaled is the LRNA profit of the solver
        self._profit_A = None  # self._profit_A @ x_descaled is the leftover in each asset
        self._last_omnipool_deltas = None
        self._last_amm_deltas = None

        if trading_tkns is None:
            self.trading_tkns = ["LRNA"] + self.asset_list
        else:
            self.trading_tkns = trading_tkns

        self._set_indicator_matrices()  # only depends on amm structures

    def clear(self):
        self.I = None
        self._omnipool_directional_flags = None
        self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]

        self._known_flow = None
        self._scaling = None
        self.omnipool_directions = None  # whether Omnipool is buying or selling each asset
        self._tau = None
        self._phi = None
        self._max_in = None
        self._max_out = None
        self._omnipool_lrna_coefs = None
        self._omnipool_asset_coefs = None
        self._q = None
        self._profit_A = None

    def _set_known_flow(self):
        self._known_flow = {tkn: {'in': 0, 'out': 0} for tkn in ["LRNA"] + self.asset_list}
        if self.I is not None:  # full intent executions are known
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
        self._min_in = {tkn: 0 for tkn in self.asset_list + ['LRNA']}
        self._min_out = {tkn: 0 for tkn in self.asset_list + ['LRNA']}
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
            self._min_in[tkn] += self._known_flow[tkn]['in'] - self._known_flow[tkn]['out']
            self._max_out[tkn] -= self._known_flow[tkn]['in'] - self._known_flow[tkn]['out']
            self._min_out[tkn] -= self._known_flow[tkn]['in'] - self._known_flow[tkn]['out']
        for tkn in self.asset_list:
            self._max_in[tkn] = max(self._max_in[tkn], 0)
            self._min_in[tkn] = max(self._min_in[tkn], 0)
            self._max_out[tkn] = max(self._max_out[tkn], 0)
            self._min_out[tkn] = max(self._min_out[tkn], 0)
        self._max_out["LRNA"] = 0
        self._min_out["LRNA"] = 0
        self._max_in["LRNA"] = max(self._max_in["LRNA"], 0)
        self._min_in["LRNA"] = max(self._min_in["LRNA"], 0)

    # def _set_bounds(self):
    #     # assets
    #     self._min_x = np.array([self._min_in[tkn] - self._max_out[tkn] for tkn in self.asset_list])
    #     self._max_x = np.array([self._max_in[tkn] - self._min_out[tkn] for tkn in self.asset_list])
    #     self._min_lambda = np.maximum(-self._max_x, 0)
    #     self._max_lambda = np.maximum(-self._min_x, 0)
    #     # LRNA
    #     min_y = np.array([-self.omnipool.lrna[tkn] * self._max_x[i] / (self._max_x[i] + self.omnipool.liquidity[tkn]) for i, tkn in enumerate(self.asset_list)])
    #     self._min_y = min_y - 0.1 * np.abs(min_y)
    #     max_y = np.array([-self.omnipool.lrna[tkn] * self._min_x[i] / (self._min_x[i] + self.omnipool.liquidity[tkn]) for i, tkn in enumerate(self.asset_list)])
    #     self._max_y = max_y + 0.1 * np.abs(max_y)
    #     self._min_lrna_lambda = np.maximum(-self._max_y, 0)
    #     self._max_lrna_lambda = np.maximum(-self._min_y, 0)
    #     # tkn_profit
    #     profit_i = self.asset_list.index(self.tkn_profit)
    #     self._min_x[profit_i] = -self.omnipool.liquidity[self.tkn_profit]
    #     self._max_lambda[profit_i] = np.maximum(-self._min_x[profit_i], 0)
    #     self._min_y[profit_i] = -self.omnipool.lrna[self.tkn_profit]
    #     self._max_lrna_lambda[profit_i] = np.maximum(-self._min_y[profit_i], 0)

    def _set_scaling(self):
        self._scaling = {tkn: 0 for tkn in self.asset_list}
        self._scaling["LRNA"] = 1
        for tkn in self.asset_list:
            self._scaling[tkn] = max(self._max_in[tkn], self._max_out[tkn])
            if self._scaling[tkn] == 0:
                self._scaling[tkn] = 1
            # set scaling for LRNA equal to scaling for asset, adjusted by spot price
            if tkn in self.omnipool.asset_list:
                if self._last_omnipool_deltas is not None:
                    self._scaling[tkn] = max(self._scaling[tkn], abs(self._last_omnipool_deltas[tkn]))
                scalar = self._scaling[tkn] * self.omnipool.lrna[tkn] / self.omnipool.liquidity[tkn]
                self._scaling["LRNA"] = max(self._scaling["LRNA"], scalar)
                # raise scaling for tkn_profit to scaling for asset, adjusted by spot price, if needed
                scalar_profit = self._scaling[tkn] * self.omnipool.price(self.omnipool, tkn, self.tkn_profit)
                self._scaling[self.tkn_profit] = max(self._scaling[self.tkn_profit], scalar_profit / 10000)
        for amm in self.amm_list:  # raise scaling for all assets in each AMM to match max
            max_scale = self._scaling[amm.unique_id]
            for tkn in amm.asset_list:
                max_scale = max(max_scale, self._scaling[tkn])
            self._scaling[amm.unique_id] = max_scale
            for tkn in amm.asset_list:
                self._scaling[tkn] = max_scale

        # if self._last_omnipool_deltas is not None:
        #     self._scaling[self.tkn_profit] = abs(self._last_omnipool_deltas[self.tkn_profit])

        if self._last_amm_deltas is not None:
            assert len(self._last_amm_deltas) == len(self.amm_list)
            for i, amm in enumerate(self.amm_list):
                assert len(amm.asset_list) + 1 == len(self._last_amm_deltas[i])
                self._scaling[amm.unique_id] = max(self._scaling[amm.unique_id], abs(self._last_amm_deltas[i][0]))
                for j, tkn in enumerate(amm.asset_list):
                    self._scaling[tkn] = max(self._scaling[tkn], abs(self._last_amm_deltas[i][j+1]))

    def _set_directions(self):
        # known_intent_directions = {self.tkn_profit: 'both'}  # solver collects profits in tkn_profit
        # for j, intent in enumerate(self.partial_intents):
        #     if self.partial_sell_maxs[j] > 0:
        #         if intent['tkn_sell'] not in known_intent_directions:
        #             known_intent_directions[intent['tkn_sell']] = "sell"
        #         elif known_intent_directions[intent['tkn_sell']] == "buy":
        #             known_intent_directions[intent['tkn_sell']] = "both"
        #         if intent['tkn_buy'] not in known_intent_directions:
        #             known_intent_directions[intent['tkn_buy']] = "buy"
        #         elif known_intent_directions[intent['tkn_buy']] == "sell":
        #             known_intent_directions[intent['tkn_buy']] = "both"
        #
        # for tkn in self.asset_list:
        #     if self._known_flow[tkn]['in'] > self._known_flow[tkn]['out']:  # net agent is selling tkn
        #         if tkn not in known_intent_directions:
        #             known_intent_directions[tkn] = "sell"
        #         elif known_intent_directions[tkn] == "buy":
        #             known_intent_directions[tkn] = "both"
        #     elif self._known_flow[tkn]['in'] < self._known_flow[tkn]['out']:  # net agent is buying tkn
        #         if tkn not in known_intent_directions:
        #             known_intent_directions[tkn] = "buy"
        #         elif known_intent_directions[tkn] == "sell":
        #             known_intent_directions[tkn] = "both"
        #     elif self._known_flow[tkn]['in'] > 0:  # known flow matches, will need fees from Omnipool
        #         if tkn not in known_intent_directions:
        #             known_intent_directions[tkn] = "buy"
        #         elif known_intent_directions[tkn] == "sell":
        #             known_intent_directions[tkn] = "both"

        self._omnipool_directions = {}
        # for tkn in self.omnipool.asset_list:
        #     if tkn in self._directional_flags:
        #         if self._directional_flags[tkn] == -1:
        #             self._omnipool_directions[tkn] = "sell"
        #         elif self._directional_flags[tkn] == 1:
        #             self._omnipool_directions[tkn] = "buy"
        #         elif self._directional_flags[tkn] == 0:
        #             self._omnipool_directions[tkn] = "neither"
            # elif tkn in known_intent_directions:
            #     if known_intent_directions[tkn] == "sell":
            #         self._omnipool_directions[tkn] = "buy"
            #     elif known_intent_directions[tkn] == "buy":
            #         self._omnipool_directions[tkn] = "sell"
            # else:  # no trades in the asset
            #     self._omnipool_directions[tkn] = "neither"

        for tkn in self._omnipool_directional_flags:
            if self._omnipool_directional_flags[tkn] == -1:
                self._omnipool_directions[tkn] = "sell"
            elif self._omnipool_directional_flags[tkn] == 1:
                self._omnipool_directions[tkn] = "buy"
            elif self._omnipool_directional_flags[tkn] == 0:
                self._omnipool_directions[tkn] = "neither"

        self._amm_directions = []
        for l in self._amm_directional_flags:
            new_list = []
            for f in l:
                if f == -1:
                    new_list.append("sell")
                elif f == 1:
                    new_list.append("buy")
                elif f == 0:
                    new_list.append("neither")
            self._amm_directions.append(new_list)

    def _set_tau_phi(self):
        tau1 = np.zeros((self.N + 1, self.m))
        phi1 = np.zeros((self.N + 1, self.m))
        tau2 = np.zeros((self.N + 1, self.r))
        phi2 = np.zeros((self.N + 1, self.r))
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

        self._tau = np.hstack([tau1, tau2])
        self._phi = np.hstack([phi1, phi2])

    def _set_indicator_matrices(self):
        self._o = np.zeros((self.N, self.n))
        self._rho = np.zeros((self.N, self.sigma))
        self._psi = np.zeros((self.N, self.sigma))
        for i, tkn in enumerate(self.asset_list):
            if tkn in self.omnipool.asset_list:
                j = self.omnipool.asset_list.index(tkn)
                self._o[i, j] = 1
        self._share_indices = []
        offset = 0
        for amm in self.amm_list:
            i = self.asset_list.index(amm.unique_id)
            self._share_indices.append(offset)
            self._rho[i, offset] = 1
            for k, tkn in enumerate(amm.asset_list):
                i = self.asset_list.index(tkn)
                self._psi[i, offset + k + 1] = 1
            offset += len(amm.asset_list) + 1

    def _set_omnipool_coefs(self):
        self._omnipool_lrna_coefs = {tkn: self._scaling["LRNA"] / self.omnipool.lrna[tkn] for tkn in self.omnipool.asset_list}
        self._omnipool_asset_coefs = {tkn: self._scaling[tkn] / self.omnipool.liquidity[tkn] for tkn in self.omnipool.asset_list}

    def _set_coefficients(self):
        buffer_fee = 0.00001  # 0.1 bp buffer fee
        # buffer_fee = 0.0  # TODO: remove this
        # profit calculations
        # variables are y_i, x_i, lrna_lambda_i, lambda_i, d_j, I_l
        # y_i are net LRNA into Omnipool
        profit_lrna_y_coefs = -np.ones(self.n)
        # x_i are net assets into Omnipool
        profit_lrna_x_coefs = np.zeros(self.n)
        # lrna_lambda_i are LRNA amounts coming out of Omnipool
        profit_lrna_lrna_lambda_coefs = -np.array([self.omnipool.last_lrna_fee[tkn] for tkn in self.omnipool.asset_list])
        profit_lrna_lambda_coefs = np.zeros(self.n)
        profit_lrna_X_coefs = np.zeros(self.sigma)
        profit_lrna_L_coefs = np.zeros(self.sigma)
        profit_lrna_a_coefs = np.zeros(self.u)
        profit_lrna_d_coefs = self._tau[0, :self.m].flatten()
        sell_amts = np.array([intent['sell_quantity'] for intent in self.full_intents])
        profit_lrna_I_coefs = self._tau[0, self.m:].flatten() * sell_amts / self._scaling["LRNA"]
        profit_lrna_coefs = np.concatenate([
            profit_lrna_y_coefs, profit_lrna_x_coefs, profit_lrna_lrna_lambda_coefs, profit_lrna_lambda_coefs,
            profit_lrna_X_coefs, profit_lrna_L_coefs, profit_lrna_a_coefs, profit_lrna_d_coefs, profit_lrna_I_coefs
        ])

        # leftover must be higher than required fees
        # other assets
        tkn_list = ["LRNA"] + self.asset_list
        fees = [self.omnipool.last_fee[tkn] + buffer_fee for tkn in self.omnipool.asset_list]
        stableswap_fees = [[amm.trade_fee + buffer_fee]*(len(amm.asset_list) + 1) for amm in self.amm_list]
        stableswap_fees_flat = [item - self.fee_match for sublist in stableswap_fees for item in sublist]
        partial_intent_prices = self.get_partial_intent_prices()
        profit_y_coefs = np.zeros((self.N, self.n))
        profit_x_coefs = np.zeros((self.N, self.n))
        profit_lrna_lambda_coefs = np.zeros((self.N, self.n))
        # profit_lambda_coefs = -np.diag(np.array(fees).astype(float) - self.fee_match)
        profit_lambda_coefs = np.zeros((self.N, self.n))
        for i, tkn in enumerate(self.asset_list):
            if tkn in self.omnipool.asset_list:
                j = self.omnipool.asset_list.index(tkn)
                profit_x_coefs[i, j] = -1
                profit_lambda_coefs[i, j] = self.fee_match - fees[j]
        profit_X_coefs = self._rho - self._psi
        profit_L_coefs = self._psi @ np.diag(-np.array(stableswap_fees_flat))
        profit_L_coefs_old = np.zeros((self.N, self.sigma))  # TODO: add fees for AMMs
        profit_a_coefs = np.zeros((self.N, self.u))
        scaling_vars = np.array([partial_intent_prices[j] * self._scaling[intent['tkn_sell']] / self._scaling[intent['tkn_buy']]
                                 for j, intent in enumerate(self.partial_intents)])
        scaled_phi = self._phi[1:, :self.m] * scaling_vars * 1 / (1 - self.fee_match)
        profit_d_coefs = (self._tau[1:, :self.m] - scaled_phi).astype(float)

        buy_amts = np.array([intent['buy_quantity'] for intent in self.full_intents])
        sell_amts = np.array([intent['sell_quantity'] for intent in self.full_intents])
        scaled_phi = self._phi[1:, self.m:] * buy_amts * 1 / (1 - self.fee_match)
        scaled_tau = self._tau[1:, self.m:] * sell_amts
        unscaled_diff = scaled_tau - scaled_phi
        scalars = np.array([self._scaling[tkn] for tkn in self.asset_list])
        I_coefs = (unscaled_diff / scalars[:, np.newaxis]).astype(float)
        profit_A_LRNA = np.array([profit_lrna_coefs.astype(float)])
        profit_A_assets = np.hstack(
            [profit_y_coefs, profit_x_coefs, profit_lrna_lambda_coefs, profit_lambda_coefs, profit_X_coefs,
             profit_L_coefs, profit_a_coefs, profit_d_coefs, I_coefs]
        )
        self._profit_A = np.vstack([profit_A_LRNA, profit_A_assets])

        profit_i = self.asset_list.index(self.tkn_profit)
        self._q = self._profit_A[profit_i + 1, :].flatten()

        self._S = np.array([self._scaling[tkn] for tkn in self.asset_list])
        self._C = self._rho.T @ self._S
        self._B = self._psi.T @ self._S

    def _recalculate(self, rescale: bool = True):
        self._set_known_flow()
        self._set_max_in_out()
        # self._set_bounds()
        if rescale:
            self._set_scaling()
            self._set_omnipool_coefs()
        self._set_directions()
        self._set_tau_phi()
        self._set_coefficients()

    def set_up_problem(
            self,
            I: list = None,
            omnipool_flags: dict = None,
            sell_maxes: list = None,
            force_omnipool_approx: dict = None,
            rescale: bool = True,
            clear_sell_maxes: bool = True,
            clear_I: bool = True,
            clear_omnipool_approx: bool = True,
            force_amm_approx: list = None,
            clear_amm_approx: bool = True,
            omnipool_deltas: dict = None,
            amm_deltas: list = None,
            amm_flags: dict = None
    ):
        if I is not None:
            assert len(I) == len(self.full_intents)
            self.I = I
        elif clear_I:
            self.I = None
        if sell_maxes is not None:
            self.partial_sell_maxs = sell_maxes
        elif clear_sell_maxes:
            self.partial_sell_maxs = [intent['sell_quantity'] for intent in self.partial_intents]
        if omnipool_flags is None:
            self._omnipool_directional_flags = {}
        else:
            self._omnipool_directional_flags = omnipool_flags
        if amm_flags is None:
            self._amm_directional_flags = {}
        else:
            self._amm_directional_flags = amm_flags
        if force_omnipool_approx is not None:
            self._force_omnipool_approx = force_omnipool_approx
        elif clear_omnipool_approx:
            self._force_omnipool_approx = {tkn: "none" for tkn in self.omnipool.asset_list}
        if force_amm_approx is not None:
            self._force_amm_approx = force_amm_approx
        elif clear_amm_approx:
            self._force_amm_approx = [["none"]*(len(amm.asset_list) + 1) for amm in self.amm_list]
        if omnipool_deltas is not None:
            self._last_omnipool_deltas = omnipool_deltas
        if amm_deltas is not None:
            self._last_amm_deltas = amm_deltas
        self._recalculate(rescale)

    def get_scaling(self):
        return {k: v for k, v in self._scaling.items()}

    def get_omnipool_lrna_coefs(self):
        return {k: v for k, v in self._omnipool_lrna_coefs.items()}

    def get_omnipool_asset_coefs(self):
        return {k: v for k, v in self._omnipool_asset_coefs.items()}

    def get_q(self):
        return np.array([v for v in self._q])

    def get_profit_A(self):
        return np.copy(self._profit_A)

    def get_C(self):
        return np.array([v for v in self._C])

    def get_B(self):
        return np.array([v for v in self._B])

    def get_share_indices(self):
        return [v for v in self._share_indices]

    def get_directions(self):
        return {k: v for k, v in self._omnipool_directions.items()}, copy.deepcopy(self._amm_directions)

    def get_epsilon_tkn(self):
        return {t: max([abs(self._max_in[t]), abs(self._max_out[t])]) / self.omnipool.liquidity[t] for t in self.omnipool.asset_list}

    def get_fees(self):
        return [self.omnipool.last_fee[tkn] for tkn in self.asset_list]

    def get_lrna_fees(self):
        return [self.omnipool.last_lrna_fee[tkn] for tkn in self.asset_list]

    def get_partial_intent_prices(self):
        partial_intent_prices = [intent['buy_quantity'] / intent['sell_quantity'] if intent['sell_quantity'] > 0 else 0
                                 for intent in self.partial_intents]
        return partial_intent_prices

    def get_partial_sell_maxs_scaled(self):
        partial_sell_maxes = [m for m in self.partial_sell_maxs]
        for j, intent in enumerate(self.partial_intents):
            tkn = intent['tkn_sell']
            if tkn in self.omnipool.asset_list:
                partial_sell_maxes[j] = min([partial_sell_maxes[j], self.omnipool.liquidity[tkn] / 2])
        return [partial_sell_maxes[j] / self._scaling[intent['tkn_sell']] for j, intent in enumerate(self.partial_intents)]

    def get_omnipool_approx(self, tkn):
        return self._force_omnipool_approx[tkn]

    def get_amm_approx(self, i):
        return self._force_amm_approx[i]

    def scale_obj_amt(self, amt):
        return amt * self._scaling[self.tkn_profit]

    # def get_scaled_bounds(self):
    #     scaled_min_y = self._min_y / self._scaling["LRNA"]
    #     scaled_max_y = self._max_y / self._scaling["LRNA"]
    #     scaled_min_x = [self._min_x[i] / self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
    #     scaled_max_x = [self._max_x[i] / self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
    #     scaled_min_lrna_lambda = self._min_lrna_lambda / self._scaling["LRNA"]
    #     scaled_max_lrna_lambda = self._max_lrna_lambda / self._scaling["LRNA"]
    #     scaled_min_lambda = [self._min_lambda[i] / self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
    #     scaled_max_lambda = [self._max_lambda[i] / self._scaling[tkn] for i, tkn in enumerate(self.asset_list)]
    #     return scaled_min_y, scaled_max_y, scaled_min_x, scaled_max_x, scaled_min_lrna_lambda, scaled_max_lrna_lambda, scaled_min_lambda, scaled_max_lambda

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
        N, sigma, s, u = self.N, self.sigma, self.s, self.u
        assert len(x) in [4 * n + 2 * sigma + u + m, 4 * n + 2 * sigma + m + r]
        scaled_yi = [x[i] * self._scaling["LRNA"] for i in range(n)]
        scaled_xi = [x[n + i] * self._scaling[tkn] for i, tkn in enumerate(self.omnipool.asset_list)]
        scaled_lrna_lambda = [x[2*n + i] * self._scaling["LRNA"] for i in range(n)]
        scaled_lambda = [x[3 * n + i] * self._scaling[tkn] for i, tkn in enumerate(self.omnipool.asset_list)]
        X_scaling = (self._rho + self._psi).T @ self._S
        scaled_X = x[4 * n: 4 * n + sigma] * X_scaling
        scaled_L = x[4 * n + sigma: 4 * n + 2 * sigma] * X_scaling
        if len(x) == 4 * n + 2 * sigma + m + r:  # TODO improve this logic
            scaled_d = [x[4 * n + 2 * sigma + j] * self._scaling[intent['tkn_sell']] for j, intent in
                        enumerate(self.partial_intents)]
            scaled_I = [x[4 * n + m + l] for l in range(r)]
            scaled_x = np.concatenate([scaled_yi, scaled_xi, scaled_lrna_lambda, scaled_lambda, scaled_X, scaled_L,
                                       scaled_d, scaled_I])
            scaled_x = np.concatenate([scaled_x, scaled_I])
        else:
            scaled_d = [x[4 * n + 2 * sigma + u + j] * self._scaling[intent['tkn_sell']] for j, intent in
                        enumerate(self.partial_intents)]
            scaled_a = x[4 * n + 2 * sigma: 4 * n + 2 * sigma + u]
            scaled_x = np.concatenate([scaled_yi, scaled_xi, scaled_lrna_lambda, scaled_lambda, scaled_X, scaled_L,
                                       scaled_a, scaled_d])
        return scaled_x

    def get_scaled_x(self, x):
        n, m, r, sigma = self.n, self.m, self.r, self.sigma
        assert len(x) in [4 * n + 3 * sigma + m, 4 * n + 3 * sigma + m + r]
        scaled_yi = [x[i] / self._scaling["LRNA"] for i in range(n)]
        scaled_xi = [x[n + i] / self._scaling[tkn] for i, tkn in enumerate(self.omnipool.asset_list)]
        scaled_lrna_lambda = [x[2*n + i] / self._scaling["LRNA"] for i in range(n)]
        scaled_lambda = [x[3 * n + i] / self._scaling[tkn] for i, tkn in enumerate(self.omnipool.asset_list)]
        stableswap_scalars = (self._rho + self._psi).T @ self._S
        scaled_X = x[4 * n: 4 * n + sigma] / stableswap_scalars
        scaled_L = x[4 * n + sigma: 4 * n + 2 * sigma] / stableswap_scalars
        scaled_a = x[4 * n + 2 * sigma: 4 * n + 3 * sigma]
        scaled_d = [x[4 * n + 3 * sigma + j] / self._scaling[intent['tkn_sell']] for j, intent in enumerate(self.partial_intents)]
        scaled_x = np.concatenate([scaled_yi, scaled_xi, scaled_lrna_lambda, scaled_lambda, scaled_X, scaled_L, scaled_a, scaled_d])
        if len(x) == 4 * n + 3 * sigma + m + r:
            scaled_I = [x[4 * n + m + l] for l in range(r)]
            scaled_x = np.concatenate([scaled_x, scaled_I])
        return scaled_x

    def get_intents(self):
        new_intents = []
        for intent in self.intents:
            new_intent = {k: v for k, v in intent.items() if k != 'agent'}
            new_intent['agent'] = intent['agent'].copy()
            new_intents.append(new_intent)
        return new_intents


def scale_down_partial_intents(p, trade_pcts: list, scale: float):
    zero_ct = 0
    intent_sell_maxs = [m for m in p.partial_sell_maxs]
    for i, m in enumerate(p.partial_sell_maxs):
        # we allow new solution to find trade size up to 10x old solution
        old_sell_quantity = m * trade_pcts[i]
        new_sell_max = m / scale
        if old_sell_quantity < new_sell_max:
            tkn = p.partial_intents[i]['tkn_sell']
            sell_amt_lrna_value = new_sell_max * p.omnipool.price(p.omnipool, tkn)
            # if we are scaling lower than min_partial, we eliminate the intent from execution
            if sell_amt_lrna_value < p.min_partial:
                new_sell_max = 0
                zero_ct += 1  # we count the number of intents that are eliminated
            intent_sell_maxs[i] = new_sell_max
    return intent_sell_maxs, zero_ct


def _get_leftover_bounds(p, allow_loss, indices_to_keep=None):
    k = 4 * p.n + 2 * p.sigma + p.u + p.m
    profit_A = p.get_profit_A()
    A3 = -profit_A[:, :k]
    I_coefs = -profit_A[:, k:]
    # if we want to allow a loss in tkn_profit, we remove the appropriate row
    if allow_loss:
        profit_i = p.asset_list.index(p.tkn_profit)
        A3 = np.delete(A3, profit_i+1, axis=0)
        I_coefs = np.delete(I_coefs, profit_i+1, axis=0)
    A3_trimmed = A3[:, indices_to_keep] if indices_to_keep is not None else A3
    if p.r == 0:
        b3 = np.zeros(A3_trimmed.shape[0])
    else:
        b3 = -I_coefs @ p.I
    return A3_trimmed, b3


def _get_stableswap_bounds(p, indices_to_keep=None):
    # CFMM invariants must be respected
    n, sigma, u, m, N = p.n, p.sigma, p.u, p.m, p.N
    k = 4 * n + 2 * sigma + u + m

    A5 = np.zeros((0, k))
    b5 = np.array([])
    cones5 = []
    C = p.get_C()
    B = p.get_B()
    share_indices = p.get_share_indices()
    for j, amm in enumerate(p.amm_list):
        if not isinstance(amm, StableSwapPoolState):
            raise
        l = share_indices[j]
        ann = amm.ann
        s0 = amm.shares
        D0 = amm.d
        n_amm = len(amm.asset_list)
        sum_assets = sum([amm.liquidity[tkn] for tkn in amm.asset_list])
        # TODO: think about indexing of auxiliary variables
        approx = p.get_amm_approx(j)
        # D0' = D_0 * (1 - 1/ann)
        D0_prime = D0 * (1 - 1 / ann)
        # a0 ~= -delta_s/s0 + [1 / (sum x_i^0 - D0') * sum delta_x_i - (D0'/s0) / (sum x_i^0 - D0') * delta_s]
        denom = sum_assets - D0_prime
        if approx[0] == "linear":
            A5j = np.zeros((1, k))
            A5j[0, 4 * n + 2 * sigma + l] = 1  # a_{j,0} coefficient
            A5j[0, 4 * n + l] = (1 + D0_prime / denom) * C[l] / s0  # delta_s coefficient
            for t in range(1, n_amm + 1):
                A5j[0, 4*n + l + t] = -B[l+t] / denom  # delta_x_i coefficient
            b5j = np.array([0])
            cones5.append(cb.ZeroConeT(1))
        else:
            A5j = np.zeros((3, k))
            b5j = np.array([0, 0, 0])
            # x = a_{j,0}
            A5j[0, 4*n + 2*sigma + l] = -1
            # y = 1 + C_jS_j / s_0
            A5j[1, 4*n + l] = -C[l] / s0
            b5j[1] = 1
            # z = An^n / D_0 sum(x_i^0 + B_i X_i) + (1 - An^n)(1 + C_jS_j / s_0)
            A5j[2, 4 * n + l] = D0_prime * C[l] / denom / s0
            for t in range(1, n_amm + 1):
                A5j[2, 4*n + l + t] = -B[l+t] / denom
            b5j[2] = 1
            cones5.append(cb.ExponentialConeT())

        for t in range(1, n_amm + 1):
            x0 = amm.liquidity[amm.asset_list[t - 1]]
            if approx[t] == "linear":
                A5jt = np.zeros((1, k))
                A5jt[0, 4 * n + 2 * sigma + l + t] = 1  # a_{j,0} coefficient
                A5jt[0, 4 * n + l] = C[l] / s0  # delta_s coefficient
                A5jt[0, 4 * n + l + t] = -B[l + t] / x0  # delta_x_i coefficient
                b5jt = np.array([0])
                cone5jt = cb.ZeroConeT(1)
            else:
                A5jt = np.zeros((3, k))
                b5jt = np.zeros(3)
                # x = a_{j,t}
                A5jt[0, 4 * n + 2 * sigma + l + t] = -1
                # y = 1 + C_jS_j / s_0
                A5jt[1, 4 * n + l] = -C[l] / s0
                b5jt[1] = 1
                # z = (x_t^0 + B_t X_t) / D_0
                A5jt[2, 4 * n + l + t] = -B[l + t] / x0
                b5jt[2] = 1
                cone5jt = cb.ExponentialConeT()
            cones5.append(cone5jt)
            A5j = np.vstack([A5j, A5jt])
            b5j = np.append(b5j, np.array(b5jt))

        A5j_final = np.zeros((1, k))
        # coef = 0
        # for tkn in amm.asset_list:
        #     coef += np.log(n_amm * amm.liquidity[tkn] / D0)
        # coef += np.log(c)
        # A5j_final[0, 4 * n + l] = -coef * C[l] / s0
        for t in range(n_amm + 1):
            A5j_final[0, 4 * n + 2 * sigma + l + t] = -1
        # b5j_final = np.array([coef])
        b5j_final = np.array([0])
        cones5.append(cb.NonnegativeConeT(1))

        A5 = np.vstack([A5, A5j, A5j_final])
        b5 = np.concatenate([b5, b5j, b5j_final])
    return A5[:, indices_to_keep], b5, cones5


def _find_solution_unrounded(
        p: ICEProblem,
        allow_loss: bool = False
) -> (dict, list):

    if p.I is None:
        raise
    I = p.I
    # if sum(I) + sum(p.partial_sell_maxs) == 0:  # nothing to execute
    #     return {tkn: 0 for tkn in p.asset_list}, [0] * len(p.partial_intents), np.zeros(4 * p.n + p.m), 0, 0, 'Solved'

    full_intents, partial_intents, state = p.full_intents, p.partial_intents, p.omnipool
    amm_list = p.amm_list

    # assets involved trades for which execution is being solved
    asset_list = p.asset_list
    n, m, r = p.n, p.m, p.r
    N, sigma, s, u = p.N, p.sigma, p.s, p.u
    # if len(partial_intents) + sum(I) == 0:  # nothing for solver to do
    #     return {tkn: 0 for tkn in asset_list}, [], np.zeros(4*n), 0, 0, 'Solved'  # TODO enable solver with m=0

    omnipool_directions, amm_directions = p.get_directions()
    # directions = {}
    k = 4 * n + 2 * sigma + u + m

    indices_to_keep = list(range(k))
    # for tkn in directions:
    #     if directions[tkn] in ["sell", "neither"]:
    #         indices_to_keep.remove(2 * n + asset_list.index(tkn))  # lrna_lambda_i is zero
    #     if directions[tkn] in ["buy", "neither"]:
    #         indices_to_keep.remove(3 * n + asset_list.index(tkn))  # lambda_i is zero
    #     if directions[tkn] == "neither":
    #         indices_to_keep.remove(asset_list.index(tkn))  # y_i is zero
    #         indices_to_keep.remove(n + asset_list.index(tkn))  # x_i is zero

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    k_real = len(indices_to_keep)
    P_trimmed = sparse.csc_matrix((k_real, k_real))
    q_all = p.get_q()
    objective_I_coefs = -q_all[k:]
    q = -q_all[:k]
    q_trimmed = np.array([q[i] for i in indices_to_keep])

    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#


    A1 = np.zeros((0, k))
    cones1 = []
    profit_i = p.asset_list.index(p.tkn_profit)
    op_tradeable_indices = [i for i in range(n) if p.omnipool.asset_list[i] in p.trading_tkns]
    if profit_i not in op_tradeable_indices:
        op_tradeable_indices.append(profit_i)
    for i in range(n):
        tkn = p.omnipool.asset_list[i]
        if tkn in omnipool_directions and tkn in p._last_omnipool_deltas:
            delta_pct = p._last_omnipool_deltas[tkn] / p.omnipool.liquidity[tkn]  # possibly round to zero
        else:
            delta_pct = 1  # to avoid causing any rounding
        if i in op_tradeable_indices and abs(delta_pct) > 1e-11:  # we need lambda_i >= 0, lrna_lambda_i >= 0
            A1i = np.zeros((2, k))
            A1i[0, 3 * n + i] = -1  # lambda_i
            A1i[1, 2 * n + i] = -1  # lrna_lambda_i
            cone1i = cb.NonnegativeConeT(2)
            if p.omnipool.asset_list[i] in omnipool_directions:
                if omnipool_directions[p.omnipool.asset_list[i]] == "buy":  # we need y_i <= 0, x_i >= 0
                    A1i_dir = np.zeros((2, k))
                    A1i_dir[0, i] = 1
                    A1i_dir[1, n + i] = -1
                    A1i = np.vstack([A1i, A1i_dir])
                    cone1i = cb.NonnegativeConeT(4)
                elif omnipool_directions[p.omnipool.asset_list[i]] == "sell":  # we need y_i >= 0, x_i <= 0
                    A1i_dir = np.zeros((2, k))
                    A1i_dir[0, i] = -1
                    A1i_dir[1, n + i] = 1
                    A1i = np.vstack([A1i, A1i_dir])
                    cone1i = cb.NonnegativeConeT(4)

        else:  # we need y_i = 0, x_i = 0, lambda_i = 0, lrna_lambda_i = 0
            A1i = np.zeros((4, k))
            A1i[0, i] = 1  # y_i
            A1i[1, n + i] = 1  # x_i
            A1i[2, 2 * n + i] = 1  # lrna_lambda_i
            A1i[3, 3 * n + i] = 1  # lambda_i
            cone1i = cb.ZeroConeT(4)
        A1 = np.vstack([A1, A1i])
        cones1.append(cone1i)

    offset = 0
    for i, amm in enumerate(amm_list):
        if len(amm_directions) > 0 and len(p._last_amm_deltas) > 0:
            delta_pct = p._last_amm_deltas[i][0] / amm.shares  # possibly round to zero
        else:
            delta_pct = 1  # to avoid causing any rounding
        if amm.unique_id in p.trading_tkns and abs(delta_pct) > 1e-11:
            A1i = np.zeros((1, k))
            A1i[0, 4 * n + sigma + offset] = -1
            cones1.append(cb.NonnegativeConeT(1))
            if len(amm_directions) > 0:
                if amm_directions[i][0] == "buy":  # X0 >= 0
                    A1i_dir = np.zeros((1, k))
                    A1i_dir[0, 4 * n + offset] = -1
                    A1i = np.vstack([A1i, A1i_dir])
                    cones1.append(cb.NonnegativeConeT(1))
                elif amm_directions[i][0] == "sell":  # X0 <= 0
                    A1i_dir = np.zeros((1, k))
                    A1i_dir[0, 4 * n + offset] = 1
                    A1i = np.vstack([A1i, A1i_dir])
                    cones1.append(cb.NonnegativeConeT(1))
        else:
            A1i = np.zeros((2, k))
            A1i[0, 4 * n + offset] = 1
            A1i[1, 4 * n + sigma + offset] = 1
            cones1.append(cb.ZeroConeT(2))
        for j, tkn in enumerate(amm.asset_list):
            if len(amm_directions) > 0 and len(p._last_amm_deltas) > 0:
                delta_pct = p._last_amm_deltas[i][j+1] / amm.liquidity[tkn]  # possibly round to zero
            else:
                delta_pct = 1  # to avoid causing any rounding
            if tkn in p.trading_tkns and abs(delta_pct) > 1e-11:
                A1ij = np.zeros((1, k))
                A1ij[0, 4 * n + sigma + offset + j + 1] = -1
                cones1.append(cb.NonnegativeConeT(1))
                if len(amm_directions) > 0:
                    if amm_directions[i][j+1] == "buy":  #Xj >= 0
                        A1ij_dir = np.zeros((1, k))
                        A1ij_dir[0, 4 * n + offset + j + 1] = -1
                        A1ij = np.vstack([A1ij, A1ij_dir])
                        cones1.append(cb.NonnegativeConeT(1))
                    elif amm_directions[i][j+1] == "sell":  #Xj <= 0
                        A1ij_dir = np.zeros((1, k))
                        A1ij_dir[0, 4 * n + offset + j + 1] = 1
                        A1ij = np.vstack([A1ij, A1ij_dir])
                        cones1.append(cb.NonnegativeConeT(1))
            else:
                A1ij = np.zeros((2, k))
                A1ij[0, 4 * n + offset + j + 1] = 1
                A1ij[1, 4 * n + sigma + offset + j + 1] = 1
                cones1.append(cb.ZeroConeT(2))
            A1i = np.vstack([A1i, A1ij])
        A1 = np.vstack([A1, A1i])
        offset += len(amm.asset_list) + 1

    A1_trimmed = A1[:, indices_to_keep]
    b1 = np.zeros(A1_trimmed.shape[0])

    # intent variables are constrained from above, and from below by 0
    amm_coefs = np.zeros((2 * m, k - m))
    d_coefs = np.vstack([np.eye(m), -np.eye(m)])
    A2 = np.hstack([amm_coefs, d_coefs])
    b2 = np.concatenate([np.array(p.get_partial_sell_maxs_scaled()), np.zeros(m)])
    A2_trimmed = A2[:, indices_to_keep]
    cone2 = cb.NonnegativeConeT(A2_trimmed.shape[0])

    A3_trimmed, b3 = _get_leftover_bounds(p, allow_loss, indices_to_keep)
    cone3 = cb.NonnegativeConeT(A3_trimmed.shape[0])

    # Omnipool invariants must not go down
    omnipool_lrna_coefs = p.get_omnipool_lrna_coefs()
    omnipool_asset_coefs = p.get_omnipool_asset_coefs()
    A4 = np.zeros((0, k))
    b4 = np.array([])
    cones4 = []
    epsilon_tkn = p.get_epsilon_tkn()
    for i in range(n):
        tkn = p.omnipool.asset_list[i]
        approx = p.get_omnipool_approx(tkn)
        # if approx == "none" and epsilon_tkn[tkn] <= 1e-6 and tkn != p.tkn_profit:
        #     approx = "linear"
        # elif approx == "none" and epsilon_tkn[tkn] <= 1e-3:
        #     approx = "quadratic"
        if tkn == p.tkn_profit:
            approx = "none"
        if approx == "linear":  # linearize the AMM constraint
            if tkn not in omnipool_directions:
                c1 = 1 / (1 + epsilon_tkn[tkn])
                c2 = 1 / (1 - epsilon_tkn[tkn])
                A4i = np.zeros((2, k))
                b4i = np.zeros(2)
                A4i[0, i] = -omnipool_lrna_coefs[tkn]
                A4i[0, n + i] = -omnipool_asset_coefs[tkn] * c1
                A4i[1, i] = -omnipool_lrna_coefs[tkn]
                A4i[1, n + i] = -omnipool_asset_coefs[tkn] * c2
                cones4.append(cb.NonnegativeConeT(2))
            else:
                if omnipool_directions[tkn] == "sell":
                    c = 1 / (1 - epsilon_tkn[tkn])
                else:
                    c = 1 / (1 + epsilon_tkn[tkn])
                A4i = np.zeros((1, k))
                b4i = np.zeros(1)
                A4i[0, i] = -omnipool_lrna_coefs[tkn]
                A4i[0, n+i] = -omnipool_asset_coefs[tkn] * c
                cones4.append(cb.ZeroConeT(1))
            # # if approx is linear, we need to apply some constraints to x_i, y_i
            # A4i_bounds = np.zeros((4, k))
            # b4i_bounds = np.zeros(4)
            # # y_i is bounded by [-lrna[tkn], lrna[tkn]]/2
            # max_lrna_delta = p.omnipool.lrna[tkn] / 2
            # A4i_bounds[0, i] = 1
            # b4i_bounds[0] = max_lrna_delta / p._scaling["LRNA"]
            # A4i_bounds[1, i] = -1
            # b4i_bounds[1] = max_lrna_delta / p._scaling["LRNA"]
            # # x_i is bounded by [-liquidity[tkn], liquidity[tkn]]/2
            # max_asset_delta = p.omnipool.liquidity[tkn] / 2
            # A4i_bounds[2, n + i] = 1
            # b4i_bounds[2] = max_asset_delta / p._scaling[tkn]
            # A4i_bounds[3, n + i] = -1
            # b4i_bounds[3] = max_asset_delta / p._scaling[tkn]
            # A4i = np.vstack([A4i, A4i_bounds])
            # b4i = np.append(b4i, b4i_bounds)
            # cones4.append(cb.NonnegativeConeT(4))
        elif approx == "quadratic":  # quadratic approximation to in-given-out function
            A4i = np.zeros((3, k))
            A4i[1,i] = -omnipool_lrna_coefs[tkn]
            A4i[1,n+i] = -omnipool_asset_coefs[tkn]
            A4i[2,n+i] = -omnipool_asset_coefs[tkn]
            b4i = np.array([1, 0, 0])
            cones4.append(cb.PowerConeT(0.5))
        else:  # full AMM constraint
            A4i = np.zeros((3, k))
            b4i = np.ones(3)
            A4i[0, i] = -omnipool_lrna_coefs[tkn]
            A4i[1, n + i] = -omnipool_asset_coefs[tkn]
            cones4.append(cb.PowerConeT(0.5))
        A4 = np.vstack([A4, A4i])
        b4 = np.append(b4, b4i)
    A4_trimmed = A4[:, indices_to_keep]

    # CFMM invariants must be respected
    A5_trimmed, b5, cones5 = _get_stableswap_bounds(p, indices_to_keep)

    # inequality constraints on comparison of lrna_lambda to yi, lambda to xi
    A6 = np.zeros((0, k))
    for i in range(n):
        A6i = np.zeros((2, k))
        A6i[0, i] = -1  # lrna_lambda + yi >= 0
        A6i[0, 2*n+i] = -1  # lrna_lambda + yi >= 0
        A6i[1, n+i] = -1  # lambda + xi >= 0
        A6i[1, 3*n+i] = -1  # lambda + xi >= 0
        A6 = np.vstack([A6, A6i])

    A6_trimmed = A6[:, indices_to_keep]

    b6 = np.zeros(A6.shape[0])
    cone6 = cb.NonnegativeConeT(A6.shape[0])

    # inequality constraints: X_j + L_j >= 0
    A7 = np.zeros((0, k))
    for i in range(sigma):
        A7i = np.zeros((1, k))
        A7i[0, 4*n+i] = -1
        A7i[0, 4*n+sigma+i] = -1
        A7 = np.vstack([A7, A7i])
    A7_trimmed = A7[:, indices_to_keep]
    b7 = np.zeros(A7.shape[0])
    cone7 = cb.NonnegativeConeT(A7.shape[0])

    # # we will temporarily require that L_j == 0
    # A7 = np.zeros((0, k))
    # for i in range(sigma):
    #     A7i = np.zeros((1, k))
    #     A7i[0, 4*n+sigma+i] = -1
    #     A7 = np.vstack([A7, A7i])
    # A7_trimmed = A7[:, indices_to_keep]
    # b7 = np.zeros(A7.shape[0])
    # cone7 = cb.ZeroConeT(A7.shape[0])

    A = np.vstack([A1_trimmed, A2_trimmed, A3_trimmed, A4_trimmed, A5_trimmed, A6_trimmed, A7_trimmed])
    A_sparse = sparse.csc_matrix(A)
    b = np.concatenate([b1, b2, b3, b4, b5, b6, b7])
    cones = cones1 + [cone2, cone3] + cones4 + cones5 + [cone6, cone7]

    # solve
    settings = clarabel.DefaultSettings()
    settings.max_step_fraction = 0.95
    solver = clarabel.DefaultSolver(P_trimmed, q_trimmed, A_sparse, b, cones, settings)
    solution = solver.solve()
    x = solution.x
    z = solution.z

    new_omnipool_deltas = {}
    exec_intent_deltas = [None] * len(partial_intents)
    x_expanded = [0] * k
    for i in range(k):
        if i in indices_to_keep:
            x_expanded[i] = x[indices_to_keep.index(i)]
    x_scaled = p.get_real_x(x_expanded)
    # new_omnipool_deltas["LRNA"] = 0
    for i in range(n):
        tkn = asset_list[i]
        new_omnipool_deltas[tkn] = x_scaled[n+i]
        # new_omnipool_deltas["LRNA"] += x_scaled[i]
    amm_deltas = []
    offset = 0
    for amm in p.amm_list:
        deltas = [x_scaled[4*n + offset]]
        for t in range(len(amm.asset_list)):
            deltas.append(x_scaled[4*n + offset + t + 1])
        amm_deltas.append(deltas)
        offset += len(amm.asset_list) + 1

    for j in range(len(partial_intents)):
        exec_intent_deltas[j] = -x_scaled[4 * n + 2 * sigma + u + j]

    obj_offset = objective_I_coefs @ I if I is not None else 0
    score = p.scale_obj_amt(solution.obj_val + obj_offset)
    dual_score = p.scale_obj_amt(solution.obj_val_dual + obj_offset)

    return (new_omnipool_deltas, exec_intent_deltas, x_expanded, score, dual_score, str(solution.status), amm_deltas)


def _find_good_solution(
        p: ICEProblem,
        scale_trade_max: bool = True,
        approx_amm_eqs: bool = True,
        do_directional_run: bool = True,
        allow_loss: bool = False
):
    n, m, r = p.n, p.m, p.r
    N, u, s, sigma = p.N, p.u, p.s, p.sigma
    force_omnipool_approx = {tkn: "linear" for tkn in p.omnipool.asset_list}
    force_amm_approx = [["linear" for _ in range(len(amm.asset_list) + 1)] for amm in p.amm_list]
    p.set_up_problem(clear_I=False, force_omnipool_approx=force_omnipool_approx, force_amm_approx=force_amm_approx)
    omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = _find_solution_unrounded(p, allow_loss=allow_loss)
    # if partial trade size is much higher than executed trade, lower trade max
    if scale_trade_max:
        trade_pcts = [max(-intent_deltas[i],0) / m if m > 0 else 0 for i, m in enumerate(p.partial_sell_maxs)]
    else:
        trade_pcts = [1] * len(p.partial_sell_maxs)
    trade_pcts = trade_pcts + [1 for _ in range(r)]
    # adjust AMM constraint approximation based on percent of Omnipool liquidity traded with AMM
    # omnipool_pcts = {tkn: abs(omnipool_deltas[tkn]) / p.omnipool.liquidity[tkn] for tkn in p.omnipool.asset_list}
    # stableswap_pcts = []
    # for i, amm in enumerate(p.amm_list):
    #     pcts = [abs(amm_deltas[i][0]) / amm.shares]  # first shares size constraint, delta_s / s_0 <= epsilon
    #     sum_delta_x = sum([abs(amm_deltas[i][j+1]) for j in range(len(amm.asset_list))])
    #     # sum_x_0 = sum([amm.liquidity[tkn] for tkn in amm.asset_list])
    #     # delta_D = amm.d / amm.shares * amm_deltas[i][0]
    #     # assert sum_delta_x - delta_D >= 0
    #     # assert sum_x_0 - amm.d >= 0
    #     # second shares size constraint, (sum delta_x - delta_D) / (sum x_0 - D_0 + D_0/ann) <= epsilon
    #     # second shares size constraint, sum_delta_x / D_0 <= epsilon
    #     pcts.append(sum_delta_x / amm.d)
    #     pcts.extend([abs(amm_deltas[i][j+1]) / amm.liquidity[tkn] for j, tkn in enumerate(amm.asset_list)])
    #     stableswap_pcts.append(pcts)

    # force_omnipool_approx = None
    # force_amm_approx = None
    approx_adjusted_ct = 0
    if approx_amm_eqs and status not in ['PrimalInfeasible', 'DualInfeasible']:  # update force_amm_approx if necessary
        omnipool_pcts = {tkn: abs(omnipool_deltas[tkn]) / p.omnipool.liquidity[tkn] for tkn in p.omnipool.asset_list}
        for tkn in p.omnipool.asset_list:
            if force_omnipool_approx[tkn] == "linear" and omnipool_pcts[tkn] > 1e-6:
                force_omnipool_approx[tkn] = "quadratic"  # don't actually want to force linear approximation
                approx_adjusted_ct += 1
            if force_omnipool_approx[tkn] == "quadratic" and omnipool_pcts[tkn] > 1e-3:
                force_omnipool_approx[tkn] = "full"  # don't actually want to force quadratic approximation
                approx_adjusted_ct += 1

        stableswap_pcts = []
        for _i, amm in enumerate(p.amm_list):
            pcts = [abs(amm_deltas[_i][0]) / amm.shares]  # first shares size constraint, delta_s / s_0 <= epsilon
            sum_delta_x = sum([abs(amm_deltas[_i][j + 1]) for j in range(len(amm.asset_list))])
            pcts.append(sum_delta_x / amm.d)
            pcts.extend([abs(amm_deltas[_i][j + 1]) / amm.liquidity[tkn] for j, tkn in enumerate(amm.asset_list)])
            stableswap_pcts.append(pcts)
        for s, amm in enumerate(p.amm_list):
            if force_amm_approx[s][0] == "linear" and max(stableswap_pcts[s][0], stableswap_pcts[s][1]) > 1e-5:
                force_amm_approx[s][0] = "full"
                approx_adjusted_ct += 1
            for j in range(len(amm.asset_list)):  # evaluate each asset constraint
                if force_amm_approx[s][j + 1] == "linear" and stableswap_pcts[s][j + 2] > 1e-5:
                    force_amm_approx[s][j + 1] = "full"
                    approx_adjusted_ct += 1

    for i in range(100):
        # lower maxes for intents
        # trade_pcts_nonzero = [x for x in trade_pcts if x > 0]  # list nonzero trade pcts
        # if trade pcts indicate there are no further trade limits to adjust, and no approximations were adjusted
        # then we have nothing to do
        # what's a better way to detect that there are no further trade limits to adjust?
        # if trade_pct is 0 but max is still nonzero, we may want to lower max to see if this helps
        # so we should be looking at trade_pcts where max is nonzero
        trade_pcts_nonzero_max = [max(-intent_deltas[i],0) / m for i, m in enumerate(p.partial_sell_maxs) if m > 0]

        if (len(trade_pcts_nonzero_max) == 0 or min(trade_pcts_nonzero_max) >= 0.1) and approx_adjusted_ct == 0:
            break  # no changes to problem were made
        if len(trade_pcts_nonzero_max) > 0 and min(trade_pcts_nonzero_max) < 0.1:
            new_maxes, zero_ct = scale_down_partial_intents(p, trade_pcts, 10)
        else:
            new_maxes, zero_ct = None, 0
        # constrain amm variables
        p.set_up_problem(sell_maxes=new_maxes, clear_I=False, force_omnipool_approx=force_omnipool_approx,
                         force_amm_approx=force_amm_approx, omnipool_deltas=omnipool_deltas, amm_deltas=amm_deltas)
        # if zero_ct == m:
        #     break  # all partial intents have been eliminated from execution
        # solve refined problem
        omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = _find_solution_unrounded(p, allow_loss=allow_loss)

        # need to check if amm_deltas stayed within their reasonable approximation bounds
        # if not, we may want to discard the "solution"

        if scale_trade_max:  # update trade_pcts
            trade_pcts = [-intent_deltas[i] / m if m > 0 else 0 for i, m in enumerate(p.partial_sell_maxs)]
            # trade_pcts + [1 for _ in range(r)]
        if approx_amm_eqs and status not in ['PrimalInfeasible', 'DualInfeasible']:  # update force_amm_approx if necessary
            omnipool_pcts = {tkn: abs(omnipool_deltas[tkn]) / p.omnipool.liquidity[tkn] for tkn in p.omnipool.asset_list}
            approx_adjusted_ct = 0
            for tkn in p.omnipool.asset_list:
                if force_omnipool_approx[tkn] == "linear" and omnipool_pcts[tkn] > 1e-6:
                    force_omnipool_approx[tkn] = "quadratic"  # don't actually want to force linear approximation
                    approx_adjusted_ct += 1
                if force_omnipool_approx[tkn] == "quadratic" and omnipool_pcts[tkn] > 1e-3:
                    force_omnipool_approx[tkn] = "full"  # don't actually want to force quadratic approximation
                    approx_adjusted_ct += 1
                    # elif omnipool_pcts[tkn] <= 1e-6:  # force linear
                    #     force_omnipool_approx[tkn] = "linear"
                    #     approx_adjusted_ct += 1
                # else:
                #     if omnipool_pcts[tkn] <= 1e-6:  # force linear
                #         force_omnipool_approx[tkn] = "linear"
                #         approx_adjusted_ct += 1
                #     elif omnipool_pcts[tkn] <= 1e-3:  # force quadratic
                #         force_omnipool_approx[tkn] = "quadratic"
                #         approx_adjusted_ct += 1

            stableswap_pcts = []
            for _i, amm in enumerate(p.amm_list):
                pcts = [abs(amm_deltas[_i][0]) / amm.shares]  # first shares size constraint, delta_s / s_0 <= epsilon
                sum_delta_x = sum([abs(amm_deltas[_i][j + 1]) for j in range(len(amm.asset_list))])
                pcts.append(sum_delta_x / amm.d)
                pcts.extend([abs(amm_deltas[_i][j + 1]) / amm.liquidity[tkn] for j, tkn in enumerate(amm.asset_list)])
                stableswap_pcts.append(pcts)
            for s, amm in enumerate(p.amm_list):
                if force_amm_approx[s][0] == "linear" and max(stableswap_pcts[s][0], stableswap_pcts[s][1]) > 1e-5:
                    force_amm_approx[s][0] = "full"
                    approx_adjusted_ct += 1
                for j in range(len(amm.asset_list)):  # evaluate each asset constraint
                    if force_amm_approx[s][j + 1] == "linear" and stableswap_pcts[s][j + 2] > 1e-5:
                        force_amm_approx[s][j + 1] = "full"
                        approx_adjusted_ct += 1

    # once solution is found, re-run with directional flags
    if do_directional_run:
        omnipool_flags, amm_flags = get_directional_flags(omnipool_deltas, amm_deltas)
        # TODO implement directional logic
        p.set_up_problem(omnipool_flags=omnipool_flags, clear_I=False, clear_sell_maxes=False,
                         clear_omnipool_approx=False, clear_amm_approx=False, omnipool_deltas=omnipool_deltas,
                         amm_deltas=amm_deltas, amm_flags=amm_flags)
        # omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = _find_solution_unrounded(p, allow_loss=allow_loss)
        dir_output = _find_solution_unrounded(p, allow_loss=allow_loss)
        dir_status = dir_output[5]
        if dir_status in ['Solved', 'AlmostSolved']:
        #     TODO replace with better directional run logic
        #     in particular, we should entirely eliminate variables we expect to be zero
            omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = dir_output

    if status in ['PrimalInfeasible', 'DualInfeasible']:
        omnipool_deltas = [0] * n
        intent_deltas = [0] * m
        x = np.zeros(4 * n + m)
        obj, dual_obj = 0, 0
        amm_deltas = [[0]*(len(amm.asset_list) + 1) for amm in p.amm_list]
        return omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas
    # need to rescale, if we scaled solution down to solve it accurately
    x_unscaled = p.get_real_x(x)
    # zero out rounding errors
    for i in range(n):
        tkn = p.omnipool.asset_list[i]
        if min(abs(x_unscaled[i] / p.omnipool.lrna[tkn]), abs(x_unscaled[n + i] / p.omnipool.liquidity[tkn])) < 1e-11:
            x_unscaled[i] = 0
            x_unscaled[n + i] = 0
            x_unscaled[2 * n + i] = 0
            x_unscaled[3 * n + i] = 0
            omnipool_deltas[tkn] = 0
    # if omnipool_deltas["LRNA"] / p.min_partial < 1e-6:
    #     omnipool_deltas["LRNA"] = 0
    offset = 0
    for i, amm_delta in enumerate(amm_deltas):
        if abs(amm_delta[0] / p.amm_list[i].shares) < 1e-11:
            x_unscaled[4 * n + offset] = 0
            x_unscaled[4 * n + sigma + offset] = 0
            x_unscaled[4 * n + 2 * sigma + offset] = 0
            amm_deltas[i][0] = 0
        for j, tkn in enumerate(p.amm_list[i].asset_list):
            if abs(amm_delta[j+1] / p.amm_list[i].liquidity[tkn]) < 1e-11:
                amm_deltas[i][j+1] = 0
                x_unscaled[4 * n + offset + j + 1] = 0
                x_unscaled[4 * n + sigma + offset + j + 1] = 0
                x_unscaled[4 * n + 2 * sigma + offset + j + 1] = 0
        offset += len(p.amm_list[i].asset_list) + 1

    return omnipool_deltas, intent_deltas, x_unscaled, obj, dual_obj, status, amm_deltas


def _solve_inclusion_problem(
        p: ICEProblem,
        x_real_list: np.array = None,  # NLP solution
        upper_bound: float = None,
        lower_bound: float = None,
        old_A = None,
        old_A_upper = None,
        old_A_lower = None
):
    asset_list = p.asset_list
    op_asset_list = p.omnipool.asset_list
    tkn_list = ["LRNA"] + asset_list
    n, m, r, sigma = p.n, p.m, p.r, p.sigma
    k = 4 * n + 3 * sigma + m + r
    N, s = p.N, p.s

    scaling = p.get_scaling()
    x_list = np.apply_along_axis(p.get_scaled_x, axis=1, arr=x_real_list)

    # we start with the 4n + m variables from the initial problem
    # then we add r indicator variables for the r non-partial intents

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

    partial_intent_sell_amts = p.get_partial_sell_maxs_scaled()

    max_lambda_d = {tkn: p.omnipool.liquidity[tkn]/scaling[tkn]/2 for tkn in op_asset_list}
    max_lrna_lambda_d = {tkn: p.omnipool.lrna[tkn]/scaling["LRNA"]/2 for tkn in op_asset_list}
    max_y_d = max_lrna_lambda_d
    min_y_d = {tkn: -x for tkn, x in max_lrna_lambda_d.items()}
    max_x_d = max_lambda_d
    min_x_d = {tkn: -x for tkn, x in max_lambda_d.items()}

    # max_in = {tkn: p._max_in[tkn] for tkn in asset_list + ["LRNA"]}
    # max_out = {tkn: p._max_out[tkn] for tkn in asset_list}
    for tkn in op_asset_list:
        if tkn != p.tkn_profit:
            # max_x_d[tkn] = min(max_in[tkn]/scaling[tkn] * 2, max_x_d[tkn])
            # min_x_d[tkn] = max(-max_out[tkn]/scaling[tkn] * 2, min_x_d[tkn]
            max_lambda_d[tkn] = -min_x_d[tkn]
            # need to bump up max_y for any LRNA sales
            # max_y_unscaled = max_out[tkn] * p.omnipool.lrna[tkn] / (p.omnipool.liquidity[tkn] - max_out[tkn]) + max_in["LRNA"]
            # max_y_d[tkn] = max_y_unscaled / scaling["LRNA"]
            # min_y_d[tkn] = -max_in[tkn] * p.omnipool.lrna[tkn] / (p.omnipool.liquidity[tkn] + max_in[tkn]) / scaling["LRNA"]
            # max_lrna_lambda_d[tkn] = -min_y_d[tkn]
    min_y = np.array([min_y_d[tkn] for tkn in op_asset_list])
    max_y = np.array([max_y_d[tkn] for tkn in op_asset_list])
    min_x = np.array([min_x_d[tkn] for tkn in op_asset_list])
    max_x = np.array([max_x_d[tkn] for tkn in op_asset_list])
    min_lrna_lambda = np.zeros(n)
    max_lrna_lambda = np.array([max_lrna_lambda_d[tkn] for tkn in op_asset_list])
    min_lambda = np.zeros(n)
    max_lambda = np.array([max_lambda_d[tkn] for tkn in op_asset_list])
    # min_y, max_y, min_x, max_x, min_lrna_lambda, max_lrna_lambda, min_lambda, max_lambda = p.get_scaled_bounds()
    # profit_i = asset_list.index(p.tkn_profit)
    # max_x[profit_i] = inf
    # max_y[profit_i] = inf
    # min_lambda[profit_i] = 0
    # min_lrna_lambda[profit_i] = 0

    # min_y = min_y - 1.1 * np.abs(min_y)
    # min_x = min_x - 1.1 * np.abs(min_x)
    # min_lrna_lambda = min_lrna_lambda - 1.1 * np.abs(min_lrna_lambda)
    # min_lambda = min_lambda - 1.1 * np.abs(min_lambda)
    # max_y = max_y + 1.1 * np.abs(max_y)
    # max_x = max_x + 1.1 * np.abs(max_x)
    # max_lrna_lambda = max_lrna_lambda + 1.1 * np.abs(max_lrna_lambda)
    # max_lambda = max_lambda + 1.1 * np.abs(max_lambda)

    # min_lambda = np.zeros(n)
    # min_lrna_lambda = np.zeros(n)

    max_L = np.array([])
    for amm in p.amm_list:
        max_L = np.append(max_L, amm.shares)
        for tkn in amm.asset_list:
            max_L = np.append(max_L, amm.liquidity[tkn])

    B = p.get_B()
    C = p.get_C()
    max_L = max_L / (B + C)

    min_L = np.zeros(sigma)
    min_X = [-x for x in max_L]
    max_X = [inf] * sigma
    min_a = [-inf] * sigma
    max_a = [inf] * sigma

    # max_L = np.zeros(sigma)  # TODO: remove

    lower = np.concatenate([min_y, min_x, min_lrna_lambda, min_lambda, min_X, min_L, min_a, [0] * (m + r)])
    upper = np.concatenate([max_y, max_x, max_lrna_lambda, max_lambda, max_X, max_L, max_a, partial_intent_sell_amts, [1] * r])

    # S = np.zeros((n, k))
    # S_upper = np.zeros(n)
    #
    # # add point at x = 0
    # for i, tkn in enumerate(op_asset_list):
    #     lrna_c = p.get_omnipool_lrna_coefs()
    #     asset_c = p.get_omnipool_asset_coefs()
    #     S[i, i] = -lrna_c[tkn]
    #     S[i, n + i] = -asset_c[tkn]

    S = np.zeros((0, k))
    S_upper = np.zeros(0)
    x_zero = np.zeros(len(x_list[0]))
    for x in np.vstack([x_zero, x_list]):
    # for x in np.vstack([x_zero]):
        for i, tkn in enumerate(op_asset_list):
            S_row = np.zeros((1,k))
            S_row_upper = np.zeros(1)
            lrna_c = p.get_omnipool_lrna_coefs()
            asset_c = p.get_omnipool_asset_coefs()
            grads_yi = -lrna_c[tkn] - lrna_c[tkn] * asset_c[tkn] * x[n+i]
            grads_xi = -asset_c[tkn] - lrna_c[tkn] * asset_c[tkn] * x[i]
            S_row[0, i] = grads_yi
            S_row[0, n+i] = grads_xi
            grad_dot_x = grads_yi * x[i] + grads_xi * x[n+i]
            g_neg = lrna_c[tkn] * x[i] + asset_c[tkn] * x[n+i] + lrna_c[tkn] * asset_c[tkn] * x[i] * x[n+i]
            S_row_upper[0] = grad_dot_x + g_neg
            S = np.vstack([S, S_row])
            S_upper = np.concatenate([S_upper, S_row_upper])

        offset = 0
        for _s, amm in enumerate(p.amm_list):
            D0_prime = amm.d - amm.d/amm.ann
            s0 = amm.shares
            c = C[offset]
            sum_assets = sum([amm.liquidity[tkn] for tkn in amm.asset_list])
            denom = sum_assets - D0_prime
            a0 = x[4*n + 2*sigma + offset]
            X0 = x[4*n + offset]
            exp = np.exp(float(a0 / (1 + (c*X0/s0))))
            term = (c / s0 - c*a0 / (s0 + c)) * exp
            # linearization of shares constraint, i.e. a0
            S_row = np.zeros((1, k))
            grad_s = term + c * D0_prime / (s0 * denom)
            grads_i = [- B[offset + l] / denom for l in range(1, len(amm.asset_list) + 1)]
            grad_a = exp
            S_row[0, 4*n + offset] = grad_s
            S_row[0, 4*n + 2*sigma + offset] = grad_a
            for l, tkn in enumerate(amm.asset_list):
                S_row[0, 4*n + offset + l + 1] = grads_i[l]
            grad_dot_x = grad_s * X0 + grad_a * a0 + sum([grads_i[l] * x[4*n + offset + l + 1] for l in range(len(amm.asset_list))])
            sum_deltas = sum([B[offset + l + 1] * x[4*n + offset + l + 1] for l in range(len(amm.asset_list))])
            g_neg = (1 + c * X0 / s0) * exp - sum_deltas / denom + D0_prime * c * X0 / (denom * s0) - 1
            S_row_upper = np.array([grad_dot_x + g_neg])
            S = np.vstack([S, S_row])
            S_upper = np.concatenate([S_upper, S_row_upper])

            # linearization of asset constraints, i.e. a1, a2, ...
            for l, tkn in enumerate(amm.asset_list):
                S_row = np.zeros((1, k))
                grad_s = term
                grad_a = exp
                grad_x = -B[offset + l + 1] / amm.liquidity[tkn]
                S_row[0, 4*n + offset] = grad_s
                S_row[0, 4*n + 2*sigma + offset + l + 1] = grad_a
                S_row[0, 4*n + offset + l + 1] = grad_x
                ai = x[4*n + 2*sigma + offset + l + 1]
                grad_dot_x = grad_s * X0 + grad_a * ai + grad_x * x[4*n + offset + l + 1]
                g_neg = (1 + c * X0 / s0) * exp - B[offset + l + 1] * x[4*n + offset + l + 1]  / amm.liquidity[tkn] - 1
                S_row_upper = np.array([grad_dot_x + g_neg])
                S = np.vstack([S, S_row])
                S_upper = np.concatenate([S_upper, S_row_upper])

            offset += 1 + len(amm.asset_list)

    S_lower = np.array([-inf]*len(S_upper))

    # need top level Stableswap constraint
    A_amm = np.zeros((p.s, k))
    offset = 0
    for i, amm in enumerate(p.amm_list):
        for j in range(1 + len(amm.asset_list)):
            A_amm[i, 4*n + 2*sigma + offset + j] = 1
        offset += 1 + len(amm.asset_list)
    A_amm_upper = np.array([inf]*p.s)
    A_amm_lower = np.zeros(p.s)

    # asset leftover must be above zero
    A3 = p.get_profit_A()
    A3_upper = np.array([inf]*(N+1))
    A3_lower = np.zeros(N+1)

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

    # inequality constraints: X_j + L_j >= 0
    A7 = np.zeros((sigma, k))
    for i in range(sigma):
        A7[i, 4*n + i] = 1
        A7[i, 4*n + sigma + i] = 1
    A7_upper = np.array([inf] * sigma)
    A7_lower = np.zeros(sigma)
    # A7 = np.zeros((0,k))
    # A7_upper = np.array([])
    # A7_lower = np.array([])

    # optimized value must be lower than best we have so far, higher than lower bound
    A8 = np.zeros((1, k))
    q = p.get_q()
    A8[0, :] = -q
    A8_upper = np.array([upper_bound / scaling[p.tkn_profit]])
    A8_upper = np.array([upper_bound/10 / scaling[p.tkn_profit]])
    A8_lower = np.array([lower_bound / scaling[p.tkn_profit]])

    if old_A is None:
        old_A = np.zeros((0, k))
    if old_A_upper is None:
        old_A_upper = np.array([])
    if old_A_lower is None:
        old_A_lower = np.array([])
    assert len(old_A_upper) == len(old_A_lower) == old_A.shape[0]
    A = np.vstack([old_A, S, A_amm, A3, A5, A7, A8])
    A_upper = np.concatenate([old_A_upper, S_upper, A_amm_upper, A3_upper, A5_upper, A7_upper, A8_upper])
    A_lower = np.concatenate([old_A_lower, S_lower, A_amm_lower, A3_lower, A5_lower, A7_lower, A8_lower])
    # A = np.vstack([S, A_amm, A3, A5, A7, A8])
    # A_upper = np.concatenate([S_upper, A_amm_upper, A3_upper, A5_upper, A7_upper, A8_upper])
    # A_lower = np.concatenate([S_lower, A_amm_lower, A3_lower, A5_lower, A7_lower, A8_lower])
    # A = np.vstack([S, A_amm, A3, A5, A7])
    # A_upper = np.concatenate([S_upper, A_amm_upper, A3_upper, A5_upper, A7_upper])
    # A_lower = np.concatenate([S_lower, A_amm_lower, A3_lower, A5_lower, A7_lower])

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

    lp.col_cost_ = -q
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
    options = h.getOptions()
    options.small_matrix_value = 1e-12
    options.primal_feasibility_tolerance=1e-10
    options.dual_feasibility_tolerance=1e-10
    options.mip_feasibility_tolerance=1e-10
    h.passOptions(options)
    h.run()
    status = h.getModelStatus()
    solution = h.getSolution()
    info = h.getInfo()
    basis = h.getBasis()

    x_expanded = solution.col_value

    new_omnipool_deltas = {}
    exec_partial_intent_deltas = [None] * m

    # new_omnipool_deltas["LRNA"] = 0
    for i, tkn in enumerate(p.omnipool.asset_list):
        new_omnipool_deltas[tkn] = x_expanded[n+i] * scaling[tkn]
        # new_omnipool_deltas["LRNA"] += x_expanded[i] * scaling["LRNA"]

    offset = 0
    new_amm_deltas = []
    for amm in p.amm_list:
        deltas = [x_expanded[4 * n + offset] * scaling[amm.unique_id]]
        for l, tkn in enumerate(amm.asset_list):
            deltas.append(x_expanded[4 * n + offset + l + 1] * scaling[tkn])
        new_amm_deltas.append(deltas)
        offset += len(amm.asset_list) + 1

    for i in range(m):
        exec_partial_intent_deltas[i] = -x_expanded[4 * n + 3*sigma + i] * scaling[p.partial_intents[i]['tkn_sell']]

    exec_full_intent_flags = [1 if x_expanded[4 * n + 3*sigma + m + i] > 0.5 else 0 for i in range(r)]

    save_A = np.vstack([old_A])
    save_A_upper = np.concatenate([old_A_upper])
    save_A_lower = np.concatenate([old_A_lower])

    score = -q @ x_expanded * scaling[p.tkn_profit]

    return new_omnipool_deltas, exec_partial_intent_deltas, exec_full_intent_flags, save_A, save_A_upper, save_A_lower, score, solution.value_valid, status, new_amm_deltas


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


def get_directional_flags(omnipool_deltas: dict, amm_deltas: list) -> tuple:
    omnipool_flags = {}
    for tkn in omnipool_deltas:
        delta = omnipool_deltas[tkn]
        if delta > 0:
            omnipool_flags[tkn] = 1
        elif delta < 0:
            omnipool_flags[tkn] = -1
        else:
            omnipool_flags[tkn] = 0

    amm_flags = []
    for amm in amm_deltas:
        flags = []
        for delta in amm:
            if delta > 0:
                flags.append(1)
            elif delta < 0:
                flags.append(-1)
            else:
                flags.append(0)
        amm_flags.append(flags)
    return omnipool_flags, amm_flags


def find_solution(state: OmnipoolState, intents: list) -> list:
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents)
    flags = get_directional_flags(amm_deltas)
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents, flags = flags)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)


def calc_impact_to_limit(state: OmnipoolState, intent) -> float:
    sell_spot = state.sell_spot(intent['tkn_sell'], intent['tkn_buy'])
    ex_price = intent['buy_quantity'] / intent['sell_quantity']
    slip_limit = 1 - ex_price / sell_spot
    sell_liq = state.liquidity[intent['tkn_sell']] if intent['tkn_sell'] != "LRNA" else state.lrna[intent['tkn_buy']]
    impact = intent['sell_quantity'] / (sell_liq + intent['sell_quantity'])
    return impact / slip_limit


def calc_limit_to_impact(state: OmnipoolState, intent) -> float:
    sell_spot = state.sell_spot(intent['tkn_sell'], intent['tkn_buy'])
    ex_price = intent['buy_quantity'] / intent['sell_quantity']
    slip_limit = 1 - ex_price / sell_spot
    sell_liq = state.liquidity[intent['tkn_sell']] if intent['tkn_sell'] != "LRNA" else state.lrna[intent['tkn_buy']]
    impact = intent['sell_quantity'] / (sell_liq + intent['sell_quantity'])
    return slip_limit / impact


# TODO: extend to other AMMs
def find_initial_solution(state: OmnipoolState, intents: list):
    size_limit = 1e-7  # trades less than this % of liquidity will be eligible for mandatory inclusion
    max_impact_to_limit = 0.1  # to be eligble for mandatory inclusion, price impact to slippage limit must be less than this
    x = []
    for i, intent in enumerate(intents):
        limit_to_impact = calc_limit_to_impact(state, intent)
        if limit_to_impact >= 1:
            x.append((limit_to_impact, i, intent))
    x.sort(reverse=True)  # is now all intents sorted by limit to impact, descending

    omnipool_net = {tkn: 0 for tkn in state.asset_list}
    max_price_movement = 0.0001  # max price movement from mandatory small trades
    mandatory_intents = copy.deepcopy(intents)
    exec_indices = []
    simulated_state = state.copy()

    # include small trades until we get up to 0.5 bp of impact in each asset
    for (impact, i, intent) in x:
        if impact < 1/max_impact_to_limit:
            break
        sell_liq = state.liquidity[intent['tkn_sell']] if intent['tkn_sell'] != "LRNA" else state.lrna[intent['tkn_buy']]
        sell_pct = intent['sell_quantity'] / sell_liq
        buy_liq = state.liquidity[intent['tkn_buy']] if intent['tkn_buy'] != "LRNA" else state.lrna[intent['tkn_sell']]
        buy_pct = intent['buy_quantity'] / buy_liq
        if sell_pct > size_limit or buy_pct > size_limit:
            continue
        if intent['tkn_sell'] in omnipool_net:
            new_net = omnipool_net[intent['tkn_sell']] + intent['sell_quantity']
            if new_net / state.liquidity[intent['tkn_sell']] > max_price_movement / 2:
                continue
        if intent['tkn_buy'] in omnipool_net:
            new_net = omnipool_net[intent['tkn_buy']] - intent['buy_quantity']
            if -new_net / state.liquidity[intent['tkn_buy']] > max_price_movement / 2:
                continue
        if intent['tkn_sell'] != "LRNA":
            omnipool_net[intent['tkn_sell']] += intent['sell_quantity']
        if intent['tkn_buy'] != "LRNA":
            omnipool_net[intent['tkn_buy']] -= intent['buy_quantity']
        mandatory_intents[i]['partial'] = False
        exec_indices.append(i)
        # simulate trade in simulated_state
        simulated_state.swap(intent['agent'].copy(), intent['tkn_buy'], intent['tkn_sell'], buy_quantity=intent['buy_quantity'])

    init_i = [i for i in exec_indices]
    for (impact, i, intent) in x:
        if i in exec_indices:
            continue
        simulated_state.swap(intent['agent'].copy(), intent['tkn_buy'], intent['tkn_sell'], buy_quantity=intent['buy_quantity'])
        if not simulated_state.fail:
            init_i.append(i)
        else:
            simulated_state.fail = ""

    return mandatory_intents, exec_indices, init_i


def _convert_to_all_partial(p: ICEProblem) -> ICEProblem:
    new_intents = []
    for i, intent in enumerate(p.intents):
        new_intent = copy.deepcopy(intent)
        new_intent['partial'] = True
        new_intents.append(new_intent)
    return ICEProblem(p.omnipool, new_intents, init_i=[], min_partial=0)


def add_small_trades(p: ICEProblem, init_deltas: list):
    # simulate execution of intents
    state = p.omnipool.copy()
    intents = p.get_intents()
    deltas = [[intent_deltas[0], intent_deltas[1]] for intent_deltas in init_deltas]  # make deep copy
    init_valid, init_profit = validate_and_execute_solution(state, intents, deltas, "HDX")
    assert init_valid == True
    assert init_profit >= 0
    # go through small intents that remain, simulating their execution and adding them to the intents
    # size_limit = 1e-7  # trades less than this % of liquidity will be eligible for mandatory inclusion
    # max_impact_to_limit = 0.1  # to be eligible for mandatory inclusion, price impact to slippage limit must be less than this
    intents_remaining_indices = [i for i, intent in enumerate(intents) if intent['sell_quantity'] > 0]
    intents_remaining = [intent for intent in intents if intent['sell_quantity'] > 0]
    x = []
    for i, intent in enumerate(intents_remaining):
        limit_to_impact = calc_limit_to_impact(state, intent)
        if limit_to_impact >= 1:  # only include intents that have price limit above price impact
            x.append((limit_to_impact, i, intent))
    x.sort(reverse=True)  # is now all intents sorted by limit to impact, descending

    max_price_movement = 0.0001  # max price movement from mandatory small trades
    mandatory_intents = copy.deepcopy(intents)
    exec_indices = []
    simulated_state = state.copy()
    additional_deltas = []

    # include small trades until we get up to 0.5 bp of impact in each asset
    for (impact, i, intent) in x:
        # try executing trade
        sim_agent = intent['agent'].copy()
        assert sim_agent.holdings[intent['tkn_sell']] == intent['sell_quantity']
        simulated_state.swap(sim_agent, intent['tkn_buy'], intent['tkn_sell'], buy_quantity=intent['buy_quantity'])
        if not simulated_state.fail:  # trade was successful
            sell_amt = intent['sell_quantity']
            buy_amt = intent['buy_quantity']
            if intent['agent'].holdings[intent['tkn_sell']] - sim_agent.holdings[intent['tkn_sell']] > sell_amt:
                raise
            additional_deltas.append((intents_remaining_indices[i], [-sell_amt, buy_amt], intent['tkn_buy'], intent['tkn_sell']))
        else:
            simulated_state.fail = ""  # reset fail message

    deltas = [[intent_deltas[0], intent_deltas[1]] for intent_deltas in init_deltas]  # make deep copy
    orig_intents = p.get_intents()
    for add_i, add_deltas, _, _ in additional_deltas:
        deltas[add_i][0] += add_deltas[0]
        deltas[add_i][1] += add_deltas[1]
        assert -deltas[add_i][0] <= orig_intents[add_i]['sell_quantity']
        assert deltas[add_i][1] <= orig_intents[add_i]['buy_quantity']

    state = p.omnipool.copy()
    valid, profit = validate_and_execute_solution(state, orig_intents, deltas, "HDX")
    assert valid == True
    assert profit >= 0

    return init_deltas, -profit


def get_trading_tokens(intents, omnipool, amm_list):
    tkn_locations = {tkn: 1 for tkn in (omnipool.asset_list + ["LRNA"])}
    for amm in amm_list:
        if amm.unique_id not in tkn_locations:
            tkn_locations[amm.unique_id] = 0
        tkn_locations[amm.unique_id] += 1
        for tkn in amm.asset_list:
            if tkn not in tkn_locations:
                tkn_locations[tkn] = 0
            tkn_locations[tkn] += 1
    for intent in intents:
        if intent['tkn_sell'] not in tkn_locations:
            tkn_locations[intent['tkn_sell']] = 0
        tkn_locations[intent['tkn_sell']] += 1
        if intent['tkn_buy'] not in tkn_locations:
            tkn_locations[intent['tkn_buy']] = 0
        tkn_locations[intent['tkn_buy']] += 1

    trading_tkns = [tkn for tkn in tkn_locations if tkn_locations[tkn] > 1]
    return trading_tkns



def find_solution_outer_approx(state: OmnipoolState, init_intents: list, amm_list: list = None, min_partial: float = 1) -> list:
    if amm_list is None:
        amm_list = []

    # intents, exec_indices, init_i = find_initial_solution(state, init_intents)
    # init_i = exec_indices  # only execute mandatory trades initially
    trading_tkns = get_trading_tokens(init_intents, state, amm_list)
    init_i, exec_indices = [], []
    intents = copy.deepcopy(init_intents)
    p = ICEProblem(state, intents, amm_list=amm_list, init_i=init_i, min_partial=min_partial, trading_tkns=trading_tkns)

    m, r, n, sigma = p.m, p.r, p.n, p.sigma
    inf = highspy.kHighsInf
    k_milp = 4 * n + 3*sigma + m + r
    # get initial I values
    # set Z_L = -inf, Z_U = inf
    Z_L = -inf
    Z_U = inf
    best_status = "Not Solved"
    y_best = p.I
    best_omnipool_deltas = {tkn: 0 for tkn in p.asset_list}
    best_intent_deltas = [0]*m
    milp_obj = -inf

    # - force small trades to execute
    mandatory_indicators = [0] * r
    for i in exec_indices:
        mandatory_indicators[i] = 1
    BK = np.where(np.array(mandatory_indicators) == 1)[0] + 4 * n + 3 * sigma + m
    new_A = np.zeros((1, k_milp))
    new_A[0, BK] = 1
    new_A_upper = np.array([inf])
    new_A_lower = np.array([len(BK)])

    Z_U_archive = []
    Z_L_archive = []
    indicators = [i for i in p.I]
    x_list = np.zeros((0,4 * n + 3*sigma + m))

    # loop until MILP has no solution:
    for _i in range(5):
        # - do NLP solve given I values, update x^K
        p.set_up_problem(I=indicators)
        omnipool_deltas, intent_deltas, x, obj, dual_obj, status, amm_deltas = _find_good_solution(p, allow_loss=True)
        if obj < Z_U and dual_obj <= 0:  # - update Z_U, y*, x*
            Z_U = obj
            y_best = indicators
            best_omnipool_deltas = omnipool_deltas
            best_amm_deltas = amm_deltas
            best_intent_deltas = intent_deltas
            best_status = status

        if status not in ["PrimalInfeasible", "DualInfeasible"]:
            x_list = np.vstack([x_list, np.array(x)])

        # - get new cone constraint from I^K
        BK = np.where(np.array(indicators) == 1)[0] + 4 * n + 3 * sigma + m
        NK = np.where(np.array(indicators) == 0)[0] + 4 * n + 3 * sigma + m
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
        p.set_up_problem()
        omnipool_deltas, partial_intent_deltas, indicators, new_A, new_A_upper, new_A_lower, milp_obj, valid, milp_status, amm_deltas = _solve_inclusion_problem(p, x_list, Z_U, -inf, A, A_upper, A_lower)
        Z_L = max(Z_L, milp_obj)
        Z_U_archive.append(Z_U)
        Z_L_archive.append(Z_L)
        if not valid:
            break

    if Z_U > 0 or best_status in ['PrimalInfeasible', 'DualInfeasible']:
        best_omnipool_deltas = {tkn: 0.0 for tkn in p.omnipool.asset_list}
        best_amm_deltas = [[0]*(len(amm.asset_list) + 1) for amm in amm_list]
        best_intent_deltas = [0]*m
        y_best = [0]*r
        Z_U = 0
        # return [[0,0]]*(m+r), 0, Z_L_archive, Z_U_archive  # no solution found
    # elif best_status not in ['Solved', 'AlmostSolved']:
    #     raise

    sell_deltas = round_solution(p.partial_intents, best_intent_deltas)
    partial_deltas_with_buys = add_buy_deltas(p.partial_intents, sell_deltas)
    full_deltas_with_buys = [[-p.full_intents[l]['sell_quantity'], p.full_intents[l]['buy_quantity']] if y_best[l] == 1 else [0,0] for l in range(r)]
    deltas = [None] * (m + r)
    for i in range(len(p.partial_intent_indices)):
        deltas[p.partial_intent_indices[i]] = partial_deltas_with_buys[i]
    for i in range(len(p.full_intent_indices)):
        deltas[p.full_intent_indices[i]] = full_deltas_with_buys[i]
    return {'deltas': deltas, 'profit': -Z_U, 'Z_L': Z_L_archive, 'Z_U': Z_U_archive, 'omnipool_deltas': best_omnipool_deltas, 'amm_deltas': best_amm_deltas}
    # TODO: reimplement
    # deltas_final, obj = add_small_trades(p, deltas)
    # return deltas_final, -obj, Z_L_archive, Z_U_archive
