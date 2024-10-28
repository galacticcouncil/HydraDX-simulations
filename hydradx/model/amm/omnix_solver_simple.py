import copy
import math, bisect, mpmath

import clarabel
import numpy as np
import cvxpy as cp
import clarabel as cb
import highspy
from scipy import sparse

from hydradx.model.amm.omnipool_amm import OmnipoolState


class ICEProblem():
    def __init__(self, omnipool: OmnipoolState, init_intents: list, min_partial: float = 1):
        self.intents = init_intents
        self.min_partial = min_partial
        self.omnipool = omnipool
        self.set_partial_and_full_intents()
        self.asset_list = []
        for intent in self.intents:
            if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in self.asset_list:
                self.asset_list.append(intent['tkn_sell'])
            if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in self.asset_list:
                self.asset_list.append(intent['tkn_buy'])
        self.n = len(self.asset_list)


    def set_partial_and_full_intents(self):
        # resets partial and full intents, and sets partial_intent_indices and full_intent_indices, from self.intents
        temp_intents = []
        for intent in self.intents:
            temp_intents.append(copy.deepcopy(intent))
            buy_amt_lrna_value = intent['buy_quantity'] * self.omnipool.price(self.omnipool, intent['tkn_buy'])
            selL_amt_lrna_value = intent['sell_quantity'] * self.omnipool.price(self.omnipool, intent['tkn_sell'])
            if buy_amt_lrna_value < self.min_partial and selL_amt_lrna_value < self.min_partial:
                temp_intents[-1]['partial'] = False

        self.partial_intent_indices = [i for i in range(len(temp_intents)) if temp_intents[i]['partial']]
        self.full_intent_indices = [i for i in range(len(temp_intents)) if not temp_intents[i]['partial']]
        self.partial_intents = [temp_intents[i] for i in self.partial_intent_indices]
        self.full_intents = [temp_intents[i] for i in self.full_intent_indices]
        self.set_partial_intent_directions()
        self.m = len(self.partial_intents)
        self.r = len(self.full_intents)


    def set_partial_intent_directions(self):
        self.partial_intent_directions = {}
        for intent in self.partial_intents:
            if intent['buy_quantity'] > 0 and intent['sell_quantity'] > 0:
                if intent['tkn_sell'] not in self.partial_intent_directions:
                    self.partial_intent_directions[intent['tkn_sell']] = "sell"
                elif self.partial_intent_directions[intent['tkn_sell']] == "buy":
                    self.partial_intent_directions[intent['tkn_sell']] = "both"
                if intent['tkn_buy'] not in self.partial_intent_directions:
                    self.partial_intent_directions[intent['tkn_buy']] = "buy"
                elif self.partial_intent_directions[intent['tkn_buy']] == "sell":
                    self.partial_intent_directions[intent['tkn_buy']] = "both"


    def scale_down_partial_intents(self, trade_pcts):
        zero_ct = 0
        for i, intent in enumerate(self.partial_intents):
            # we allow new solution to find trade size up to 10x old solution
            new_sell_quantity = min([intent['sell_quantity'] * trade_pcts[i] * 10, intent['sell_quantity']])
            new_buy_quantity = min([intent['buy_quantity'] * trade_pcts[i] * 10, intent['buy_quantity']])
            buy_amt_lrna_value = new_buy_quantity * self.omnipool.price(self.omnipool, intent['tkn_buy'])
            selL_amt_lrna_value = new_sell_quantity * self.omnipool.price(self.omnipool, intent['tkn_sell'])
            # if we are scaling lower than min_partial, we eliminate the intent from execution
            if buy_amt_lrna_value < self.min_partial and selL_amt_lrna_value < self.min_partial:
                new_sell_quantity = 0
                new_buy_quantity = 0
                zero_ct += 1  # we count the number of intents that are eliminated
            self.partial_intents[i]['sell_quantity'] = new_sell_quantity
            self.partial_intents[i]['buy_quantity'] = new_buy_quantity
        self.set_partial_intent_directions()
        return zero_ct



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
        if intent['tkn_sell'] != "LRNA":
            scaling[intent['tkn_sell']] = max(scaling[intent['tkn_sell']], intent['sell_quantity'])
        if intent['tkn_buy'] != "LRNA":
            scaling[intent['tkn_buy']] = max(scaling[intent['tkn_buy']], intent['buy_quantity'])
        max_in[intent['tkn_sell']] += intent['sell_quantity']
        max_out[intent['tkn_buy']] += intent['buy_quantity']
    for i, intent in enumerate(full_intents):
        if I[i] > 0.5:
            if intent['tkn_sell'] != "LRNA":
                scaling[intent['tkn_sell']] = max(scaling[intent['tkn_sell']], intent['sell_quantity'])
            if intent['tkn_buy'] != "LRNA":
                scaling[intent['tkn_buy']] = max(scaling[intent['tkn_buy']], intent['buy_quantity'])
            max_in[intent['tkn_sell']] += intent['sell_quantity']
            max_out[intent['tkn_sell']] -= intent['sell_quantity']
            max_out[intent['tkn_buy']] += intent['buy_quantity']
            max_in[intent['tkn_buy']] -= intent['buy_quantity']
    for tkn in asset_list:
        if scaling[tkn] == 0:
            scaling[tkn] = 1
        else:
            scaling[tkn] = min(scaling[tkn], state.liquidity[tkn])
        # set scaling for LRNA equal to scaling for asset, adjusted by spot price
        scalar = scaling[tkn] * state.lrna[tkn] / state.liquidity[tkn]
        scaling["LRNA"] = max(scaling["LRNA"], scalar)
    return scaling, max_in, max_out


# def _calculate_objective_coefs(indices_to_keep: list, n: int, m: int, lrna_fees: list, tau, full_intents, scaling):
#     k_real = len(indices_to_keep)
#     P_trimmed = sparse.csc_matrix((k_real, k_real))
#
#     y_coefs = np.ones(n)
#     x_coefs = np.zeros(n)
#     lrna_lambda_coefs = np.array(lrna_fees)
#     lambda_coefs = np.zeros(n)
#     d_coefs = np.array([-tau[0,j] for j in range(m)])
#     objective_I_coefs = np.array([-tau[0,m+l]*full_intents[l]['sell_quantity']/scaling["LRNA"] for l in range(r)])
#     q = np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs])
#     q_trimmed = np.array([q[i] for i in indices_to_keep])


def _find_solution_unrounded(
        p: ICEProblem,
        I: list,
        flags: dict = None,
        # force_linear: list = None,
        buffer_fee: float = 0.0
) -> (dict, list):

    full_intents, partial_intents, state = p.full_intents, p.partial_intents, p.omnipool

    # if force_linear is None:
    #     force_linear = []

    # assets involved trades for which execution is being solved
    asset_list = p.asset_list
    n, m, r = p.n, p.m, p.r
    if len(partial_intents) + sum(I) == 0:  # nothing for solver to do
        return {tkn: 0 for tkn in asset_list}, [], np.zeros(4*n), 0, 0, 'Solved'  # TODO enable solver with m=0

    intent_directions = copy.deepcopy(p.partial_intent_directions)

    known_flow = {tkn: 0 for tkn in ["LRNA"] + asset_list}
    for i, intent in enumerate(full_intents):
        if I[i] > 0.5:
            known_flow[intent['tkn_sell']] += intent["sell_quantity"]
            known_flow[intent['tkn_buy']] -= intent["buy_quantity"]
    for tkn in asset_list:
        if known_flow[tkn] > 0:  # net agent is selling tkn
            if tkn not in intent_directions:
                intent_directions[tkn] = "sell"
            elif intent_directions[tkn] == "buy":
                intent_directions[tkn] = "both"
        elif known_flow[tkn] < 0:  # net agent is buying tkn
            if tkn not in intent_directions:
                intent_directions[tkn] = "buy"
            elif intent_directions[tkn] == "sell":
                intent_directions[tkn] = "both"

    assert len(I) == len(full_intents)
    k = 4 * n + m

    if flags is None:
        flags = {}
    directions = {}
    indices_to_keep = list(range(k))
    for i, tkn in enumerate(asset_list):
        if tkn in flags and flags[tkn] != 0:
            directions[tkn] = "buy" if flags[tkn] == 1 else "sell"
        elif tkn in intent_directions:
            if intent_directions[tkn] == "sell":
                directions[tkn] = "buy"
            elif intent_directions[tkn] == "buy":
                directions[tkn] = "sell"
    for tkn in directions:
        if directions[tkn] == "sell":
            indices_to_keep.remove(2 * n + asset_list.index(tkn))  # lrna_lambda_i is zero
        elif directions[tkn] == "buy":
            indices_to_keep.remove(3 * n + asset_list.index(tkn))  # lambda_i is zero

    tkn_list = ["LRNA"] + asset_list

    partial_intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) if intent['sell_quantity'] > 0 else 0 for intent in partial_intents]

    fees = [float(state.last_fee[tkn]) for tkn in asset_list]  # f_i
    lrna_fees = [float(state.last_lrna_fee[tkn]) for tkn in asset_list]  # l_i
    fee_match = 0.0005
    if len(fees) > 0:
        assert fee_match <= min(fees)  # breaks otherwise

    # calculate tau, phi
    scaling, max_in, max_out = _calculate_scaling(partial_intents, full_intents, I, state, asset_list)
    tau1, phi1 = _calculate_tau_phi(partial_intents, tkn_list, scaling)
    tau2, phi2 = _calculate_tau_phi(full_intents, tkn_list, scaling)
    tau = sparse.hstack([tau1, tau2])
    phi = sparse.hstack([phi1, phi2])
    epsilon_tkn = {t: max([abs(max_in[t]), abs(max_out[t])]) / state.liquidity[t] for t in asset_list}

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    k_real = len(indices_to_keep)
    P_trimmed = sparse.csc_matrix((k_real, k_real))

    y_coefs = np.ones(n)
    x_coefs = np.zeros(n)
    lrna_lambda_coefs = np.array(lrna_fees) + buffer_fee
    lambda_coefs = np.zeros(n)
    d_coefs = np.array([-tau[0,j] for j in range(m)])
    objective_I_coefs = np.array([-tau[0,m+l]*full_intents[l]['sell_quantity']/scaling["LRNA"] for l in range(r)])
    q = np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs])
    q_trimmed = np.array([q[i] for i in indices_to_keep])

    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#

    # intent variables are non-negative
    diff_coefs = sparse.csc_matrix((2*n + m,2*n))
    nonzero_coefs = -sparse.identity(2 * n + m, format='csc')
    A1 = sparse.hstack([diff_coefs, nonzero_coefs])
    rows_to_keep = [i for i in range(2*n+m) if 2*n+i in indices_to_keep]
    A1_trimmed = A1[:, indices_to_keep][rows_to_keep, :]
    b1 = np.zeros(A1_trimmed.shape[0])
    cone1 = cb.NonnegativeConeT(A1_trimmed.shape[0])

    # intents cannot sell more than they have
    amm_coefs = sparse.csc_matrix((m, 4*n))
    d_coefs = sparse.identity(m, format='csc')
    A2 = sparse.hstack([amm_coefs, d_coefs], format='csc')
    b2 = np.array([float(i['sell_quantity']/scaling[i['tkn_sell']]) for i in partial_intents])
    A2_trimmed = A2[:, indices_to_keep]
    cone2 = cb.NonnegativeConeT(m)

    # leftover must be higher than required fees
    # LRNA
    y_coefs = np.ones(n)
    x_coefs = np.zeros(n)
    lrna_lambda_coefs = np.array(lrna_fees) + buffer_fee
    lambda_coefs = np.zeros(n)
    d_coefs = -(tau1[0, :].toarray()[0])
    I_coefs_lrna = sparse.csc_matrix(np.array([[-tau[0,m+l]*float(full_intents[l]['sell_quantity']/scaling["LRNA"]) for l in range(r)]]))
    A30 = sparse.csc_matrix(np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs]))

    # other assets
    y_coefs = sparse.csc_matrix((n,n))
    x_coefs = sparse.identity(n, format='csc')
    lrna_lambda_coefs = sparse.csc_matrix((n,n))
    lambda_coefs = sparse.diags(np.array(fees)-fee_match+buffer_fee, format='csc')
    d_coefs = sparse.csc_matrix([[1/(1-fee_match)*phi[i,j]*partial_intent_prices[j]*float(scaling[partial_intents[j]['tkn_sell']]/scaling[partial_intents[j]['tkn_buy']]) - tau[i, j] for j in range(m)] for i in range(1,n+1)])
    I_coefs = sparse.csc_matrix([[float((1 / (1 - fee_match) * phi[i, m+l] * full_intents[l]['buy_quantity'] - tau[i, m+l] * full_intents[l]['sell_quantity'])/scaling[tkn_list[i]]) for l in range(r)] for i in range(1,n+1)])
    A31 = sparse.hstack([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs])

    A3 = sparse.vstack([A30, A31], format='csc')
    A3_trimmed = A3[:, indices_to_keep]
    if r == 0:
        b3 = np.zeros(n+1)
    else:
        b3 = -sparse.vstack([I_coefs_lrna, I_coefs], format='csc') @ I
    cone3 = cb.NonnegativeConeT(n + 1)

    # AMM invariants must not go down
    A4 = sparse.csc_matrix((0, k))
    b4 = np.array([])
    cones4 = []
    for i in range(n):
        tkn = asset_list[i]
        if epsilon_tkn[tkn] <= 1e-6:  # linearize the AMM constraint
            if tkn not in directions:
                c1 = 1 / (1 + epsilon_tkn[tkn])
                c2 = 1 / (1 - epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((2, k))
                b4i = np.zeros(2)
                A4i[0, i] = -scaling["LRNA"]/state.lrna[tkn]
                A4i[0, n + i] = -scaling[tkn]/state.liquidity[tkn] * c1
                A4i[1, i] = -scaling["LRNA"]/state.lrna[tkn]
                A4i[1, n + i] = -scaling[tkn]/state.liquidity[tkn] * c2
                cones4.append(cb.NonnegativeConeT(2))
            else:
                if directions[tkn] == "sell":
                    c = 1 / (1 - epsilon_tkn[tkn])
                else:
                    c = 1 / (1 + epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((1, k))
                b4i = np.zeros(1)
                A4i[0, i] = -scaling["LRNA"]/state.lrna[tkn]
                A4i[0, n+i] = -scaling[tkn]/state.liquidity[tkn] * c
                cones4.append(cb.ZeroConeT(1))
        elif epsilon_tkn[tkn] <= 1e-3:  # quadratic approximation to in-given-out function
            A4i = sparse.csc_matrix((3, k))
            A4i[1,i] = -scaling["LRNA"]/state.lrna[tkn]
            A4i[1,n+i] = -scaling[tkn]/state.liquidity[tkn]
            A4i[2,n+i] = -scaling[tkn]/state.liquidity[tkn]
            b4i = np.array([1, 0, 0])
            cones4.append(cb.PowerConeT(0.5))
        else:  # full AMM constraint
            A4i = sparse.csc_matrix((3, k))
            b4i = np.ones(3)
            A4i[0, i] = -scaling["LRNA"] / state.lrna[tkn]
            A4i[1, n + i] = -scaling[tkn] / state.liquidity[tkn]
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
    for i in range(n):
        tkn = tkn_list[i+1]
        new_amm_deltas[tkn] = x_expanded[n+i] * scaling[tkn]

    for i in range(len(partial_intents)):
        exec_intent_deltas[i] = -x_expanded[4 * n + i] * scaling[partial_intents[i]['tkn_sell']]

    fixed_profit = objective_I_coefs @ I if I is not None else 0
    return new_amm_deltas, exec_intent_deltas, x_expanded, (solution.obj_val + fixed_profit) * scaling["LRNA"], (solution.obj_val_dual + fixed_profit) * scaling["LRNA"], str(solution.status)


def _solve_inclusion_problem(
        state: OmnipoolState,
        intents: list,
        x: np.array = None,  # NLP solution
        upper_bound: float = None,
        lower_bound: float = None,
        old_A = None,
        old_A_upper = None,
        old_A_lower = None,
        buffer_fee: float = 0.0
):

    asset_list = []
    for intent in intents:
        if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in asset_list:
            asset_list.append(intent['tkn_sell'])
        if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in asset_list:
            asset_list.append(intent['tkn_buy'])
    tkn_list = ["LRNA"] + asset_list

    partial_intents = [i for i in intents if i['partial']]
    full_intents = [i for i in intents if not i['partial']]
    n = len(asset_list)
    m = len(partial_intents)
    r = len(full_intents)
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
    lrna_lambda_coefs = np.array(lrna_fees) + buffer_fee
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
        A3[i+1, 3*n+i] = fees[i] - fee_match + buffer_fee
        for j in range(m):
            A3[i+1, 4*n+j] = 1/(1-fee_match)*phi[i+1, j] *scaling[intents[j]['tkn_sell']]/scaling[intents[j]['tkn_buy']] * partial_intent_prices[j] - tau[i+1, j]
        for l in range(r):
            buy_amt = 1 / (1 - fee_match) * phi[i+1, m+l] * full_intents[l]['buy_quantity']
            sell_amt = tau[i + 1, m+l] * full_intents[l]['sell_quantity']
            A3[i+1, 4*n+m+l] = (buy_amt - sell_amt)/scaling[asset_list[i]]
    A3_upper = np.zeros(n+1)
    A3_lower = np.array([-inf]*(n+1))

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

    lp.col_cost_ = c
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
        exec_partial_intent_deltas[i] = -x_expanded[4 * n + i] * scaling[intents[i]['tkn_sell']]

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

    p = ICEProblem(state, init_intents, min_partial)

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
        amm_deltas, intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded(p, I=indicators)
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
        amm_deltas, partial_intent_deltas, indicators, new_A, new_A_upper, new_A_lower, milp_obj, valid = _solve_inclusion_problem(state, p.partial_intents + p.full_intents, x, Z_U, Z_L, A, A_upper, A_lower)
        if not valid:
            break

    if valid == True:  # this means we did not get to a solution
        return [[0,0]]*(m + r)

    trade_pcts = [-best_intent_deltas[i] / intent['sell_quantity'] for i, intent in enumerate(p.partial_intents)]
    new_partial_intents = copy.deepcopy(p.partial_intents)

    # if solution is not good yet, try scaling down partial intent sizes, to get scaling better
    while len(new_partial_intents) > 0 and (best_status != "Solved" or Z_U > 0) and min(trade_pcts) < 0.05:
        zero_ct = p.scale_down_partial_intents(trade_pcts)
        if zero_ct == m:
            break  # all partial intents have been eliminated from execution

        amm_deltas, intent_deltas, x, obj, dual_obj, temp_status = _find_solution_unrounded(p, I=y_best)
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
        trade_pcts = [-best_intent_deltas[i] / intent['sell_quantity'] if intent['sell_quantity'] > 0 else 0 for i, intent in enumerate(new_partial_intents)]


    flags = get_directional_flags(best_amm_deltas)
    best_amm_deltas, best_intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded(p, I=y_best, flags=flags)
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
    sell_deltas = round_solution(new_partial_intents, best_intent_deltas)
    partial_deltas_with_buys = add_buy_deltas(new_partial_intents, sell_deltas)
    full_deltas_with_buys = [[-p.full_intents[l]['sell_quantity'], p.full_intents[l]['buy_quantity']] if y_best[l] == 1 else [0,0] for l in range(r)]
    deltas = [None] * (m + r)
    for i in range(len(p.partial_intent_indices)):
        deltas[p.partial_intent_indices[i]] = partial_deltas_with_buys[i]
    for i in range(len(p.full_intent_indices)):
        deltas[p.full_intent_indices[i]] = full_deltas_with_buys[i]
    return deltas
