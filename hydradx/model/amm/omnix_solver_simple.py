import copy
import math, bisect, mpmath

import clarabel
import numpy as np
import cvxpy as cp
import clarabel as cb
import highspy
from scipy import sparse

from hydradx.model.amm.omnipool_amm import OmnipoolState


def convert_intents(intents, tkn_list):
    tkn_map = {tkn: i for i, tkn in enumerate(tkn_list)}

    intent_indices = []
    intent_reserves = []
    intent_prices = []
    full_intent_indices = []
    full_intent_reserves = []
    full_intent_prices = []
    for intent in intents:
        sell_amt = intent['sell_quantity']
        buy_amt = intent['buy_quantity']
        if 'partial' in intent and not intent['partial']:
            full_intent_indices.append([tkn_map[intent['tkn_sell']], tkn_map[intent['tkn_buy']]])
            full_intent_reserves.append([sell_amt, 0])
            full_intent_prices.append(buy_amt / sell_amt)
        else:
            intent_indices.append([tkn_map[intent['tkn_sell']], tkn_map[intent['tkn_buy']]])
            intent_reserves.append([sell_amt, 0])
            intent_prices.append(buy_amt / sell_amt)

    return intent_indices, intent_reserves, intent_prices, full_intent_indices, full_intent_reserves, full_intent_prices


def _find_solution_unrounded(state: OmnipoolState, intents: list, flags: dict = None) -> (dict, list):

    if flags is None:
        flags = {}

    tkn_list = ["LRNA"] + state.asset_list

    reserves_list = [[state.lrna[tkn], state.liquidity[tkn]] for tkn in state.lrna]

    fees = [state.last_fee[tkn] for tkn in state.asset_list]
    lrna_fees = [state.last_lrna_fee[tkn] for tkn in state.asset_list]

    profit_value = [1] + [0] * len(state.asset_list)  # for taking profits in LRNA

    # transform data for cvxpy
    intent_indices, intent_reserves, intent_prices, full_intent_indices, full_intent_reserves, full_intent_prices = convert_intents(intents, tkn_list)
    global_indices = list(range(len(tkn_list)))
    local_indices = [[0, i+1] for i in range(len(state.asset_list))]

    reserves = list(map(np.array, reserves_list))
    reserves2 = list(map(np.array, intent_reserves))
    reserves3 = list(map(np.array, full_intent_reserves))

    # Build local-global matrices
    n = len(global_indices)

    A = []
    for l in local_indices:
        n_i = len(l)
        A_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1
        A.append(A_i)

    B = []
    for l in intent_indices:
        n_i = len(l)
        B_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            B_i[idx, i] = 1
        B.append(B_i)

    C = []
    for l in full_intent_indices:
        n_i = len(l)
        C_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            C_i[idx, i] = 1
        C.append(C_i)

    # Build variables
    deltas, lambdas, lrna_deltas, lrna_lambdas = [], [], [], []
    for tkn in state.asset_list:
        if tkn not in flags:
            deltas.append(cp.Variable(1, nonneg=True, name="delta_" + tkn))
            lambdas.append(cp.Variable(1, nonneg=True, name="lambda_" + tkn))
            lrna_deltas.append(cp.Variable(1, nonneg=True, name="lrna_delta_" + tkn))
            lrna_lambdas.append(cp.Variable(1, nonneg=True, name="lrna_lambda_" + tkn))
        elif flags[tkn] == 1:  # asset going into Omnipool
            deltas.append(cp.Variable(1, nonneg=True, name="delta_" + tkn))
            lambdas.append(cp.Constant(np.zeros(1)))
            lrna_deltas.append(cp.Constant(np.zeros(1)))
            lrna_lambdas.append(cp.Variable(1, nonneg=True, name="lrna_lambda_" + tkn))
        elif flags[tkn] == -1:  # asset going out of Omnipool
            deltas.append(cp.Constant(np.zeros(1)))
            lambdas.append(cp.Variable(1, nonneg=True, name="lambda_" + tkn))
            lrna_deltas.append(cp.Variable(1, nonneg=True, name="lrna_delta_" + tkn))
            lrna_lambdas.append(cp.Constant(np.zeros(1)))
        else:  # no change in asset
            deltas.append(cp.Constant(np.zeros(1)))
            lambdas.append(cp.Constant(np.zeros(1)))
            lrna_deltas.append(cp.Constant(np.zeros(1)))
            lrna_lambdas.append(cp.Constant(np.zeros(1)))

    intent_deltas = [cp.Variable(1, nonneg=True, name="intent_delta_" + str(i)) for i, l in enumerate(intent_indices)]
    full_intent_flags = [cp.Variable(1, boolean=True, name="full_intent_flag_" + str(i)) for i, l in enumerate(full_intent_indices)]

    profits = [A_i @ cp.hstack([(1-l) * qL[0] - qD[0], L[0] - D[0]]) for A_i, D, L, qD, qL, l in zip(A, deltas, lambdas, lrna_deltas, lrna_lambdas, lrna_fees)]  # assets from AMMs
    intent_profits = [B_i @ cp.hstack([D[0], -D[0] * p]) for B_i, D, p in zip(B, intent_deltas, intent_prices)]# assets from intents
    full_intent_profits = [C_i @ cp.hstack([R[0] * f, -R[0] * f * p]) for C_i, R, p, f in zip(C, full_intent_reserves, full_intent_prices, full_intent_flags)]  # assets from full intents
    total_profits = profits + intent_profits + full_intent_profits

    psi = cp.sum(total_profits)

    # Objective is to maximize "total market value" of coins out
    obj = cp.Maximize(profit_value @ psi)  # take profits in LRNA

    # Reserves after trade
    new_reserves = [R + cp.hstack([qD[0] - qL[0], D[0] - L[0] / (1 - f_i)]) for R, f_i, D, L, qD, qL in zip(reserves, fees, deltas, lambdas, lrna_deltas, lrna_lambdas)]
    new_intent_reserves = [R[0] - D[0] for R, D in zip(reserves2, intent_deltas)]

    cons = [psi >= 0]  # no negative profits
    for i in range(len(reserves)):  # AMM invariants must not go down
        tkn = state.asset_list[i]
        if tkn not in flags or flags[tkn] != 0:
            cons.append(cp.geo_mean(new_reserves[i]) >= cp.geo_mean(reserves[i]))

    for i in range(len(intent_indices)):  # intent constraints
        cons.append(new_intent_reserves[i] >= 0)  # cannot sell more than you have

    # # build pos_indices and neg_indices
    # pos_indices = [state.asset_list.index(tkn) for tkn in flags if flags[tkn] >= 0]  # asset going into AMM, LRNA going out
    # neg_indices = [state.asset_list.index(tkn) for tkn in flags if flags[tkn] < 0]  # asset going out of AMM, LRNA going in
    # for i in pos_indices:
    #     cons.append(lambdas[i] == 0)
    #     cons.append(lrna_deltas[i] == 0)
    # for i in neg_indices:
    #     cons.append(deltas[i] == 0)
    #     cons.append(lrna_lambdas[i] == 0)

    # # we can constrain total amount in/out by total desired in/out of intents
    # # however these constraints are really only helpful to contain rounding errors
    # max_out = [0] * len(tkn_list)
    # max_in = [0] * len(tkn_list)
    # for intent in intents:
    #     sell_amt = intent['sell_quantity']
    #     buy_amt = intent['buy_quantity']
    #     sell_i = tkn_list.index(intent['tkn_sell'])
    #     buy_i = tkn_list.index(intent['tkn_buy'])
    #     max_in[sell_i] += sell_amt
    #     max_out[buy_i] += buy_amt
    #
    # for i in range(len(tkn_list)):
    #     cons.append(deltas[i-1][0] <= max_in[i])
    #     cons.append(lambdas[i-1][0] <= max_out[i])

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True)

    # extract solution
    amm_ct = len(local_indices)
    new_amm_deltas = {}
    exec_intent_deltas = [None] * len(intent_indices)
    for i in range(amm_ct):
        tkn = tkn_list[i+1]
        new_amm_deltas[tkn] = 0
        new_amm_deltas[tkn] += prob.var_dict['delta_' + tkn].value if 'delta_' + tkn in prob.var_dict else 0
        new_amm_deltas[tkn] -= prob.var_dict['lambda_' + tkn].value if 'lambda_' + tkn in prob.var_dict else 0
    for i in range(len(intent_indices)):
        exec_intent_deltas[i] = -prob.var_dict['intent_delta_' + str(i)].value

    return new_amm_deltas, exec_intent_deltas


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


def _find_solution_unrounded2(state: OmnipoolState, intents: list, flags: dict = None, fee_buffer: float = 0.0001) -> (dict, list):

    intent_directions = {}
    asset_list = []
    for intent in intents:
        if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in asset_list:
            asset_list.append(intent['tkn_sell'])
        if intent['tkn_sell'] != "LRNA" and intent['tkn_buy'] not in asset_list:
            asset_list.append(intent['tkn_buy'])
        if intent['tkn_sell'] not in intent_directions:
            intent_directions[intent['tkn_sell']] = "sell"
        elif intent_directions[intent['tkn_sell']] == "buy":
            intent_directions[intent['tkn_sell']] = "both"
        if intent['tkn_buy'] not in intent_directions:
            intent_directions[intent['tkn_buy']] = "buy"
        elif intent_directions[intent['tkn_buy']] == "sell":
            intent_directions[intent['tkn_buy']] = "both"

    n = len(asset_list)
    m = len(intents)
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
            indices_to_keep.remove(n + asset_list.index(tkn))
            indices_to_keep.remove(2 * n + asset_list.index(tkn))
        elif directions[tkn] == "buy":
            indices_to_keep.remove(asset_list.index(tkn))
            indices_to_keep.remove(3 * n + asset_list.index(tkn))

    tkn_list = ["LRNA"] + asset_list

    intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) for intent in intents]

    fees = [float(state.last_fee[tkn]) for tkn in asset_list]  # f_i
    lrna_fees = [float(state.last_lrna_fee[tkn]) for tkn in asset_list]  # l_i
    fee_match = 0.0005
    assert fee_match <= min(fees)  # breaks otherwise

    # calculate tau, phi
    scaling = _calculate_scaling(intents, state, asset_list)
    tau, phi = _calculate_tau_phi(intents, tkn_list, scaling)

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    k_real = len(indices_to_keep)
    P_trimmed = sparse.csc_matrix((k_real, k_real))

    scaled_spot_prices = [1] + [float(scaling[tkn] * state.lrna[tkn] / state.liquidity[tkn]) for tkn in asset_list]
    spot_prices = [1] + [float(state.lrna[tkn] / state.liquidity[tkn]) for tkn in asset_list]

    delta_lrna_coefs = np.ones(n)  # need to multiply by each Qi
    lambda_lrna_coefs = -np.ones(n)
    delta_coefs = np.array([scaled_spot_prices[i+1] for i in range(n)])
    lambda_coefs = np.array([(fees[i] - 1) * scaled_spot_prices[i+1] for i in range(n)])
    d_coefs = np.array([sum([(phi[i,j]*intent_prices[j] - tau[i,j])*scaled_spot_prices[i] for i in range(n+1)]) for j in range(m)])
    q = np.concatenate([delta_lrna_coefs, lambda_lrna_coefs, delta_coefs, lambda_coefs, d_coefs])
    q_trimmed = np.array([q[i] for i in indices_to_keep])


    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#

    # all variables are non-negative
    A1_trimmed = -sparse.identity(k_real, format='csc')
    b1_trimmed = np.zeros(k_real)
    cone1_trimmed = cb.NonnegativeConeT(k_real)

    # intents cannot sell more than they have
    amm_coefs = sparse.csc_matrix((m, 4*n))
    d_coefs = sparse.identity(m, format='csc')
    A2 = sparse.hstack([amm_coefs, d_coefs], format='csc')
    b2 = np.array([float(i['sell_quantity']/scaling[i['tkn_sell']]) for i in intents])
    A2_trimmed = A2[:, indices_to_keep]
    cone2 = cb.NonnegativeConeT(m)

    # leftover must be higher than required fees
    # LRNA
    delta_lrna_coefs = np.ones(n)  # need to multiply by each Qi
    lambda_lrna_coefs = np.array(lrna_fees) - 1 + fee_buffer
    zero_coefs = np.zeros(2 * n)
    d_coefs = -(tau[0, :].toarray()[0])
    A30 = sparse.csc_matrix(np.concatenate([delta_lrna_coefs, lambda_lrna_coefs, zero_coefs, d_coefs]))
    b30 = np.zeros(1)
    # other assets
    lrna_coefs = sparse.csc_matrix((n, 2*n))
    delta_coefs = sparse.identity(n, format='csc')
    lambda_coefs = sparse.diags(np.array(fees)-fee_match-1, format='csc')
    d_coefs = sparse.csc_matrix([[1/(1-fee_match)*phi[i,j]*intent_prices[j] - tau[i, j] for j in range(m)] for i in range(1,n+1)])
    A31 = sparse.hstack([lrna_coefs, delta_coefs, lambda_coefs, d_coefs], format='csc')
    b31 = np.zeros(n)
    A3 = sparse.vstack([A30, A31], format='csc')
    A3_trimmed = A3[:, indices_to_keep]
    b3 = np.concatenate([b30, b31])
    cone3 = cb.NonnegativeConeT(n + 1)

    # AMM invariants must not go down
    A4 = sparse.csc_matrix((3 * n, k))
    b4 = np.ones(3 * n)
    cones4 = []
    for i in range(n):  # affected rows are 3i through 3i+2
        tkn = asset_list[i]
        A4[3*i, i] = -scaling["LRNA"]/state.lrna[tkn]
        A4[3*i, n+i] = -A4[3*i, i]
        A4[3*i+1, 2*n+i] = -scaling[tkn]/state.liquidity[tkn]
        A4[3*i+1, 3*n+i] = -A4[3*i+1, 2*n+i]
        cones4.append(cb.PowerConeT(0.5))
    A4_trimmed = A4[:, indices_to_keep]

    A = sparse.vstack([A1_trimmed, A2_trimmed, A3_trimmed, A4_trimmed], format='csc')
    b = np.concatenate([b1_trimmed, b2, b3, b4])
    cones = [cone1_trimmed, cone2, cone3] + cones4
    q = q_trimmed
    P = P_trimmed

    # solve
    settings = clarabel.DefaultSettings()
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    x = solution.x
    z = solution.z
    s = solution.s

    new_amm_deltas = {}
    exec_intent_deltas = [None] * len(intents)
    x_expanded = [0] * k
    for i in range(k):
        if i in indices_to_keep:
            x_expanded[i] = x[indices_to_keep.index(i)]
    for i in range(n):
        tkn = tkn_list[i+1]
        new_amm_deltas[tkn] = (x_expanded[2*n+i] - x_expanded[3*n+i]) * scaling[tkn]
    for i in range(len(intents)):
        exec_intent_deltas[i] = -x_expanded[4 * n + i] * scaling[intents[i]['tkn_sell']]

    return new_amm_deltas, exec_intent_deltas


def _find_solution_unrounded3(
        state: OmnipoolState,
        partial_intents: list,
        full_intents: list = None,
        I: list = None,
        flags: dict = None,
        epsilon: float = 1e-5,
        force_linear: list = None,
        buffer_fee: float = 0.0
) -> (dict, list):

    if full_intents is None:
        full_intents = []
    if force_linear is None:
        force_linear = []

    asset_list = []
    for intent in partial_intents + full_intents:
        if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in asset_list:
            asset_list.append(intent['tkn_sell'])
        if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in asset_list:
            asset_list.append(intent['tkn_buy'])
    n = len(asset_list)
    if len(partial_intents) + sum(I) == 0:
        return {tkn: 0 for tkn in asset_list}, [], np.zeros(4*n), 0, 0, 'Solved'

    intent_directions = {}
    for intent in partial_intents:
        if intent['tkn_sell'] not in intent_directions:
            intent_directions[intent['tkn_sell']] = "sell"
        elif intent_directions[intent['tkn_sell']] == "buy":
            intent_directions[intent['tkn_sell']] = "both"
        if intent['tkn_buy'] not in intent_directions:
            intent_directions[intent['tkn_buy']] = "buy"
        elif intent_directions[intent['tkn_buy']] == "sell":
            intent_directions[intent['tkn_buy']] = "both"

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


    if I is None:  # run all intents as though they are partial
        assert full_intents == []
    else:
        assert len(I) == len(full_intents)
    m = len(partial_intents)
    r = len(full_intents)
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
        if epsilon_tkn[tkn] <= epsilon or tkn in force_linear:  # linearize the AMM constraint
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


def get_directional_flags(amm_deltas: dict) -> list:
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
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents, flags)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)


def find_solution2(state: OmnipoolState, intents: list) -> list:
    amm_deltas, intent_deltas = _find_solution_unrounded2(state, intents)
    flags = get_directional_flags(amm_deltas)
    amm_deltas, intent_deltas = _find_solution_unrounded2(state, intents, flags)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)

def find_solution3(state: OmnipoolState, intents: list, epsilon: float = 1e-5) -> list:
    amm_deltas, intent_deltas = _find_solution_unrounded3(state, intents, epsilon = epsilon)
    flags = get_directional_flags(amm_deltas)
    amm_deltas, intent_deltas = _find_solution_unrounded3(state, intents, flags = flags, epsilon = epsilon)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)


def find_solution_outer_approx(state: OmnipoolState, init_intents: list, epsilon: float = 1e-5, min_partial: float = 1) -> list:
    if len(init_intents) == 0:
        return []

    # intents in which both buy and sell value are less than min_partial value of LRNA at spot prices must be fully executed
    intents = []
    for intent in init_intents:
        intents.append(copy.deepcopy(intent))
        buy_amt_lrna_value = intent['buy_quantity'] * state.price(state, intent['tkn_buy'])
        selL_amt_lrna_value = intent['sell_quantity'] * state.price(state, intent['tkn_sell'])
        if buy_amt_lrna_value < min_partial and selL_amt_lrna_value < min_partial:
            intents[-1]['partial'] = False

    partial_intent_indices = [i for i in range(len(intents)) if intents[i]['partial']]
    full_intent_indices = [i for i in range(len(intents)) if not intents[i]['partial']]
    partial_intents = [intents[i] for i in partial_intent_indices]
    full_intents = [intents[i] for i in full_intent_indices]

    asset_list = []
    for intent in intents:
        if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in asset_list:
            asset_list.append(intent['tkn_sell'])
        if intent['tkn_buy'] != "LRNA" and intent['tkn_buy'] not in asset_list:
            asset_list.append(intent['tkn_buy'])
    buffer_fee = 0.0000
    m = len(partial_intents)
    r = len(full_intents)
    inf = highspy.kHighsInf
    n = len(asset_list)
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
    # loop until linearization has no solution:
    for _i in range(50):
        # - update I^(K+1), Z_L
        Z_L = max(Z_L, milp_obj)
        # - do NLP solve given I values, update x^K
        amm_deltas, intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded3(state, partial_intents, full_intents, I=indicators, epsilon=epsilon, buffer_fee=buffer_fee)
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
        amm_deltas, partial_intent_deltas, indicators, new_A, new_A_upper, new_A_lower, milp_obj, valid = _solve_inclusion_problem(state, partial_intents + full_intents, x, Z_U, Z_L, A, A_upper, A_lower, buffer_fee)
        if not valid:
            break

    if valid == True:  # this means we did not get to a solution
        return [[0,0]]*len(intents)

    trade_pcts = [-best_intent_deltas[i] / intent['sell_quantity'] for i, intent in enumerate(partial_intents)]
    new_partial_intents = copy.deepcopy(partial_intents)

    # if solution is not good yet, try scaling down partial intent sizes, to get scaling better
    while len(new_partial_intents) > 0 and (best_status != "Solved" or Z_U > 0) and min(trade_pcts) < 0.05:
        zero_ct = 0
        for i, intent in enumerate(new_partial_intents):
            # we allow new solution to find trade size up to 10x old solution
            new_sell_quantity = min([intent['sell_quantity'] * trade_pcts[i] * 10, intent['sell_quantity']])
            new_buy_quantity = min([intent['buy_quantity'] * trade_pcts[i] * 10, intent['buy_quantity']])
            buy_amt_lrna_value = new_buy_quantity * state.price(state, intent['tkn_buy'])
            selL_amt_lrna_value = new_sell_quantity * state.price(state, intent['tkn_sell'])
            if buy_amt_lrna_value < min_partial and selL_amt_lrna_value < min_partial:
                new_sell_quantity = 0
                new_buy_quantity = 0
                zero_ct += 1
            new_partial_intents[i]['sell_quantity'] = new_sell_quantity
            new_partial_intents[i]['buy_quantity'] = new_buy_quantity

        if zero_ct == m:
            break  # no partial intents are executed, nothing more can be improved

        amm_deltas, intent_deltas, x, obj, dual_obj, temp_status = _find_solution_unrounded3(state, new_partial_intents,
                                                                                        full_intents, I=y_best,
                                                                                        epsilon=epsilon)
        if temp_status in ['PrimalInfeasible', 'DualInfeasible']:
            # the better scaling revealed that there is no actual solution
            return [[0,0]]*len(intents)
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
    best_amm_deltas, best_intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded3(state, new_partial_intents,
                                                                                    full_intents, I=y_best,
                                                                                    flags=flags, epsilon=epsilon)
    linearize = []
    _, max_in, max_out = _calculate_scaling(new_partial_intents, full_intents, y_best, state, asset_list)
    epsilon_tkn_ls = [(max([abs(max_in[t]), abs(max_out[t])]) / state.liquidity[t], t) for t in asset_list]
    epsilon_tkn_ls.sort()
    loc = bisect.bisect_right([x[0] for x in epsilon_tkn_ls], epsilon)
    while status != "Solved" and loc < len(epsilon_tkn_ls) and epsilon_tkn_ls[loc][0] < 1e-4:
        # force linearization of asset with smallest epsilon
        linearize.append(epsilon_tkn_ls[loc][1])
        loc += 1
        best_amm_deltas, best_intent_deltas, x, obj, dual_obj, status = _find_solution_unrounded3(state, new_partial_intents,
                                                                                        full_intents, I=y_best,
                                                                                        flags=flags, epsilon=epsilon,
                                                                                        force_linear = linearize)
    # if status != "Solved":
    #     if obj > 0:
    #         return [[0,0]]*len(intents)  # no solution found
    #     else:
    #         raise
    sell_deltas = round_solution(new_partial_intents, best_intent_deltas)
    partial_deltas_with_buys = add_buy_deltas(new_partial_intents, sell_deltas)
    full_deltas_with_buys = [[-full_intents[l]['sell_quantity'], full_intents[l]['buy_quantity']] if y_best[l] == 1 else [0,0] for l in range(r)]
    deltas = [None] * len(intents)
    for i in range(len(partial_intent_indices)):
        deltas[partial_intent_indices[i]] = partial_deltas_with_buys[i]
    for i in range(len(full_intent_indices)):
        deltas[full_intent_indices[i]] = full_deltas_with_buys[i]
    return deltas
