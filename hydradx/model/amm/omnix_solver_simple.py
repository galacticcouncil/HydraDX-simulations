import math

import clarabel
import numpy as np
import cvxpy as cp
import clarabel as cb
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
        phi[buy_i, j] = scaling[intent['tkn_sell']] / scaling[intent['tkn_buy']]
    return tau, phi


def _calculate_scaling(intents: list, state: OmnipoolState, asset_list: list):
    scaling = {tkn: 0 for tkn in asset_list}
    net_supply = {tkn: 0 for tkn in asset_list + ['LRNA']}
    net_demand = {tkn: 0 for tkn in asset_list}
    scaling["LRNA"] = float('inf')
    for intent in intents:
        if intent['tkn_sell'] != "LRNA":
            scaling[intent['tkn_sell']] = max(scaling[intent['tkn_sell']], intent['sell_quantity'])
        if intent['tkn_buy'] != "LRNA":
            scaling[intent['tkn_buy']] = max(scaling[intent['tkn_buy']], intent['buy_quantity'])
        net_supply[intent['tkn_sell']] += intent['sell_quantity']
        net_demand[intent['tkn_buy']] += intent['buy_quantity']
    for tkn in asset_list:
        if scaling[tkn] == 0:
            scaling[tkn] = 1
        else:
            scaling[tkn] = min(scaling[tkn], state.liquidity[tkn])
        # set scaling for LRNA equal to scaling for asset, adjusted by spot price
        scalar = scaling[tkn] * state.lrna[tkn] / state.liquidity[tkn]
        scaling["LRNA"] = min(scaling["LRNA"], scalar)
    return scaling, net_supply, net_demand


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
        intents: list,
        flags: dict = None,
        epsilon: float = 1e-5,
        fee_buffer: float = 0.0001
) -> (dict, list):

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
            indices_to_keep.remove(2 * n + asset_list.index(tkn))  # lrna_lambda_i is zero
        elif directions[tkn] == "buy":
            indices_to_keep.remove(3 * n + asset_list.index(tkn))  # lambda_i is zero

    tkn_list = ["LRNA"] + asset_list

    intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) for intent in intents]

    fees = [float(state.last_fee[tkn]) for tkn in asset_list]  # f_i
    lrna_fees = [float(state.last_lrna_fee[tkn]) for tkn in asset_list]  # l_i
    fee_match = 0.0005
    assert fee_match <= min(fees)  # breaks otherwise

    # calculate tau, phi
    scaling, net_supply, net_demand = _calculate_scaling(intents, state, asset_list)
    tau, phi = _calculate_tau_phi(intents, tkn_list, scaling)
    epsilon_tkn = {t: max([net_supply[t], net_demand[t]]) / state.liquidity[t] for t in asset_list}

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    k_real = len(indices_to_keep)
    P_trimmed = sparse.csc_matrix((k_real, k_real))

    y_coefs = np.ones(n)
    x_coefs = np.zeros(n)
    lrna_lambda_coefs = np.array(lrna_fees)
    lambda_coefs = np.zeros(n)
    d_coefs = np.array([-tau[0,j] for j in range(m)])
    q = np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs])
    q_trimmed = np.array([q[i] for i in indices_to_keep])

    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#

    # intent variables are non-negative
    diff_coefs = sparse.csc_matrix((2*n + m,2*n))
    nonzero_coefs = -sparse.identity(2 * n + m, format='csc')
    A1 = sparse.hstack([diff_coefs, nonzero_coefs])
    b1 = np.zeros(2 * n + m)
    cone1 = cb.NonnegativeConeT(2 * n + m)
    A1_trimmed = A1[:, indices_to_keep]

    # intents cannot sell more than they have
    amm_coefs = sparse.csc_matrix((m, 4*n))
    d_coefs = sparse.identity(m, format='csc')
    A2 = sparse.hstack([amm_coefs, d_coefs], format='csc')
    b2 = np.array([float(i['sell_quantity']/scaling[i['tkn_sell']]) for i in intents])
    A2_trimmed = A2[:, indices_to_keep]
    cone2 = cb.NonnegativeConeT(m)

    # leftover must be higher than required fees
    # LRNA
    y_coefs = np.ones(n)
    x_coefs = np.zeros(n)
    lrna_lambda_coefs = np.array(lrna_fees)
    lambda_coefs = np.zeros(n)
    d_coefs = -(tau[0, :].toarray()[0])
    A30 = sparse.csc_matrix(np.concatenate([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs]))
    b30 = np.zeros(1)
    # other assets
    y_coefs = sparse.csc_matrix((n,n))
    x_coefs = sparse.identity(n, format='csc')
    lrna_lambda_coefs = sparse.csc_matrix((n,n))
    lambda_coefs = sparse.diags(np.array(fees)-fee_match, format='csc')
    d_coefs = sparse.csc_matrix([[1/(1-fee_match)*phi[i,j]*intent_prices[j] - tau[i, j] for j in range(m)] for i in range(1,n+1)])
    A31 = sparse.hstack([y_coefs, x_coefs, lrna_lambda_coefs, lambda_coefs, d_coefs])
    b31 = np.zeros(n)
    A3 = sparse.vstack([A30, A31], format='csc')
    A3_trimmed = A3[:, indices_to_keep]
    b3 = np.concatenate([b30, b31])
    cone3 = cb.NonnegativeConeT(n + 1)

    # AMM invariants must not go down
    A4 = sparse.csc_matrix((0, k))
    b4 = np.array([])
    cones4 = []
    for i in range(n):
        tkn = asset_list[i]
        if epsilon_tkn[tkn] <= epsilon:  # linearize the AMM constraint
            if tkn not in directions:
                c1 = 1 / (1 + epsilon_tkn[tkn])
                c2 = 1 / (1 - epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((2, k))
                b4i = np.zeros(2)
                A4i[0, i] = -scaling["LRNA"]/state.lrna[tkn]
                A4i[0, n+i] = -scaling[tkn]/state.liquidity[tkn] * c1
                A4i[1, i] = -scaling["LRNA"]/state.lrna[tkn]
                A4i[1, n + i] = -scaling[tkn] / state.liquidity[tkn] * c2
                cones4.append(cb.NonnegativeConeT(2))
            else:
                if directions[tkn] == "sell":
                    c = 1 / (1 - epsilon_tkn[tkn])
                else:
                    c = 1 / (1 + epsilon_tkn[tkn])
                A4i = sparse.csc_matrix((1, k))
                b4i = np.zeros(1)
                A4i[0, i] = -scaling["LRNA"] / state.lrna[tkn]
                A4i[0, n+i] = -scaling[tkn]/state.liquidity[tkn] * c
                cones4.append(cb.ZeroConeT(1))
        else:  # full AMM constraint
            A4i = sparse.csc_matrix((3, k))
            b4i = np.ones(3)
            A4i[0, i] = -scaling["LRNA"]/state.lrna[tkn]
            A4i[1, n+i] = -scaling[tkn]/state.liquidity[tkn]
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
    solver = clarabel.DefaultSolver(P_trimmed, q_trimmed, A, b, cones, settings)
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
        new_amm_deltas[tkn] = x_expanded[n+i] * scaling[tkn]

    for i in range(len(intents)):
        exec_intent_deltas[i] = -x_expanded[4 * n + i] * scaling[intents[i]['tkn_sell']]

    return new_amm_deltas, exec_intent_deltas


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
    amm_deltas, intent_deltas = _find_solution_unrounded3(state, intents, flags, epsilon = epsilon)
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)
