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


def _calculate_tau_phi(intents: list, tkn_list: list, scaling: dict = None) -> tuple:
    n = len(tkn_list)
    m = len(intents)
    tau = sparse.csc_matrix((n, m))
    phi = sparse.csc_matrix((n, m))
    for j, intent in enumerate(intents):
        sell_i = tkn_list.index(intent['tkn_sell'])
        buy_i = tkn_list.index(intent['tkn_buy'])
        if scaling is not None:
            if intent['tkn_sell'] in scaling:
                tau[sell_i, j] = scaling[intent['tkn_sell']]['asset']
                phi[buy_i, j] = scaling[intent['tkn_sell']]['asset']
            elif intent['tkn_sell'] == "LRNA":
                tau[sell_i, j] = scaling[intent['tkn_sell']]['lrna']
                phi[buy_i, j] = scaling[intent['tkn_sell']]['lrna']
        else:
            tau[sell_i, j] = 1
            phi[buy_i, j] = 1
    return tau, phi


def _calculate_scaling(intents: list, state: OmnipoolState, asset_list: list):
    scaling = {tkn: {'asset': 0, 'lrna': 0} for tkn in asset_list}
    for intent in intents:
        if intent['tkn_sell'] != "LRNA":
            scaling[intent['tkn_sell']]['asset'] = max(scaling[intent['tkn_sell']]["asset"], intent['sell_quantity'])
        if intent['tkn_buy'] != "LRNA":
            scaling[intent['tkn_buy']]['asset'] = max(scaling[intent['tkn_buy']]["asset"], intent['buy_quantity'])
    for tkn in scaling:
        if scaling[tkn]["asset"] == 0:
            scaling[tkn]["asset"] = 1
        else:
            scaling[tkn]["asset"] = min(scaling[tkn]["asset"], state.liquidity[tkn])
        # set scaling for LRNA equal to scaling for asset, adjusted by spot price
        scaling[tkn]["lrna"] = scaling[tkn]["asset"] * state.lrna[tkn] / state.liquidity[tkn]
        scaling[tkn]["lrna"] = min(scaling[tkn]["lrna"], state.lrna[tkn])
    return scaling


def _find_solution_unrounded2(state: OmnipoolState, intents: list) -> (dict, list):

    asset_list = []
    for intent in intents:
        if intent['tkn_sell'] != "LRNA" and intent['tkn_sell'] not in asset_list:
            asset_list.append(intent['tkn_sell'])
        if intent['tkn_sell'] != "LRNA" and intent['tkn_buy'] not in asset_list:
            asset_list.append(intent['tkn_buy'])

    tkn_list = ["LRNA"] + asset_list

    lrna_reserves = [float(state.lrna[tkn]) for tkn in asset_list]
    asset_reserves = [float(state.liquidity[tkn]) for tkn in asset_list]

    fees = [float(state.last_fee[tkn]) for tkn in asset_list]  # f_i
    lrna_fees = [float(state.last_lrna_fee[tkn]) for tkn in asset_list]  # l_i

    n = len(asset_list)
    m = len(intents)
    k = 4 * n + m

    # calculate tau, phi
    # scaling = {tkn: state.liquidity[tkn] for tkn in state.liquidity}
    # scaling['LRNA'] = state.lrna_total
    scaling = _calculate_scaling(intents, state, asset_list)
    tau, phi = _calculate_tau_phi(intents, tkn_list, scaling)

    #----------------------------#
    #          OBJECTIVE         #
    #----------------------------#

    P = sparse.csc_matrix((k, k))

    delta_lrna_coefs = np.array(lrna_reserves)  # need to multiply by each Qi
    lambda_lrna_coefs = np.array([lrna_reserves[i] * (l - 1) for i, l in enumerate(lrna_fees)])  # need to multiply by each Qi
    zero_coefs = np.zeros(2 * n)
    d_coefs = -(tau[0, :].toarray()[0])

    # Concatenate the segments to form q
    q = np.concatenate([delta_lrna_coefs, lambda_lrna_coefs, zero_coefs, d_coefs])

    #----------------------------#
    #        CONSTRAINTS         #
    #----------------------------#

    # all variables are non-negative
    A1 = -sparse.identity(k, format='csc')
    b1 = np.zeros(k)
    cone1 = cb.NonnegativeConeT(k)

    # intents cannot sell more than they have
    amm_coefs = sparse.csc_matrix((m, 4*n))
    d_coefs = sparse.identity(m, format='csc')
    A2 = sparse.hstack([amm_coefs, d_coefs], format='csc')
    intent_scalars = [scaling[i['tkn_buy']]['lrna'] if i['tkn_sell'] == "LRNA" else scaling[i['tkn_sell']]['asset'] for i in intents]
    b2 = np.array([float(intents[i]['sell_quantity']/intent_scalars[i]) for i in range(len(intents))])
    cone2 = cb.NonnegativeConeT(m)

    # leftover must be higher than required fees
    # LRNA
    A30 = sparse.csc_matrix(q)
    b30 = np.zeros(1)
    # other assets
    intent_prices = [float(intent['buy_quantity'] / intent['sell_quantity']) for intent in intents]
    lrna_coefs = sparse.csc_matrix((n, 2*n))
    delta_coefs = sparse.diags(asset_reserves, format='csc')
    lambda_coefs = sparse.diags([asset_reserves[i]*(f-1) for i, f in enumerate(fees)], format='csc')
    d_coefs = sparse.csc_matrix([[phi[i,j]*intent_prices[j] - tau[i, j] for j in range(m)] for i in range(1,n+1)])
    A31 = sparse.hstack([lrna_coefs, delta_coefs, lambda_coefs, d_coefs], format='csc')
    b31 = np.zeros(n)
    A3 = sparse.vstack([A30, A31], format='csc')
    b3 = np.concatenate([b30, b31])
    cone3 = cb.NonnegativeConeT(n + 1)

    # AMM invariants must not go down
    A4 = sparse.csc_matrix((3 * n, k))
    b4 = np.ones(3 * n)
    cones4 = []
    for i in range(n):  # affected rows are 3i through 3i+2
        A4[3*i, i] = -1
        A4[3*i, n+i] = 1
        A4[3*i+1, 2*n+i] = -1
        A4[3*i+1, 3*n+i] = 1
        cones4.append(cb.PowerConeT(0.5))

    A = sparse.vstack([A1, A2, A3, A4], format='csc')
    b = np.concatenate([b1, b2, b3, b4])
    cones = [cone1, cone2, cone3] + cones4

    # solve
    settings = clarabel.DefaultSettings()
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    x = solution.x
    z = solution.z
    s = solution.s

    new_amm_deltas = {}
    exec_intent_deltas = [None] * len(intents)
    for i in range(n):
        tkn = tkn_list[i+1]
        new_amm_deltas[tkn] = (x[2*n+i] - x[3*n+i]) * asset_reserves[i]
    for i in range(len(intents)):
        exec_intent_deltas[i] = -x[4 * n + i] * intent_scalars[i]

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
    sell_deltas = round_solution(intents, intent_deltas)
    return add_buy_deltas(intents, sell_deltas)
