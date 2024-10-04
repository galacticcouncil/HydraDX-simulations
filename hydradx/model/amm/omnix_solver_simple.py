import numpy as np
import cvxpy as cp

from hydradx.model.amm.omnipool_amm import OmnipoolState


def convert_intents(intents, tkn_list):
    tkn_map = {tkn: i for i, tkn in enumerate(tkn_list)}

    intent_indices = []
    intent_reserves = []
    intent_prices = []
    for intent in intents:
        intent_indices.append([tkn_map[intent['tkn_sell']], tkn_map[intent['tkn_buy']]])
        sell_amt = intent['sell_quantity']
        buy_amt = intent['buy_quantity']
        intent_reserves.append([sell_amt, 0])
        intent_prices.append(buy_amt / sell_amt)

    return intent_indices, intent_reserves, intent_prices


def _find_solution_unrounded(state: OmnipoolState, intents: list, flags: dict = None) -> (dict, list):

    if flags is None:
        flags = {}

    tkn_list = ["LRNA"] + state.asset_list

    reserves_list = [[state.lrna[tkn], state.liquidity[tkn]] for tkn in state.lrna]

    fees = [state.last_fee[tkn] for tkn in state.asset_list]
    lrna_fees = [state.last_lrna_fee[tkn] for tkn in state.asset_list]

    profit_value = [1, 0, 0, 0]  # for taking profits in LRNA

    # transform data for cvxpy
    intent_indices, intent_reserves, intent_prices = convert_intents(intents, tkn_list)
    global_indices = list(range(len(tkn_list)))
    local_indices = [[0, i+1] for i in range(len(state.asset_list))]

    reserves = list(map(np.array, reserves_list))
    reserves2 = list(map(np.array, intent_reserves))

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

    profits = [A_i @ cp.hstack([(1-l) * qL[0] - qD[0], L[0] - D[0]]) for A_i, D, L, qD, qL, l in zip(A, deltas, lambdas, lrna_deltas, lrna_lambdas, lrna_fees)]  # assets from AMMs
    intent_profits = [B_i @ cp.hstack([D[0], -D[0] * p]) for B_i, D, p in zip(B, intent_deltas, intent_prices)]# assets from intents
    total_profits = profits + intent_profits

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


def round_solution(intents, intent_deltas, tolerance=0.0001):
    deltas = []
    for i in range(len(intent_deltas)):
        # don't leave dust in intent due to rounding error
        if intents[i]['sell_quantity'] + intent_deltas[i][0] < tolerance * intents[i]['sell_quantity']:
            deltas.append(-intents[i]['sell_quantity'])
        # don't trade dust amount due to rounding error
        elif -intent_deltas[i][0] <= tolerance * intents[i]['sell_quantity']:
            deltas.append(0)
        else:
            deltas.append(intent_deltas[i][0])
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
