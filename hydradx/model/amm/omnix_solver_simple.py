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


def _find_solution_unrounded(state: OmnipoolState, intents: list):

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
    deltas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    intent_deltas = [cp.Variable(len(l), nonneg=False) for l in intent_indices]

    # profits = [A_i @ (L-D) for A_i, D, L in zip(A, deltas, lambdas)]  # assets from AMMs
    profits = [A_i @ (cp.hstack([L[0], L[1] * (1-l)]) - D) for A_i, D, L, l in zip(A, deltas, lambdas, lrna_fees)]  # assets from AMMs
    intent_profits = [B_i @ (-D) for B_i, D in zip(B, intent_deltas)]  # assets from intents
    total_profits = profits + intent_profits

    psi = cp.sum(total_profits)

    # Objective is to maximize "total market value" of coins out
    obj = cp.Maximize(profit_value @ psi)  # take profits in LRNA

    # Reserves after trade
    # new_reserves = [R + (1 - f_i) * (D - L) for R, f_i, D, L in zip(reserves, fees, deltas, lambdas)]
    new_reserves = [R + D - cp.hstack([L[0], L[1] / (1 - f_i)]) for R, f_i, D, L in zip(reserves, fees, deltas, lambdas)]
    new_intent_reserves = [R + D for R, D in zip(reserves2, intent_deltas)]

    cons = [psi >= 0]  # no negative profits
    for i in range(len(reserves)):  # AMM invariants must not go down
        cons.append(cp.geo_mean(new_reserves[i]) >= cp.geo_mean(reserves[i]))  # this doesn't account for fees correctly

    for i in range(len(intent_indices)):  # intent constraints
        cons.append(intent_prices[i] * intent_deltas[i][0] + intent_deltas[i][1] == 0)  # sale must be at set price
        cons.append(intent_deltas[i][0] <= 0)  # cannot buy the sell asset
        cons.append(new_intent_reserves[i][0] >= 0)  # cannot sell more than you have

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True)

    # extract solution
    amm_ct = len(local_indices)
    amm_deltas = [None] * amm_ct
    intent_deltas = [None] * len(intent_indices)
    for i in range(amm_ct):
        amm_deltas[i] = prob.solution.primal_vars[i+1] - prob.solution.primal_vars[amm_ct+i+1]
    for i in range(len(intent_indices)):
        intent_deltas[i] = prob.solution.primal_vars[2*amm_ct+i+1]

    return amm_deltas, intent_deltas


def round_solution(intents, intent_deltas, tolerance=0.0001):
    for i in range(len(intent_deltas)):
        # don't leave dust in intent due to rounding error
        if intents[i]['sell_quantity'] + intent_deltas[i][0] < tolerance * intents[i]['sell_quantity']:
            intent_deltas[i][0] = -intents[i]['sell_quantity']
            intent_deltas[i][1] = intents[i]['buy_quantity']
        # don't trade dust amount due to rounding error
        elif -intent_deltas[i][0] <= tolerance * intents[i]['sell_quantity']:
            intent_deltas[i][0] = 0
            intent_deltas[i][1] = 0
        # guarantee that price is better than limit
        # elif abs(intent_deltas[i][1] / intent_deltas[i][0]) < abs(intents[i]['buy_quantity'] / intents[i]['sell_quantity']):
        #     raise
    return intent_deltas


def find_solution(state: OmnipoolState, intents: list):
    amm_deltas, intent_deltas = _find_solution_unrounded(state, intents)
    return round_solution(intents, intent_deltas)
