from pprint import pprint

import numpy as np
import cvxpy as cp

from mpmath import mp, mpf


def convert_intents(intents, tkn_list):
    tkn_map = {tkn: i for i, tkn in enumerate(tkn_list)}

    intent_indices = []
    intent_reserves = []
    intent_prices = []
    for intent in intents:
        intent_indices.append([tkn_map[intent['tkn_sell']], tkn_map[intent['tkn_buy']]])
        if 'sell_quantity' in intent:
            sell_amt = intent['sell_quantity']
            buy_amt = intent['buy_limit']
        else:
            sell_amt = intent['sell_limit']
            buy_amt = intent['buy_quantity']
        intent_reserves.append([sell_amt, 0])
        intent_prices.append(buy_amt / sell_amt)

    return intent_indices, intent_reserves, intent_prices


def test_convex():

    # Problem data

    intents = [
        {'sell_quantity': 100, 'buy_limit': 700, 'tkn_sell': 'DOT', 'tkn_buy': 'USDT'},  # selling DOT for $7
        {'sell_quantity': 1500, 'buy_limit': 100000, 'tkn_sell': 'USDT', 'tkn_buy': 'HDX'},  # buying HDX for $0.015
        {'sell_quantity': 400, 'buy_limit': 50, 'tkn_sell': 'USDT', 'tkn_buy': 'DOT'},  # buying DOT for $8
        {'sell_quantity': 100, 'buy_limit': 100, 'tkn_sell': 'HDX', 'tkn_buy': 'USDT'},  # selling HDX for $1
    ]

    asset_list = ["HDX", "DOT", "USDT"]
    tkn_list = ["LRNA"] + asset_list



    reserves_list = [
        [mpf(1000000), mpf(100000000)],  # LRNA, liquidity
        [mpf(10000000), mpf(10000000/7.5)],  # spot price of DOT is $7.5
        [mpf(10000000), mpf(10000000)]
    ]

    fees = [0.003] * 3

    profit_value = [1, 0, 0, 0]  # for taking profits in LRNA

    # transform data for cvxpy
    intent_indices, intent_reserves, intent_prices = convert_intents(intents, tkn_list)
    global_indices = list(range(len(tkn_list)))
    local_indices = [[0, i+1] for i in range(len(asset_list))]

    reserves = list(map(np.array, reserves_list))
    reserves2 = list(map(np.array, intent_reserves))

    # Build local-global matrices
    n = len(global_indices)

    A = []
    for l in local_indices + intent_indices:
        n_i = len(l)
        A_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1
        A.append(A_i)

    # Build variables
    deltas = [cp.Variable(len(l), nonneg=False) for l in local_indices]
    intent_deltas = [cp.Variable(len(l), nonneg=False) for l in intent_indices]

    profits = [A_i @ (-D) for A_i, D, in zip(A, deltas + intent_deltas)]  # assets from AMMs

    psi = cp.sum(profits)

    # Objective is to maximize "total market value" of coins out
    obj = cp.Maximize(profit_value @ psi)  # take profits in LRNA

    # Reserves after trade
    new_reserves = [R + (1 - f_i) * D for R, f_i, D in zip(reserves, fees, deltas)]
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
    for i in (prob.solution.primal_vars):
        if i-1 < amm_ct:  # AMM deltas
            amm_deltas[i-1] = prob.solution.primal_vars[i]
        else:  # intent deltas
            intent_deltas[i-1 - amm_ct] = prob.solution.primal_vars[i]

    print(f"Total output value: {prob.value}")
    pprint(amm_deltas)
    pprint(intent_deltas)