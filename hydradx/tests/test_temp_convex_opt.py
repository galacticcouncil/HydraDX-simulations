import numpy as np
import cvxpy as cp

from mpmath import mp, mpf


def convert_intents_to_cvxpy(intents, tkn_list):
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
        # {'sell_quantity': 1500, 'buy_limit': 100000, 'tkn_sell': 'USDT', 'tkn_buy': 'HDX'}
    ]

    tkn_list = ["LRNA", "HDX", "DOT", "USDT"]

    intent_indices, intent_reserves, intent_prices = convert_intents_to_cvxpy(intents, tkn_list)

    global_indices = list(range(len(tkn_list)))


    local_indices = [  # AMMs
        [0, 1],  # HDX
        [0, 2],  # DOT
        [0, 3]  # USDT
    ]

    reserves_list = [
        [mpf(1000000), mpf(100000000)],
        [mpf(10000000), mpf(10000000/7.5)],  # spot price of DOT is $7.5
        [mpf(10000000), mpf(10000000)]
    ]

    reserves = list(map(np.array, reserves_list + intent_reserves))

    fees = [0.003] * 3
    # fees = [0.0] * 3
    fees.extend([0.0]*len(intents))  # intents

    # "Market value" of tokens (say, in a centralized exchange)
    market_value = [
        1,
        0.01,
        7.5,
        1
    ]

    profit_value = [1, 0, 0, 0]  # for taking profits in LRNA

    # Build local-global matrices
    n = len(global_indices)
    m = len(local_indices + intent_indices)

    A = []
    for l in local_indices + intent_indices:
        n_i = len(l)
        A_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1
        A.append(A_i)

    # Build variables
    deltas = [cp.Variable(len(l), nonneg=False) for l in local_indices]
    # lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    intent_deltas = [cp.Variable(len(l), nonneg=False) for l in intent_indices]
    # intent_lambdas = [cp.Variable(len(l), nonneg=True) for l in intent_indices]

    # profits = [A_i @ (L - D) for A_i, D, L in zip(A, deltas + intent_deltas, lambdas + intent_lambdas)]  # assets from AMMs
    profits = [A_i @ (-D) for A_i, D, in zip(A, deltas + intent_deltas)]  # assets from AMMs

    psi = cp.sum(profits)

    # Objective is to maximize "total market value" of coins out
    # obj = cp.Maximize(market_value @ psi)
    obj = cp.Maximize(profit_value @ psi)  # take profits in LRNA

    # Reserves after trade
    # new_reserves = [R + (1-f_i) * D - L for R, f_i, D, L in zip(reserves, fees, deltas + intent_deltas, lambdas + intent_lambdas)]
    new_reserves = [R + (1 - f_i) * D for R, f_i, D in zip(reserves, fees, deltas + intent_deltas)]

    cons = [
        # Uniswap v2 pools
        cp.geo_mean(new_reserves[0]) >= cp.geo_mean(reserves[0]),
        cp.geo_mean(new_reserves[1]) >= cp.geo_mean(reserves[1]),
        cp.geo_mean(new_reserves[2]) >= cp.geo_mean(reserves[2]),

        # Arbitrage constraint
        psi >= 0
    ]

    m_amm = len(local_indices)
    for i in range(len(intent_indices)):
        # cons.append(intent_prices[i] * new_reserves[m_amm+i][0] + new_reserves[m_amm+i][1] == intent_prices[i] * new_reserves[m_amm+i][0] + new_reserves[m_amm+i][1])
        cons.append(intent_prices[i] * intent_deltas[i][0] + intent_deltas[i][1] == 0)
        cons.append(intent_deltas[i][0] <= 0)
        # cons.append(intent_lambdas[i][1] == 0)
        cons.append(new_reserves[m_amm+i] >= 0)

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True)

    print(f"Total output value: {prob.value}")