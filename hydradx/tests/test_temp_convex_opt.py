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
        {'sell_quantity': 100, 'buy_limit': 700, 'tkn_sell': 'DOT', 'tkn_buy': 'USDT'},
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
        [mpf(10000000), mpf(10000000/7.5)],
        [mpf(10000000), mpf(10000000)]
    ]

    reserves = list(map(np.array, reserves_list + intent_reserves))

    fees = [0.003] * 3
    fees.extend([0.0]*len(intents))  # intents

    # "Market value" of tokens (say, in a centralized exchange)
    market_value = [
        1,
        0.01,
        7.5,
        1
    ]

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
    deltas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    # intent_deltas = [cp.reshape(cp.vstack([cp.Constant(0), cp.Variable(1, nonneg=True)]), (2,)) for l in intent_indices]
    # intent_lambdas = [cp.reshape(cp.vstack([intent_deltas[i][1] * intent_prices[i], cp.Constant(0)]), (2,))for i in range(len(intent_deltas))]
    # intent_lambdas = [cp.reshape(cp.vstack([cp.Variable(1, nonneg=True), cp.Constant(0)]), (2,)) for l in intent_indices]
    intent_deltas = [cp.Variable(len(l), nonneg=True) for l in intent_indices]
    intent_lambdas = [cp.Variable(len(l), nonneg=True) for l in intent_indices]

    # intent_amts_sold = [cp.Variable(1, nonneg=True) for _ in intent_indices]
    # intent_amts_bought = [(intent_reserves[i][0] - intent_amts_sold[i]) * intent_prices[i] for i in range(len(intent_indices))]

    # psi = cp.sum(
    #     [A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)] +
    #     # [B_i @ [-D, L] for B_i, D, L in zip(B, intent_deltas, intent_lambdas)]  # (L - D) = (-D, L)
    #     [[[r_ij[0] * (-D), r_ij[1] * L] for r_ij in B_i] for B_i, D, L in zip(B, intent_deltas, intent_lambdas)]
    # )

    profits = [A_i @ (L - D) for A_i, D, L in zip(A, deltas + intent_deltas, lambdas + intent_lambdas)]  # assets from AMMs
    # for i in range(len(intents)):
    #     B_i = B[i]
    #     sell_amt = intent_lambdas[i]
    #     buy_amt = sell_amt * intent_prices[i]
    #     delta = [0] * n
    #     Ls, Ds = [0,0], [0] * n
    #     for i in range(len(B_i)):
    #         r_ij = B_i[i]
    #         if r_ij[0] != 0:
    #             Ds[i] = D
    #         elif r_ij[1] != 0:
    #             Ls[i] = L
    #     temp = B_i @ (Ls - Ds)
    #     profits.append(delta)
    psi = cp.sum(profits)

    # Objective is to maximize "total market value" of coins out
    obj = cp.Maximize(market_value @ psi)

    # Reserves after trade
    new_reserves = [R + (1-f_i) * D - L for R, f_i, D, L in zip(reserves, fees, deltas + intent_deltas, lambdas + intent_lambdas)]
    # new_intent_reserves = [[R[0] - L, R[1] + D] for R, D, L in zip(intent_reserves, intent_deltas, intent_lambdas)]

    # new_intent_reserves = [R + D - L for R, D, L in zip(intent_reserves, intent_deltas, intent_lambdas)]

    # Intent, buy DOT at price of 8 USDT

    # intent_price = 7
    # hdx_buy_price = 0.015

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
        cons.append(intent_prices[i] * new_reserves[m_amm+i][0] + new_reserves[m_amm+i][1] == intent_prices[0] * new_reserves[m_amm+i][0] + new_reserves[m_amm+i][1])
        cons.append(intent_deltas[i][0] == 0)
        cons.append(intent_lambdas[i][1] == 0)
        cons.append(new_reserves[m_amm+i] >= 0)

    # Set up and solve problem
    prob = cp.Problem(obj, cons)
    prob.solve(verbose=True)

    print(f"Total output value: {prob.value}")