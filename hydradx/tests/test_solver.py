import copy
from pprint import pprint
import random

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity, Phase, reproduce_failure

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, simulate_swap
from mpmath import mp, mpf
import highspy
import numpy as np

from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution, \
    _find_solution_unrounded, add_buy_deltas, round_solution, find_solution_outer_approx, _solve_inclusion_problem, \
    ICEProblem, _get_leftover_bounds
from hydradx.model.amm.stableswap_amm import StableSwapPoolState


#######################
# Omnipool only tests #
#######################

def test_single_trade_settles():
    agents = [Agent(holdings={'DOT': 100, 'LRNA': 750})]

    init_intents_partial = [  # selling DOT for $7
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True}
    ]
    init_intents_full = [  # selling DOT for $7
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False}
    ]
    init_intents_partial_lrna = [
        {'sell_quantity': mpf(750), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': True}
    ]
    init_intents_full_lrna = [
        {'sell_quantity': mpf(750), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': False}
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intents = copy.deepcopy(init_intents_partial)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == -init_intents_partial[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_partial[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_full)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == -init_intents_full[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_full[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_partial_lrna)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == -init_intents_partial_lrna[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_partial_lrna[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_full_lrna)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == -init_intents_full_lrna[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_full_lrna[0]['buy_quantity']


def test_single_trade_does_not_settle():
    agents = [Agent(holdings={'DOT': 100, 'USDT': 0})]

    init_intents_partial = [  # selling DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(800), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True}
    ]
    init_intents_full = [  # selling DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(800), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False}
    ]
    init_intents_partial_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': True}
    ]
    init_intents_full_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': False}
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intents = copy.deepcopy(init_intents_partial)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_partial_lrna)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full_lrna)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]
    omnipool_deltas = x[4]
    amm_deltas = x[5]
    assert validate_and_execute_solution(initial_state.copy(), [], intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0



###############
# Other tests #
###############

# @reproduce_failure('6.39.6', b'AAEZHfyXOrk=')
@given(st.floats(min_value=1e-7, max_value=0.01))
@settings(verbosity=Verbosity.verbose, print_blob=True)
def test_fuzz_single_trade_settles(size_factor: float):

    # AMM setup

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    initial_state.last_fee = {tkn: 0.00 for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: 0.00 for tkn in lrna}

    ss_fee = 0.0005

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]
    # amm_list = [stablepool, stablepool4]
    # amm_list = [stablepool, stablepool_btc]

    router = OmnipoolRouter([initial_state] + amm_list)

    # trade setup
    tkn_sell, tkn_buy = "DOT", "WETH"
    # size_factor = 0.001  # pct of total liquidity that is being traded
    partial = True
    # get buy amount, sell amount from size_factor
    total_buy_liq = initial_state.liquidity[tkn_buy] if tkn_buy in initial_state.liquidity else 0
    total_sell_liq = initial_state.liquidity[tkn_sell] if tkn_sell in initial_state.liquidity else 0
    for amm in amm_list:
        if amm.unique_id == tkn_buy:
            total_buy_liq = max(amm.shares, total_buy_liq)
        elif amm.unique_id == tkn_sell:
            total_sell_liq = max(amm.shares, total_sell_liq)
        else:
            total_buy_liq += amm.liquidity[tkn_buy] if tkn_buy in amm.liquidity else 0
            total_sell_liq += amm.liquidity[tkn_sell] if tkn_sell in amm.liquidity else 0
    if tkn_buy == "LRNA":
        total_buy_liq = initial_state.lrna["DOT"]
    elif tkn_sell == "LRNA":
        total_sell_liq = initial_state.lrna["DOT"]
    max_buy_amount = total_buy_liq * size_factor
    max_sell_amount = total_sell_liq * size_factor
    agent = Agent(holdings={tkn_sell: max_sell_amount})
    # get sell_amount by simulating swap
    test_state, test_agent = router.simulate_swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=max_buy_amount)
    if test_state.fail == '':  # swap was succesful
        sell_amt = agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
        intent = {'sell_quantity': sell_amt / 0.999, 'buy_quantity': max_buy_amount, 'tkn_sell': tkn_sell,
                  'tkn_buy': tkn_buy, 'agent': Agent(holdings={tkn_sell: sell_amt / 0.999}), 'partial': partial}
    else:  # swap was unsuccesful
        test_state, test_agent = router.simulate_swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=max_sell_amount)
        buy_amt = test_agent.holdings[tkn_buy]
        intent = {'sell_quantity': max_sell_amount, 'buy_quantity': buy_amt * 0.999, 'tkn_sell': tkn_sell,
                  'tkn_buy': tkn_buy, 'agent': Agent(holdings={tkn_sell: max_sell_amount}), 'partial': partial}

    intents = [intent]
    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")

    assert valid


def test_matching_trades_execute_more():
    agents = [Agent(holdings={'DOT': 1000, 'LRNA': 7500}), Agent(holdings={'USDT': 7600})]

    intent1 = {  # selling DOT for $7.49
        'sell_quantity': mpf(1000), 'buy_quantity': mpf(7470), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True
    }

    intent2 = {  # buying DOT for $7.51
        'sell_quantity': mpf(7530), 'buy_quantity': mpf(1000), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1], 'partial': True
    }

    intent1_lrna = {  # selling DOT for $7.49
        'sell_quantity': mpf(7500), 'buy_quantity': mpf(7480), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True
    }

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    # do the DOT sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x[0]
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_buy, intents_buy)
    buy_deltas = x[0]
    assert validate_and_execute_solution(state_buy, intents_buy, buy_deltas)

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x[0]
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) > 0
    assert abs(buy_deltas[0][0]) > 0
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])

    # do the LRNA sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1_lrna)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x[0]
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do both LRNA sale & DOT buy together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1_lrna), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x[0]
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) > 0
    assert abs(buy_deltas[0][0]) > 0
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])

def test_matching_trades_execute_more_full_execution():
    agents = [Agent(holdings={'DOT': 1000, 'LRNA': 7500}), Agent(holdings={'USDT': 7600})]

    intent1 = {  # selling DOT for $7.49
        'sell_quantity': mpf(1000), 'buy_quantity': mpf(7470), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False
    }

    intent2 = {  # buying DOT for $7.51
        'sell_quantity': mpf(7530), 'buy_quantity': mpf(1000), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1], 'partial': False
    }

    intent1_lrna = {  # selling DOT for $7.49
        'sell_quantity': mpf(7500), 'buy_quantity': mpf(7480), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False
    }

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    # do the DOT sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x[0]
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_buy, intents_buy)
    buy_deltas = x[0]
    assert validate_and_execute_solution(state_buy, intents_buy, buy_deltas)

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x[0]
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) == 0
    assert abs(buy_deltas[0][0]) == 0
    assert abs(match_deltas[0][0]) > 0
    assert abs(match_deltas[1][0]) > 0

    # do the LRNA sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1_lrna)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x[0]
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do both LRNA sale & DOT buy together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1_lrna), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x[0]
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) == 0
    assert abs(buy_deltas[0][0]) == 0
    assert abs(match_deltas[0][0]) > 0
    assert abs(match_deltas[1][0]) > 0

def test_convex():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)


def test_with_lrna_intent():
    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'LRNA': 1000}),
        Agent(holdings={'DOT': 1000000})
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(100), 'tkn_sell': 'LRNA', 'tkn_buy': 'DOT', 'agent': agents[4], 'partial': True},  # buying DOT for $10
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False},  # selling DOT for $7
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)


def test_small_trade():  # this is to test that rounding errors don't screw up small trades
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'CRU': 5}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        {'sell_quantity': mpf(1.150), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)
    assert intent_deltas[0][0] == -intents[0]['sell_quantity']
    assert intent_deltas[0][1] == intents[0]['buy_quantity']
    assert intent_deltas[1][0] == 0
    assert intent_deltas[1][1] == 0

@given(st.floats(min_value=1e-7, max_value=1e-3))
@settings(verbosity=Verbosity.verbose, print_blob=True)
def test_inclusion_problem_small_trade_fuzz(trade_size_pct: float):
    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    buy_tkn = 'DOT'
    selL_tkn = '2-Pool'
    buy_amt = trade_size_pct * liquidity[buy_tkn]
    # buy_amt = mpf(.01)
    price = initial_state.price(initial_state, buy_tkn, selL_tkn)
    sell_amt = buy_amt * price * 1.01
    # sell_amt = mpf(.05)
    agents = [Agent(holdings={selL_tkn: sell_amt})]

    intents = [
        {'sell_quantity': sell_amt, 'buy_quantity': buy_amt, 'tkn_sell': selL_tkn, 'tkn_buy': buy_tkn, 'agent': agents[0], 'partial': False},
    ]

    # intent_deltas, _ = find_solution_outer_approx(initial_state, intents)
    p = ICEProblem(initial_state, intents)
    p.set_up_problem()

    inf = highspy.kHighsInf
    Z_L = -inf
    Z_U = 0
    x = np.zeros((1,13))
    A, A_upper, A_lower = np.zeros((0, 13)), np.array([]), np.array([])
    # - get new cone constraint from I^K
    indicators = [0]
    BK = np.where(np.array(indicators) == 1)[0] + 12
    NK = np.where(np.array(indicators) == 0)[0] + 12
    IC_A = np.zeros((1, 13))
    IC_A[0, BK] = 1
    IC_A[0, NK] = -1
    IC_upper = np.array([len(BK) - 1])
    IC_lower = np.array([-inf])

    # - add cone constraint to A, A_upper, A_lower
    A = np.vstack([A, IC_A])
    A_upper = np.concatenate([A_upper, IC_upper])
    A_lower = np.concatenate([A_lower, IC_lower])
    amm_deltas, partial_intent_deltas, indicators, new_A, new_A_upper, new_A_lower, milp_obj, valid, status = _solve_inclusion_problem(p, x, Z_U, Z_L, A, A_upper, A_lower)
    assert indicators[0] == 1
    assert str(status) == 'HighsModelStatus.kOptimal'
    # assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)
    # assert intent_deltas[0][0] == -intents[0]['sell_quantity']
    # assert intent_deltas[0][1] == pytest.approx(intents[0]['buy_quantity'], rel=1e-10)

@given(st.floats(min_value=1e-10, max_value=1e-3))
@settings(verbosity=Verbosity.verbose, print_blob=True)
def test_small_trade_fuzz(trade_size_pct: float):  # this is to test that rounding errors don't screw up small trades

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    buy_tkn = 'DOT'
    selL_tkn = '2-Pool'
    buy_amt = trade_size_pct * liquidity[buy_tkn]
    price = initial_state.price(initial_state, buy_tkn, selL_tkn)
    sell_amt = buy_amt * price * 1.01
    agents = [Agent(holdings={selL_tkn: sell_amt})]

    intents = [
        {'sell_quantity': sell_amt, 'buy_quantity': buy_amt, 'tkn_sell': selL_tkn, 'tkn_buy': buy_tkn, 'agent': agents[0], 'partial': True},
    ]

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)
    assert intent_deltas[0][0] == -intents[0]['sell_quantity']
    assert intent_deltas[0][1] == pytest.approx(intents[0]['buy_quantity'], rel=1e-10)


def test_solver_with_real_omnipool_one_full():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0],
         'partial': False},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
         'partial': True},
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    full_intent_indicators = [1]

    p = ICEProblem(initial_state, intents, min_partial = 0)
    p.set_up_problem(I = full_intent_indicators)

    amm_deltas, sell_deltas, _, _, _, _ = _find_solution_unrounded(p)
    for i in p.full_intents:
        if full_intent_indicators.pop(0) == 1:
            sell_deltas.append(-i['sell_quantity'])

    sell_deltas = round_solution(p.partial_intents + p.full_intents, sell_deltas)
    intent_deltas = add_buy_deltas(p.partial_intents + p.full_intents, sell_deltas)

    assert sell_deltas[0] == -100
    assert sell_deltas[1] == -100
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(p.partial_intents + p.full_intents), intent_deltas)

    pprint(intent_deltas)


def test_full_solver():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x[0]

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)

    pprint(intent_deltas)

@given(st.lists(st.floats(min_value=1e-10, max_value=0.5), min_size=3, max_size=3),
       st.lists(st.floats(min_value=0.9, max_value=1.1), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=18), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=17), min_size=3, max_size=3),
        st.lists(st.booleans(), min_size=3, max_size=3)
       )
@settings(print_blob=True, verbosity=Verbosity.verbose, deadline=None, phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target))
def test_solver_random_intents(sell_ratios, price_ratios, sell_is, buy_is, partial_flags):

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    good_indices = [i for i in range(len(sell_is)) if sell_is[i]-1 != buy_is[i]]
    intents = []
    for i in good_indices:
        sell_tkn = initial_state.asset_list[sell_is[i]-1] if sell_is[i] > 0 else "LRNA"
        buy_tkn = initial_state.asset_list[buy_is[i]]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(initial_state, sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})

    intent_deltas, predicted_profit, _, _ = find_solution_outer_approx(initial_state, intents)

    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas, "HDX")
    assert valid
    abs_error = predicted_profit - profit
    if profit > 0:
        pct_error = abs_error/profit
        assert pct_error < 0.01 or abs_error < 1
        assert abs(pct_error) < 0.10 or abs(abs_error) < 100
    else:
        assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_case_Martin():

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    agent = Agent(holdings={'GLMR': 1001500})

    intents = [
        {'sell_quantity': mpf(1001497.604662274886037302), 'buy_quantity': mpf(1081639.587746551400027), 'tkn_sell': 'GLMR', 'tkn_buy': 'KILT', 'agent': agent, 'partial': True},
    ]

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas, predicted_profit, Z_Ls, Z_Us = x[0], x[1], x[2], x[3]
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas, "HDX")
    assert valid
    assert profit == 0


def test_more_random_intents():
    r = 50
    random.seed(r)
    np.random.seed(r)

    intent_ct = 500
    min_sell_ratio, max_sell_ratio = 1e-10, 0.01
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    min_price_ratio, max_price_ratio = 0.99, 1.01
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    initial_state.last_fee = {tkn: 0.0025 for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: 0.0005 for tkn in lrna}

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(initial_state.asset_list + ['LRNA'], 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(initial_state, sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})
        # intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
        #                 'tkn_buy': buy_tkn, 'agent': agent, 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 100000, 'tkn_sell': '2-Pool', 'tkn_buy': 'HDX', 'agent': Agent(holdings={"2-Pool": 10}), 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 4000, 'tkn_sell': 'DOT', 'tkn_buy': '2-Pool', 'agent': Agent(holdings={'DOT': 10}), 'partial': False})

    intent_deltas, predicted_profit, Z_Ls, Z_Us = find_solution_outer_approx(initial_state, intents)

    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas, "HDX")
    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_more_random_intents_with_small():
    r = 50
    random.seed(r)
    np.random.seed(r)

    intent_ct = 500
    min_sell_ratio, max_sell_ratio = 1e-5, 0.01
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    for i in range(int(intent_ct/2)):
        sell_ratios[i] /= 1000
    min_price_ratio, max_price_ratio = 0.99, 1.01
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    initial_state.last_fee = {tkn: 0.0025 for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: 0.0005 for tkn in lrna}

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(initial_state.asset_list + ['LRNA'], 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(initial_state, sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})
        # intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
        #                 'tkn_buy': buy_tkn, 'agent': agent, 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 100000, 'tkn_sell': '2-Pool', 'tkn_buy': 'HDX', 'agent': Agent(holdings={"2-Pool": 10}), 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 4000, 'tkn_sell': 'DOT', 'tkn_buy': '2-Pool', 'agent': Agent(holdings={'DOT': 10}), 'partial': False})

    intent_deltas, predicted_profit, Z_Ls, Z_Us = find_solution_outer_approx(initial_state, intents)

    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas, "HDX")
    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_get_leftover_bounds():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'USDT': 100}),
        Agent(holdings={'USDC': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(80.0), 'tkn_sell': 'USDT', 'tkn_buy': 'USDC', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(10.0), 'tkn_sell': 'USDC', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},
    ]

    # liquidity = {'HDX': mpf(140474254.46393022), 'CRU': mpf(337868.26827475097),
    #              '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855)}
    # lrna = {'HDX': mpf(24725.8021660851), 'CRU': mpf(4744.442135139952),
    #         '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808)}

    liquidity = {'HDX': mpf(140474254.46393022), 'CRU': mpf(337868.26827475097)}
    lrna = {'HDX': mpf(24725.8021660851), 'CRU': mpf(4744.442135139952)}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    # sp_tokens = {
    #     "USDT": 7600000,
    #     "USDC": 9200000
    # }
    # stablepool = StableSwapPoolState(
    #     tokens=sp_tokens,
    #     amplification=1000,
    #     trade_fee=0.0,
    #     unique_id="2-Pool"
    # )

    amm_list = []

    init_i, exec_indices = [], []
    p = ICEProblem(initial_state, intents, amm_list=amm_list, init_i=init_i, apply_min_partial=False)
    p.set_up_problem(I=[])

    A3, b3 = _get_leftover_bounds(p, allow_loss=False)
    # b - A x >= 0
    # n = 2, sigma = 0, u = 0, m = 1, r = 0. k = 9
    x_real = np.array([
        -0.01760107,  # y_0
        0.01613315,  # y_1
        100,  # x_0
        -1.149/(1-0.0025),  # x_1
        0.01760107,  # lrna_lambda_0
        0,  # lrna_lambda_1
        0,  # lambda_0
        1.149,  # lambda_1
        100   # d_0
    ])

    x_scaled = p.get_scaled_x(x_real)

    leftovers = -A3 @ x_scaled * np.concatenate([[p._scaling['LRNA']], p._S])
    assert len(leftovers) == len(b3)
    for i in range(len(leftovers)):
        assert leftovers[i] >= b3[i]


def test_full_solver_stableswap():
    agents = [
        Agent(holdings={'HDX': 10000}),
        Agent(holdings={'HDX': 10000}),
        Agent(holdings={'USDT': 100}),
        Agent(holdings={'USDC': 100}),
        Agent(holdings={'2-Pool': 1000}),
        Agent(holdings={'2-Pool': 100}),
        Agent(holdings={'2-Pool': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(90.0), 'tkn_sell': 'USDT', 'tkn_buy': 'USDC', 'agent': agents[2], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(10.0), 'tkn_sell': 'USDC', 'tkn_buy': 'DOT', 'agent': agents[3], 'partial': False},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(2), 'tkn_sell': 'HDX', 'tkn_buy': 'DOT',
        #  'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(2), 'tkn_sell': 'HDX', 'tkn_buy': 'DOT',
        #  'agent': agents[1], 'partial': False},
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(500.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
         'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(1000), 'buy_quantity': mpf(500.0), 'tkn_sell': '2-Pool', 'tkn_buy': '4-Pool',
        #  'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(90.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
        #  'agent': agents[5], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
        #  'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    # liquidity = {'4-Pool': mpf(1392263.9295618401 + 15000), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
    #              'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
    #              'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
    #              'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
    #              '2-Pool': mpf(14626788.977583803 - 15000), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
    #              'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    # lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
    #         'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
    #         'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
    #         'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
    #         '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
    #         'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {'4-Pool': mpf(1392263.9295618401 + 15000), 'HDX': mpf(140474254.46393022), '2-Pool-btc': mpf(80.37640742108785),
                 'CRU': mpf(337868.26827475097), '2-Pool': mpf(14626788.977583803 - 15000)}
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), '2-Pool-btc': mpf(197326.54331209575),
            'CRU': mpf(4744.442135139952), '2-Pool': mpf(523282.70722423657)}

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), '2-Pool-btc': mpf(80.37640742108785),
                 'CRU': mpf(337868.26827475097), '2-Pool': mpf(14626788.977583803)}
    lrna = {'4-Pool': mpf(50483.454258911326/10), 'HDX': mpf(24725.8021660851), '2-Pool-btc': mpf(197326.54331209575),
            'CRU': mpf(4744.442135139952), '2-Pool': mpf(523282.70722423657)}

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), '2-Pool': mpf(14626788.977583803)}
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), '2-Pool': mpf(523282.70722423657)}


    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    sp_tokens = {
        "USDT": 7600000 - 81080,
        "USDC": 9200000 + 74500
    }
    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000 - 74600,
        "USDT": 340000 + 81160,
        # "DAI": 365000,
        # "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]
    amm_list = [stablepool, stablepool4]
    # amm_list = [stablepool]

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    intent_deltas, omnipool_deltas, amm_deltas = x[0], x[4], x[5]

    # valid, profit =  validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents),
                                                  intent_deltas, omnipool_deltas, amm_deltas)
    assert valid

    pprint(intent_deltas)


def test_more_random_intents_with_stableswap():
    r = 52
    random.seed(r)
    np.random.seed(r)

    intent_ct = 5
    min_sell_ratio, max_sell_ratio = 1e-10, 0.001
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    min_price_ratio, max_price_ratio = 0.99, 1.01
    min_price_ratio, max_price_ratio = 0.5, 0.9
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)
    partial_flags = np.array([False] * intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    initial_state.last_fee = {tkn: 0.0025 for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: 0.0005 for tkn in lrna}

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]

    total_asset_list = [tkn for tkn in initial_state.asset_list] + ['LRNA']
    for amm in amm_list:
        if amm.unique_id not in total_asset_list:
            total_asset_list.append(amm.unique_id)
        for tkn in amm.asset_list:
            if tkn not in total_asset_list:
                total_asset_list.append(tkn)

    lrna_prices = {}
    for tkn in total_asset_list:
        if tkn == 'LRNA':
            lrna_prices[tkn] = 1
        elif tkn in initial_state.asset_list:
            lrna_prices[tkn] = initial_state.price(initial_state, tkn, 'LRNA')
        else:  # tkn is in a stableswap pool which has shares in Omnipool
            prices = []  # we will take average of several prices if token is in multiple stableswap pools
            for amm in amm_list:
                if tkn in amm.asset_list:
                    spot = amm.withdraw_asset_spot(tkn)  # spot is tkn / shares
                    prices.append(spot * initial_state.price(initial_state, amm.unique_id, 'LRNA'))
            lrna_prices[tkn] = np.mean(prices)

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(total_asset_list, 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    total_liquidity = {tkn: initial_state.liquidity[tkn] for tkn in initial_state.asset_list}
    for amm in amm_list:
        for tkn in amm.asset_list:
            if tkn not in total_liquidity:
                total_liquidity[tkn] = 0
            total_liquidity[tkn] += amm.liquidity[tkn]

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn == "LRNA":
            if buy_tkn in initial_state.asset_list:
                sell_quantity = sell_ratios[i] * lrna[buy_tkn]
            else:
                sell_quantity = sell_ratios[i] * initial_state.lrna_total / 20
        else:
            sell_quantity = sell_ratios[i] * total_liquidity[sell_tkn]
        buy_quantity = sell_quantity * lrna_prices[sell_tkn] / lrna_prices[buy_tkn] * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    z_l_archive = x[2]
    z_u_archive = x[3]
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    # valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas)

    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_temp_milp():

    agent1 = Agent(holdings={'vASTR': 1400})
    agent2 = Agent(holdings={'DOT': 1465})
    agent3 = Agent(holdings={'iBTC': .00072869})
    agent4 = Agent(holdings={'ZTG': 2046})
    agent5 = Agent(holdings={'HDX': 0.0078998})
    agent6 = Agent(holdings={'HDX': 10000})
    intents = [
        {'sell_quantity': 1400, 'buy_quantity': 15000, 'tkn_sell': 'vASTR', 'tkn_buy': 'HDX', 'agent': agent1, 'partial': False},
        {'sell_quantity': .00072869, 'buy_quantity': 2520, 'tkn_sell': 'iBTC', 'tkn_buy': 'INTR', 'agent': agent3, 'partial': False},
        {'sell_quantity': 2046, 'buy_quantity': 55, 'tkn_sell': 'ZTG', 'tkn_buy': 'CRU', 'agent': agent4, 'partial': False},
        # {'sell_quantity': 10000, 'buy_quantity': 100, 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agent6, 'partial': True},
        {'sell_quantity': 1465, 'buy_quantity': 1139472, 'tkn_sell': 'DOT', 'tkn_buy': 'HDX', 'agent': agent2, 'partial': False},
        {'sell_quantity': 0.0078998, 'buy_quantity': 2286, 'tkn_sell': '2-pool-btc', 'tkn_buy': 'GLMR', 'agent': agent5, 'partial': False}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.00,
        lrna_fee=0.00
    )
    initial_state.last_fee = {tkn: 0.00 for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: 0.00 for tkn in lrna}

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    z_l_archive = x[2]
    z_u_archive = x[3]
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    # valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas)

    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)