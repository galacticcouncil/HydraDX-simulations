import copy
from pprint import pprint

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from mpmath import mp, mpf

from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution, find_solution2


def test_single_trade_settles():
    agents = [Agent(holdings={'DOT': 100})]

    init_intents = [  # selling DOT for $7
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]}
    ]
    intents = copy.deepcopy(init_intents)

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

    intent_deltas = find_solution2(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)
    assert intent_deltas[0][0] == -init_intents[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents[0]['buy_quantity']


def test_single_trade_does_not_settle():
    agents = [Agent(holdings={'DOT': 100})]

    init_intents = [  # selling DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(800), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]}
    ]
    intents = copy.deepcopy(init_intents)

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

    intent_deltas = find_solution2(initial_state, intents)

    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0


def test_matching_trades_execute_more():
    agents = [Agent(holdings={'DOT': 1000}), Agent(holdings={'USDT': 7600})]

    intent1 = {  # selling DOT for $7.49
        'sell_quantity': mpf(1000), 'buy_quantity': mpf(7470), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]
    }

    intent2 = {  # buying DOT for $7.51
        'sell_quantity': mpf(7530), 'buy_quantity': mpf(1000), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1]
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
    sale_deltas = find_solution2(state_sale, intents_sale)
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    buy_deltas = find_solution2(state_buy, intents_buy)
    assert validate_and_execute_solution(state_buy, intents_buy, buy_deltas)

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    match_deltas = find_solution2(state_match, intents_match)
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])


def test_convex():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1]},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2]},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3]},  # selling HDX for $1
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

    intent_deltas = find_solution(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)


def test_convex2():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1]},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2]},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3]},  # selling HDX for $1
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

    intent_deltas = find_solution2(initial_state, intents)

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
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0]},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1]},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2]},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3]},  # selling HDX for $1
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(100), 'tkn_sell': 'LRNA', 'tkn_buy': 'DOT', 'agent': agents[4]},  # buying DOT for $10
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

    intent_deltas = find_solution2(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)
