from hypothesis import given, strategies as st, settings
from mpmath import mp, mpf

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnix import validate_solution

def test_validate_solution_simple():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 10000000/7.5, 'LRNA': 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': 100000})
    agent_bob = Agent()

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 75500, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ]

    amt_processed = [10000, 0]
    lrna_swaps = {'DOT': 75567, 'USDT': -75567}
    buy_prices = {'DOT': 7.5566, 'USDT': 1}
    sell_prices = {'DOT': 7.5567, 'USDT': 0.9924}

    validate_solution(omnipool, intents, amt_processed, lrna_swaps, buy_prices, sell_prices, 0.0001)


def test_validate_solution_partial_matching():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 10000000/7.5, 'LRNA': 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': 100000})
    agent_bob = Agent(holdings={'DOT': 10000})

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 75500, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ]

    amt_processed = [10000, 100]
    lrna_swaps = {'DOT': 74805.8, 'USDT': -74805.8}
    buy_prices = {'DOT': 7.55666931, 'USDT': 0.9851}
    sell_prices = {'DOT': 7.6126, 'USDT': 0.99243}
    tolerance = 0.0001
    validate_solution(omnipool, intents, amt_processed, lrna_swaps, buy_prices, sell_prices, tolerance)


def test_validate_solution_with_matching():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 10000000/7.5, 'LRNA': 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': 100000})
    agent_bob = Agent(holdings={'DOT': 10000})

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 75500, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ]

    amt_processed = [10000, 7788]
    lrna_swaps = {'DOT': 16618, 'USDT': -16618}
    buy_prices = {'DOT': 7.522185628, 'USDT': 0.996679094}
    sell_prices = {'DOT': 7.524947064, 'USDT': 0.99703}
    tolerance = 0.0001
    validate_solution(omnipool, intents, amt_processed, lrna_swaps, buy_prices, sell_prices, tolerance)


def test_validate_solution_simple_three_intents():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 10000000/7.5, 'LRNA': 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': 100000})
    agent_bob = Agent()
    agent_charlie = Agent()

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 7550000, 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': 5000000, 'buy_limit': 45000, 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'}
    ]

    amt_processed = [10000, 0, 0]
    lrna_swaps = {'DOT': 75567, 'USDT': -75567}
    buy_prices = {'DOT': 7.5566, 'USDT': 1}
    sell_prices = {'DOT': 7.5567, 'USDT': 0.9924}

    validate_solution(omnipool, intents, amt_processed, lrna_swaps, buy_prices, sell_prices, 0.0001)

