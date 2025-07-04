import pytest

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.solver.omnix import validate_solution, calculate_transfers


def test_calculate_transfers():

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

    amt_processed = [10000, 7788]
    buy_prices = {'DOT': 7.522185628, 'USDT': 0.996679094}
    sell_prices = {'DOT': 7.524947064, 'USDT': 0.99703}

    transfers, deltas = calculate_transfers(intents, amt_processed, buy_prices, sell_prices)

    expected_transfers = [
        {'agent': agent_alice, 'buy_quantity': amt_processed[0], 'sell_quantity': 75221.85628 / 0.99703, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': amt_processed[1], 'buy_quantity': 7.524947064 * 7788 / 0.996679094, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ]

    expected_deltas = {
        'DOT': {"in": expected_transfers[1]['sell_quantity'], "out": expected_transfers[0]['buy_quantity']},
        'USDT': {"in": expected_transfers[0]['sell_quantity'], "out": expected_transfers[1]['buy_quantity']}
    }

    if len(expected_transfers) != len(transfers):
        raise Exception(f"Expected transfers doesn't have the same length as transfers")
    for i in range(len(expected_transfers)):
        if transfers[i]["agent"] != expected_transfers[i]["agent"]:
            raise Exception(f"Expected transfers doesn't match transfers")
        if transfers[i]["tkn_buy"] != expected_transfers[i]["tkn_buy"]:
            raise Exception(f"Expected transfers doesn't match transfers")
        if transfers[i]["tkn_sell"] != expected_transfers[i]["tkn_sell"]:
            raise Exception(f"Expected transfers doesn't match transfers")
        if transfers[i]["buy_quantity"] != pytest.approx(expected_transfers[i]["buy_quantity"], rel=1e-12):
            raise Exception(f"Expected transfers doesn't match transfers")
        if transfers[i]["sell_quantity"] != pytest.approx(expected_transfers[i]["sell_quantity"], rel=1e-12):
            raise Exception(f"Expected transfers doesn't match transfers")
    if len(deltas) != len(expected_deltas):
        raise Exception(f"Expected net_deltas doesn't have the same length as net_deltas")
    for tkn in expected_deltas:
        if deltas[tkn]["in"] != pytest.approx(expected_deltas[tkn]["in"], rel=1e-12):
            raise Exception(f"Expected net_deltas doesn't match net_deltas")
        if deltas[tkn]["out"] != pytest.approx(expected_deltas[tkn]["out"], rel=1e-12):
            raise Exception(f"Expected net_deltas doesn't match net_deltas")


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
    agent_bob = Agent(holdings={'DOT': 0})

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 75500, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ]

    amt_processed = [10000, 0]
    buy_prices = {'DOT': 7.5566, 'USDT': 1}
    sell_prices = {'DOT': 7.5567, 'USDT': 0.9924}

    validate_solution(omnipool, intents, amt_processed, buy_prices, sell_prices, 0.0001)


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
    buy_prices = {'DOT': 7.55666931, 'USDT': 0.9851}
    sell_prices = {'DOT': 7.6126, 'USDT': 0.99243}
    tolerance = 0.0001
    validate_solution(omnipool, intents, amt_processed, buy_prices, sell_prices, tolerance)


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
    buy_prices = {'DOT': 7.522185628, 'USDT': 0.996679094}
    sell_prices = {'DOT': 7.524947064, 'USDT': 0.99703}
    tolerance = 0.0001
    validate_solution(omnipool, intents, amt_processed, buy_prices, sell_prices, tolerance)


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
    agent_bob = Agent(holdings={'DOT': 0})
    agent_charlie = Agent(holdings={'HDX': 0})

    intents = [
        {'agent': agent_alice, 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': 10000, 'buy_limit': 7550000, 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': 5000000, 'buy_limit': 45000, 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'}
    ]

    amt_processed = [10000, 0, 0]
    buy_prices = {'DOT': 7.5566, 'USDT': 1, 'HDX': 0.009}
    sell_prices = {'DOT': 7.5567, 'USDT': 0.9924, 'HDX': 111.111111111}

    validate_solution(omnipool, intents, amt_processed, buy_prices, sell_prices, 0.0001)

