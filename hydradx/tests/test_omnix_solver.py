import pytest
from hypothesis import given, strategies as st, settings
from mpmath import mp, mpf

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnix_solver import calculate_price_slippage_to_impact, get_sorted_intents, \
    calculate_transfers, calculate_solution_first_trade, calculate_solution_iteratively, calculate_net_intents
from hydradx.model.amm.omnix_solver import get_new_prices_for_tkn
from hydradx.model.amm.omnix import validate_solution


def test_calculate_price_slippage_to_impact():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 10000000 / 7.5, 'LRNA': 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    intent = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(9.87654321):
        raise

    intent = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 74000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(-1.801801802):
        raise

    intent = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 74000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(1.777777777777):
        raise

    intent = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 81000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(-10.6666666666666):
        raise

    # doubling liquidity of DOT should not change any results
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 2 * 10000000 / 7.5, 'LRNA': 2 * 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    intent = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(9.87654321):
        raise

    intent = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 74000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(-1.801801802):
        raise

    intent = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 74000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(1.777777777777):
        raise

    intent = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 81000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    ratio = calculate_price_slippage_to_impact(omnipool, intent)
    if ratio != pytest.approx(-10.6666666666666):
        raise


def test_get_sorted_intents():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': 2 * 10000000 / 7.5, 'LRNA': 2 * 10000000},
            "USDT": {'liquidity': 10000000, 'LRNA': 10000000},
            "HDX": {'liquidity': 100000000, 'LRNA': 1000000}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    intent1 = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 81000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    intent2 = {'agent': Agent(), 'buy_quantity': 10000, 'sell_limit': 74000, 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    intent3 = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 74000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}
    intent4 = {'agent': Agent(), 'sell_quantity': 10000, 'buy_limit': 81000, 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'}

    intents = [intent1, intent2, intent3, intent4]
    sorted_intents = get_sorted_intents(omnipool, intents)
    if sorted_intents != [intent1, intent3, intent4, intent2]:
        raise


def test_calculate_net_intents():
    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(10000)})
    agent_charlie = Agent(holdings={'HDX': mpf(10000000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788 * 90), 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(70000*100), 'buy_limit': mpf(65000), 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'},
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(80000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(80000*100), 'tkn_buy': 'DOT', 'tkn_sell': 'HDX'},
        {'agent': agent_alice, 'buy_quantity': mpf(80000 * 100), 'sell_limit': mpf(80000), 'tkn_buy': 'HDX', 'tkn_sell': 'USDT'},

    ]

    tkn_buy = 'DOT'
    tkn_sell = 'USDT'

    buy_prices = {'DOT': mpf(7.522185628), 'USDT': mpf(0.996679094), "HDX": mpf(0.01)}
    sell_prices = {'DOT': mpf(7.524947064), 'USDT': mpf(0.99703), "HDX": mpf(0.01)}

    net_intents, price_limits = calculate_net_intents(intents, tkn_buy, tkn_sell, buy_prices, sell_prices)
    print('done')


def test_calculate_transfers_reduced():
    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(10000)})
    agent_charlie = Agent(holdings={'HDX': mpf(10000000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788 * 90), 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(70000*100), 'buy_limit': mpf(65000), 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'}
    ]

    tkn_buy = 'DOT'
    tkn_sell = 'USDT'

    buy_prices = {'DOT': mpf(7.522185628), 'USDT': mpf(0.996679094), "HDX": mpf(0.01)}
    sell_prices = {'DOT': mpf(7.524947064), 'USDT': mpf(0.99703), "HDX": mpf(0.01)}

    net_intents, price_limits = calculate_net_intents(intents, tkn_buy, tkn_sell, buy_prices, sell_prices)

    transfers, deltas = calculate_transfers(net_intents, buy_prices, sell_prices)


def test_add_trade_to_solution():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(10000)})
    agent_charlie = Agent(holdings={'HDX': mpf(10000000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788 * 90), 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(70000*100), 'buy_limit': mpf(65000), 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'},
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(80000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(80000*100), 'tkn_buy': 'DOT', 'tkn_sell': 'HDX'},
        {'agent': agent_alice, 'buy_quantity': mpf(80000 * 100), 'sell_limit': mpf(80000), 'tkn_buy': 'HDX', 'tkn_sell': 'USDT'},

    ]

    tkn_buy = 'DOT'
    tkn_sell = 'USDT'

    buy_prices = {'DOT': mpf(7.522185628), 'USDT': mpf(0.996679094), "HDX": mpf(0.01)}
    sell_prices = {'DOT': mpf(7.524947064), 'USDT': mpf(0.99703), "HDX": mpf(0.01)}

    net_intents, price_limits = calculate_net_intents(intents, tkn_buy, tkn_sell, buy_prices, sell_prices)
    transfers, deltas = calculate_transfers(net_intents, buy_prices, sell_prices)
    # validate_prices(
    #     omnipool,
    #
    # )
    print('done')


def test_calculate_solution_first_trade():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )
    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    new_intent = {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'}
    intents, amts, buy_prices, sell_prices, omnipool_new = calculate_solution_first_trade(omnipool, new_intent)
    print('done')


def test_calculate_solution_interactively():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'USDT': mpf(100000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        # {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788), 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'},
        {'agent': agent_bob, 'buy_quantity': mpf(7788), 'sell_limit': mpf(8.1 * 7788), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
    ]

    validation_tolerance = 0.0001
    solution_tolerance = validation_tolerance / 2

    included_intents, amts, buy_prices, sell_prices, omnipool_new = calculate_solution_iteratively(omnipool, intents, validation_tolerance, solution_tolerance)
    valid = validate_solution(omnipool, included_intents, amts, buy_prices, sell_prices, validation_tolerance)
    if not valid:
        raise Exception("solution fails validation")


def test_calculate_solution_interactively2():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(100000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        # {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788), 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.4 * 7788), 'tkn_buy': 'USDT', 'tkn_sell': 'DOT'},
    ]

    validation_tolerance = 0.0001
    solution_tolerance = validation_tolerance / 2

    included_intents, amts, buy_prices, sell_prices, omnipool_new = calculate_solution_iteratively(omnipool, intents, validation_tolerance, solution_tolerance)
    valid = validate_solution(omnipool, included_intents, amts, buy_prices, sell_prices, validation_tolerance)
    if not valid:
        raise Exception("solution fails validation")


def test_calculate_solution_interactively3():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(100000)})
    agent_charlie = Agent(holdings={'HDX': mpf(10000000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788 * 90), 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(70000*100), 'buy_limit': mpf(65000), 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'},
    ]

    validation_tolerance = 0.0001
    solution_tolerance = validation_tolerance / 2

    included_intents, amts, buy_prices, sell_prices, omnipool_new = calculate_solution_iteratively(omnipool, intents, validation_tolerance, solution_tolerance)
    valid = validate_solution(omnipool, included_intents, amts, buy_prices, sell_prices, validation_tolerance)
    if not valid:
        raise Exception("solution fails validation")


def test_calculate_solution_interactively4():
    omnipool = OmnipoolState(
        tokens={
            "DOT": {'liquidity': mpf(10000000/7.5), 'LRNA': mpf(10000000)},
            "USDT": {'liquidity': mpf(10000000), 'LRNA': mpf(10000000)},
            "HDX": {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0,
        lrna_fee=0.0
    )

    agent_alice = Agent(holdings={'USDT': mpf(100000)})
    agent_bob = Agent(holdings={'DOT': mpf(100000)})
    agent_charlie = Agent(holdings={'HDX': mpf(10000000), 'DOT': mpf(1000)})

    intents = [
        {'agent': agent_alice, 'buy_quantity': mpf(10000), 'sell_limit': mpf(81000), 'tkn_buy': 'DOT', 'tkn_sell': 'USDT'},
        {'agent': agent_bob, 'sell_quantity': mpf(7788), 'buy_limit': mpf(7.40 * 7788 * 90), 'tkn_buy': 'HDX', 'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(1000), 'buy_limit': mpf(8100), 'tkn_buy': 'USDT',
         'tkn_sell': 'DOT'},
        {'agent': agent_charlie, 'sell_quantity': mpf(70000*100), 'buy_limit': mpf(65000), 'tkn_buy': 'USDT', 'tkn_sell': 'HDX'},
    ]

    validation_tolerance = 0.0001
    solution_tolerance = validation_tolerance / 2

    included_intents, amts, buy_prices, sell_prices, omnipool_new = calculate_solution_iteratively(omnipool, intents, validation_tolerance, solution_tolerance)
    valid = validate_solution(omnipool, included_intents, amts, buy_prices, sell_prices, validation_tolerance)
    if not valid:
        raise Exception("solution fails validation")


def test_get_new_prices_for_tkn():
    remaining_lrna = mpf(10)
    tkn_out = 1000
    tkn_in = 800
    lrna_out = 0
    lrna_in = 0
    tkn_to_omnipool = -200
    lrna_to_omnipool = -tkn_to_omnipool*7.6
    sell_price = 7.7
    buy_price = 7.65

    new_buy_price, new_sell_price = get_new_prices_for_tkn(remaining_lrna, tkn_out, tkn_in, lrna_out, lrna_in,
                                                           buy_price, sell_price, lrna_to_omnipool, tkn_to_omnipool)
    assert new_buy_price > 0
    assert new_sell_price > 0
