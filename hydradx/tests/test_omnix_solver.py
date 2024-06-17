import pytest
from hypothesis import given, strategies as st, settings
from mpmath import mp, mpf

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnix_solver import calculate_price_slippage_to_impact, get_sorted_intents


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
