import copy

import pytest
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import protected_xyk_amm as pxyk
from hydradx.model.amm.agents import Agent
from hydradx.tests.strategies_omnipool import reasonable_market_dict, reasonable_holdings
from hydradx.tests.strategies_omnipool import reasonable_pct, asset_number_strategy


def test_protected_xyk_constructor_works():
    pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=3,
        trade_fee=0,
        unique_id='pool'
    )


def test_price_above_break():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=3,
        trade_fee=0,
        unique_id='pool'
    )

    p = pxyk.price(state)

    assert p == 15 / 2


def test_price_below_break():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=10,
        trade_fee=0,
        unique_id='pool'
    )

    p = pxyk.price(state)

    assert p == 1000000 / (200000 - 500000 / 10)


def test_price_at_break():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=7.5,
        trade_fee=0,
        unique_id='pool'
    )

    p = pxyk.price(state)

    assert p == 7.5


def test_get_invariant_one():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=5,
        trade_fee=0,
        unique_id='pool'
    )

    invariant = state.get_invariant_one()

    assert invariant == 1500000 * 200000


def test_get_invariant_two():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=10,
        trade_fee=0,
        unique_id='pool'
    )

    invariant = state.get_invariant_two()

    assert invariant == 1000000 * (200000 - 500000 / 10)


def test_calculate_reserve_at_intersection_from_above():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=5,
        trade_fee=0,
        unique_id='pool'
    )

    reserve_stable = pxyk.calculate_reserve_at_intersection(state, 0)
    reserve_volatile = pxyk.calculate_reserve_at_intersection(state, 1)

    after_state = copy.deepcopy(state)
    after_state.liquidity[0] = reserve_stable
    after_state.liquidity[1] = reserve_volatile

    assert state.get_invariant_one() == after_state.get_invariant_one()
    assert pxyk.price(after_state) == state.p


def test_calculate_reserve_at_intersection_from_below():
    state = pxyk.ProtectedXYKState(
        stable_asset='LRNA',
        stable_asset_quantity=1000000,
        stable_asset_virtual_quantity=500000,
        volatile_asset='TKN',
        volatile_asset_quantity=200000,
        break_price=10,
        trade_fee=0,
        unique_id='pool'
    )

    reserve_stable = pxyk.calculate_reserve_at_intersection(state, 0)
    reserve_volatile = pxyk.calculate_reserve_at_intersection(state, 1)

    after_state = copy.deepcopy(state)
    after_state.liquidity[0] = reserve_stable
    after_state.liquidity[1] = reserve_volatile

    assert state.get_invariant_two() == after_state.get_invariant_two()
    assert pxyk.price(after_state) == state.p
