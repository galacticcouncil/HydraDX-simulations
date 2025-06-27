import copy
import math

import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
import mpmath
from mpmath import mp, mpf
import os
os.chdir('../..')

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.xyk_amm import XykState


def test_xyk_init():
    tokens = {'A': 1000, 'B': 2000}
    xyk = XykState(tokens=tokens, trade_fee=0.01)

    assert xyk.unique_id == 'xyk'
    assert xyk.asset_list == ['A', 'B']
    assert xyk.liquidity == {'A': 1000, 'B': 2000}
    assert xyk.trade_fee == 0.01
    assert xyk.shares == math.sqrt(1000 * 2000)

    xyk = XykState(tokens=tokens, unique_id='custom_id', shares=5000)
    assert xyk.unique_id == 'custom_id'
    assert xyk.shares == 5000


def test_xyk_init_invalid_tokens():
    with pytest.raises(ValueError, match='Need exactly two tokens for XYK AMM'):
        XykState(tokens={'A': 1000, 'B': 2000, 'C': 3000})

    with pytest.raises(ValueError, match='Need exactly two tokens for XYK AMM'):
        XykState(tokens={})  # No tokens provided


@given(
    st.floats(min_value=100, max_value=100000),
    st.floats(min_value=100, max_value=100000)
)
def test_calculate_k(liq_A, liq_B):
    tokens = {'A': liq_A, 'B': liq_B}
    xyk = XykState(tokens=tokens)
    assert xyk.calculate_k() == math.sqrt(liq_A * liq_B)


@given(
    st.floats(min_value=100, max_value=100000),
    st.floats(min_value=100, max_value=100000),
    st.floats(min_value=0, max_value=0.1, exclude_min=True)
)
def test_spot(liq_A, liq_B, fee):
    tokens = {'A': liq_A, 'B': liq_B}
    xyk = XykState(tokens=tokens)

    price_A_to_B = xyk.sell_spot('A', 'B')
    assert price_A_to_B == pytest.approx(liq_B / liq_A, rel=1e-12)
    price_B_to_A = xyk.sell_spot('B', 'A')
    assert price_B_to_A == pytest.approx(liq_A / liq_B, rel=1e-12)
    price_A_to_B_with_fee = xyk.sell_spot('A', 'B', fee)
    assert price_A_to_B_with_fee == pytest.approx(liq_B / liq_A * (1 - fee), rel=1e-12)
    price_B_to_A_with_fee = xyk.sell_spot('B', 'A', fee)
    assert price_B_to_A_with_fee == pytest.approx(liq_A / liq_B * (1 - fee), rel=1e-12)
    price_A_to_B_buy = xyk.buy_spot('B', 'A')
    assert price_A_to_B_buy == pytest.approx(1 / price_A_to_B, rel=1e-12)
    price_B_to_A_buy = xyk.buy_spot('A', 'B')
    assert price_B_to_A_buy == pytest.approx(1 / price_B_to_A, rel=1e-12)
    price_A_to_B_buy_with_fee = xyk.buy_spot('B', 'A', fee)
    assert price_A_to_B_buy_with_fee == pytest.approx(1 / price_A_to_B_with_fee, rel=1e-12)
    price_B_to_A_buy_with_fee = xyk.buy_spot('A', 'B', fee)
    assert price_B_to_A_buy_with_fee == pytest.approx(1 / price_B_to_A_with_fee, rel=1e-12)


@given(
    st.floats(min_value=100, max_value=100000),
    st.floats(min_value=100, max_value=100000),
    st.floats(min_value=0, max_value=0.1, exclude_min=True),
    st.floats(min_value=0.001, max_value=0.5)
)
def test_swap(liq_A, liq_B, fee, swap_pct):
    tokens = {'A': liq_A, 'B': liq_B}
    xyk = XykState(tokens=tokens)

    k_init = xyk.calculate_k()
    swap_quantity = tokens['A'] * swap_pct
    agent = Agent(holdings={'A': swap_quantity})

    # Swap A for B

    xyk.swap(agent, 'A', 'B', sell_quantity=swap_quantity)
    k_final = xyk.calculate_k()
    assert k_init == pytest.approx(k_final, rel=1e-12)
