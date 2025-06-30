import copy
import math
from datetime import timedelta

import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
import mpmath
from mpmath import mp, mpf
import os
os.chdir('../..')

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.xyk_amm import XykState

settings.register_profile("long", deadline=timedelta(milliseconds=500), print_blob=True)
settings.load_profile("long")

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
    from hydradx.model.amm.xyk_amm import simulate_swap
    # Test sells
    tokens = {'A': liq_A, 'B': liq_B}
    swap_quantity = tokens['A'] * swap_pct
    agent = Agent(holdings={'A': swap_quantity})
    for f in [0.0, fee]:
        xyk = XykState(tokens=tokens, trade_fee=f)
        k_init = xyk.calculate_k()
        for buy_tkn, sell_tkn in [('B', 'A'), ('A', 'B')]:
            new_state, new_agent = simulate_swap(xyk, agent, sell_tkn, buy_tkn, sell_quantity=swap_quantity)
            if f == 0.0:
                k_final = new_state.calculate_k()
            else:
                fee_taken = (xyk.liquidity[buy_tkn] - new_state.liquidity[buy_tkn]) * f/(1 - f)
                k_final = math.sqrt(new_state.liquidity[sell_tkn] * (new_state.liquidity[buy_tkn] - fee_taken))
            if k_init != pytest.approx(k_final, rel=1e-12):
                raise AssertionError("K value changed after swap")
            init_A_total = xyk.liquidity['A'] + agent.get_holdings('A')
            final_A_total = new_state.liquidity['A'] + new_agent.get_holdings('A')
            init_B_total = xyk.liquidity['B'] + agent.get_holdings('B')
            final_B_total = new_state.liquidity['B'] + new_agent.get_holdings('B')
            if init_A_total != pytest.approx(final_A_total, rel=1e-12):
                raise AssertionError("A total quantity changed after swap")
            if init_B_total != pytest.approx(final_B_total, rel=1e-12):
                raise AssertionError("B total quantity changed after swap")

    # Test buys, feeless
    swap_A_quantity = tokens['A'] * swap_pct
    swap_B_quantity = tokens['B'] * swap_pct
    agent = Agent(enforce_holdings=False)
    for f in [0.0, fee]:
        xyk = XykState(tokens=tokens, trade_fee=f)
        k_init = xyk.calculate_k()
        for buy_tkn, sell_tkn in [('B', 'A'), ('A', 'B')]:
            buy_quantity = swap_A_quantity if buy_tkn == 'A' else swap_B_quantity
            new_state, new_agent = simulate_swap(xyk, agent, sell_tkn, buy_tkn, buy_quantity=buy_quantity)
            if f == 0.0:
                k_final = new_state.calculate_k()
            else:
                fee_taken = buy_quantity * f/(1 - f)
                k_final = math.sqrt(new_state.liquidity[sell_tkn] * (new_state.liquidity[buy_tkn] - fee_taken))
            if k_init != pytest.approx(k_final, rel=1e-12):
                raise AssertionError("K value changed after swap")
            init_A_total = xyk.liquidity['A'] + agent.get_holdings('A')
            final_A_total = new_state.liquidity['A'] + new_agent.get_holdings('A')
            init_B_total = xyk.liquidity['B'] + agent.get_holdings('B')
            final_B_total = new_state.liquidity['B'] + new_agent.get_holdings('B')
            if init_A_total != pytest.approx(final_A_total, rel=1e-12):
                raise AssertionError("A total quantity changed after swap")
            if init_B_total != pytest.approx(final_B_total, rel=1e-12):
                raise AssertionError("B total quantity changed after swap")
