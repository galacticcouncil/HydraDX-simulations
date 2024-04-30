import math
import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from mpmath import mp, mpf

from hydradx.model.amm.concentrated_liquidity_pool import ConcentratedLiquidityState, price_to_tick, tick_to_price
from hydradx.model.amm.agents import Agent

mp.dps = 50

token_amounts = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
price_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
fee_strategy = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)

@given(price_strategy, fee_strategy)
def test_price_boundaries(initial_price, trade_fee):
    tick_spacing = 10
    price_spread = 10
    price = tick_to_price(price_to_tick(initial_price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityState(
        assets={'A': 1000 / price, 'B': 1000},
        min_tick=price_to_tick(price) - price_spread,
        tick_spacing=tick_spacing,
        fee=trade_fee
    )
    second_state = ConcentratedLiquidityState(
        assets=initial_state.liquidity.copy(),
        max_tick=initial_state.max_tick,
        fee=trade_fee
    )
    if initial_state.min_tick != second_state.min_tick:
        raise AssertionError('min_tick is not the same')
    buy_x_agent = Agent(holdings={'B': 10000})
    buy_x_state = initial_state.copy().swap(
        buy_x_agent, tkn_buy='A', tkn_sell='B', buy_quantity=initial_state.liquidity['A']
    )
    new_price = buy_x_state.price('A')
    # if new_price != pytest.approx(initial_state.max_price, rel=1e-08):
    #     raise AssertionError(f"Buying all initial liquidity[A] should raise price to max.")
    buy_y_agent = Agent(holdings={'A': 100000})
    buy_y_state = initial_state.copy().swap(
        buy_y_agent, tkn_buy='B', tkn_sell='A', buy_quantity=initial_state.liquidity['B']
    )
    new_price = buy_y_state.price('A')
    if new_price != pytest.approx(initial_state.min_price, rel=1e-08):
        raise AssertionError(f"Buying all initial liquidity[B] should lower price to min.")
