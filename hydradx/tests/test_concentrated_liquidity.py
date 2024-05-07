import math
import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from mpmath import mp, mpf

from hydradx.model.amm.concentrated_liquidity_pool import ConcentratedLiquidityState, price_to_tick, tick_to_price
from hydradx.model.amm.agents import Agent

mp.dps = 50

token_amounts = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
price_strategy = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
fee_strategy = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)

@given(price_strategy, st.integers(min_value=1, max_value=100))
def test_price_boundaries_no_fee(initial_price, price_spread):
    tick_spacing = 1
    price_spread *= tick_spacing
    price = tick_to_price(price_to_tick(initial_price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityState(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - price_spread,
        tick_spacing=tick_spacing,
        fee=0
    )
    second_state = ConcentratedLiquidityState(
        assets=initial_state.liquidity.copy(),
        max_tick=initial_state.max_tick,
        tick_spacing=tick_spacing,
        fee=0
    )
    if initial_state.min_tick != second_state.min_tick:
        raise AssertionError('min_tick is not the same')
    buy_x_agent = Agent(holdings={'B': 1000000})
    buy_x_state = initial_state.copy().swap(
        buy_x_agent, tkn_buy='A', tkn_sell='B', buy_quantity=initial_state.liquidity['A']
    )
    new_price = buy_x_state.price('A')
    if new_price != pytest.approx(initial_state.max_price, rel=1e-12):
        raise AssertionError(f"Buying all initial liquidity[A] should raise price to max.")
    buy_y_agent = Agent(holdings={'A': 1000000})
    buy_y_state = initial_state.copy().swap(
        buy_y_agent, tkn_buy='B', tkn_sell='A', buy_quantity=initial_state.liquidity['B']
    )
    new_price = buy_y_state.price('A')
    if new_price != pytest.approx(initial_state.min_price, rel=1e-12):
        raise AssertionError(f"Buying all initial liquidity[B] should lower price to min.")


@given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_asset_conservation(price, trade_fee, price_spread):
    tick_spacing = 1
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityState(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_spread,
        tick_spacing=tick_spacing,
        fee=trade_fee
    )
    sell_quantity = 1000
    sell_agent = Agent(holdings={'B': 1000000})
    sell_state = initial_state.copy().swap(
        sell_agent, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )
    if sell_state.liquidity['A'] + sell_agent.holdings['A'] != initial_state.liquidity['A']:
        raise AssertionError('Asset A was not conserved.')
    if sell_state.liquidity['B'] + sell_agent.holdings['B'] != initial_state.liquidity['B'] + sell_agent.initial_holdings['B']:
        raise AssertionError('Asset B was not conserved.')

    buy_quantity = initial_state.calculate_buy_from_sell(tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity)
    buy_agent = Agent(holdings={'B': 1000000})
    buy_state = initial_state.copy().swap(
        buy_agent, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )
    if buy_state.liquidity['A'] + buy_agent.holdings['A'] != initial_state.liquidity['A']:
        raise AssertionError('Asset A was not conserved.')
    if buy_state.liquidity['B'] + buy_agent.holdings['B'] != initial_state.liquidity['B'] + buy_agent.initial_holdings['B']:
        raise AssertionError('Asset B was not conserved.')

@given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_buy_sell_equivalency(price, trade_fee, price_spread):
    tick_spacing = 10
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityState(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_spread,
        tick_spacing=tick_spacing,
        fee=0.0111
    )
    sell_quantity = 1000
    sell_y_agent = Agent(holdings={'B': 1000000})
    initial_state.copy().swap(
        sell_y_agent, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )
    buy_quantity = initial_state.calculate_buy_from_sell(tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity)
    buy_x_agent = Agent(holdings={'B': 1000000})
    initial_state.copy().swap(
        buy_x_agent, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )
    if buy_x_agent.holdings['A'] != buy_quantity != sell_y_agent.holdings['A']:
        raise AssertionError('Buy quantity was not bought correctly.')
    if sell_y_agent.holdings['A'] != pytest.approx(buy_x_agent.holdings['A'], rel=1e-12):
        raise AssertionError('Sell quantity was not calculated correctly.')

# @given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_price_boundaries_with_fee():  # initial_price, trade_fee, price_spread):
    tick_spacing = 1
    trade_fee = 0.0111
    price_spread = 100
    initial_price = 5
    price = tick_to_price(price_to_tick(initial_price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityState(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - price_spread,
        tick_spacing=tick_spacing,
        fee=trade_fee
    )
    second_state = ConcentratedLiquidityState(
        assets=initial_state.liquidity.copy(),
        max_tick=initial_state.max_tick,
        tick_spacing=tick_spacing,
        fee=trade_fee
    )
    if initial_state.min_tick != second_state.min_tick:
        raise AssertionError('min_tick is not the same')
    buy_x_agent = Agent(holdings={'B': 100000})
    buy_x_state = initial_state.copy().swap(
        buy_x_agent, tkn_buy='A', tkn_sell='B', buy_quantity=initial_state.liquidity['A']
    )
    new_price = buy_x_state.price('A')
    if new_price != pytest.approx(initial_state.max_price, rel=1e-20):
        print()
        if new_price < initial_state.max_price:
            print(f"new price is {100 * (1 - new_price / initial_state.max_price)} % lower than max")
        else:
            print(f"new price is {100 * (new_price / initial_state.max_price - 1)} % higher than max")
        # raise AssertionError(f"Buying all initial liquidity[A] should raise price to max.")
    buy_y_agent = Agent(holdings={'A': 1000000})
    buy_y_state = initial_state.copy().swap(
        buy_y_agent, tkn_buy='B', tkn_sell='A', buy_quantity=initial_state.liquidity['B']
    )
    new_price = buy_y_state.price('A')
    if new_price != pytest.approx(initial_state.min_price, rel=1e-20):
        print()
        if new_price < initial_state.min_price:
            print(f"new price is {100 * (1 - new_price / initial_state.min_price)} % lower than min")
        else:
            print(f"new price is {100 * (new_price / initial_state.min_price - 1)} % higher than min")
        # raise AssertionError(f"Buying all initial liquidity[B] should lower price to min.")
