import math
import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from mpmath import mp, mpf

from hydradx.model.amm.concentrated_liquidity_pool import ConcentratedLiquidityPosition, price_to_tick, tick_to_price, \
    ConcentratedLiquidityPoolState
from hydradx.model.amm.agents import Agent

mp.dps = 50

token_amounts = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
price_strategy = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
fee_strategy = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)

@given(price_strategy, st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100))
def test_price_boundaries_no_fee(initial_price, tick_spacing, price_range):
    price_range *= tick_spacing
    price = tick_to_price(price_to_tick(initial_price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityPosition(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - price_range,
        tick_spacing=tick_spacing,
        fee=0
    )
    second_state = ConcentratedLiquidityPosition(
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

    if buy_x_state.invariant != pytest.approx(buy_y_state.invariant, rel=1e12):
        raise AssertionError('Invariant is not the same')


@given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_asset_conservation(price, fee, price_range):
    tick_spacing = 1
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityPosition(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_range,
        tick_spacing=tick_spacing,
        fee=fee
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
def test_buy_sell_equivalency(price, fee, price_range):
    tick_spacing = 10
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityPosition(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_range,
        tick_spacing=tick_spacing,
        fee=fee
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


@given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_buy_spot(price, fee, price_range):
    tick_spacing = 10
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityPosition(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_range,
        tick_spacing=tick_spacing,
        fee=fee
    )
    buy_quantity = 1 / mpf(1e20)
    agent = Agent(holdings={'B': 1000})
    buy_spot = initial_state.buy_spot(tkn_buy='A', tkn_sell='B', fee=fee)
    initial_state.swap(
        agent, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )
    ex_price = (agent.initial_holdings['B'] - agent.holdings['B']) / agent.holdings['A']
    if ex_price != pytest.approx(buy_spot, rel=1e-20):
        raise AssertionError('Buy spot price was not calculated correctly.')


@given(price_strategy, fee_strategy, st.integers(min_value=1, max_value=100))
def test_sell_spot(price, fee, price_range):
    tick_spacing = 10
    price = tick_to_price(price_to_tick(price, tick_spacing=tick_spacing))
    initial_state = ConcentratedLiquidityPosition(
        assets={'A': mpf(1000 / price), 'B': mpf(1000)},
        min_tick=price_to_tick(price, tick_spacing) - tick_spacing * price_range,
        tick_spacing=tick_spacing,
        fee=fee
    )
    sell_quantity = 1 / mpf(1e20)
    agent = Agent(holdings={'A': 1000})
    sell_spot = initial_state.sell_spot(tkn_sell='A', tkn_buy='B', fee=fee)
    initial_state.swap(
        agent, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )
    ex_price = agent.holdings['B'] / (agent.initial_holdings['A'] - agent.holdings['A'])
    if ex_price != pytest.approx(sell_spot, rel=1e-20):
        raise AssertionError('Sell spot price was not calculated correctly.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_buy_x_vs_single_position(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    buy_quantity = mpf(10)

    agent1 = Agent(holdings={'B': 1000})
    one_position = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=0.0025
    ).swap(
        agent1, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )

    agent1_copy = Agent(holdings={'B': 1000})
    one_position_feeless = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent1_copy, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )

    agent2 = Agent(holdings={'B': 1000})
    whole_pool = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=math.sqrt(one_position.invariant),
        tick_spacing = tick_spacing,
        fee=0.0025
    ).swap(
        agent2, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )

    agent2_copy = Agent(holdings={'B': 1000})
    whole_pool_feeless = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=math.sqrt(one_position.invariant),
        tick_spacing = tick_spacing,
        fee=0
    ).swap(
        agent2_copy, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )

    effective_fee_one_pool = (agent1_copy.holdings['B'] - agent1.holdings['B']) / (agent1_copy.initial_holdings['B'] - agent1_copy.holdings['B'])
    effective_fee_whole_pool = (agent2_copy.holdings['B'] - agent2.holdings['B']) / (agent2_copy.initial_holdings['B'] - agent2_copy.holdings['B'])

    if agent1.holdings['A'] != agent2.holdings['A']:
        raise AssertionError('Buy quantity was not applied correctly.')
    if agent1.holdings['B'] != pytest.approx(agent2.holdings['B'], rel=1e-8):
        raise AssertionError('Sell quantity was not calculated correctly.')
    if effective_fee_whole_pool != pytest.approx(effective_fee_one_pool, rel=1e-8):
        raise AssertionError('Fee levels do not match.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_buy_y_vs_single_position(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    buy_quantity = mpf(10)

    agent1 = Agent(holdings={'A': 1000})
    one_position = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=fee
    ).swap(
        agent1, tkn_buy='B', tkn_sell='A', buy_quantity=buy_quantity
    )

    agent1_copy = Agent(holdings={'A': 1000})
    one_position_feeless = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent1_copy, tkn_buy='B', tkn_sell='A', buy_quantity=buy_quantity
    )

    agent2 = Agent(holdings={'A': 1000})
    whole_pool = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing = tick_spacing,
        fee=fee
    ).swap(
        agent2, tkn_buy='B', tkn_sell='A', buy_quantity=buy_quantity
    )

    agent2_copy = Agent(holdings={'A': 1000})
    whole_pool_feeless = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing = tick_spacing,
        fee=0
    ).swap(
        agent2_copy, tkn_buy='B', tkn_sell='A', buy_quantity=buy_quantity
    )

    effective_fee_one_pool = (agent1_copy.holdings['A'] - agent1.holdings['A']) / (agent1_copy.initial_holdings['A'] - agent1_copy.holdings['A'])
    effective_fee_whole_pool = (agent2_copy.holdings['A'] - agent2.holdings['A']) / (agent2_copy.initial_holdings['A'] - agent2_copy.holdings['A'])
    if agent1.holdings['A'] != pytest.approx(agent2.holdings['A'], rel=1e-8):
        raise AssertionError('Sell quantity was not calculated correctly.')
    if agent1.holdings['B'] != agent2.holdings['B']:
        raise AssertionError('Buy quantity was not applied correctly.')
    if effective_fee_whole_pool != pytest.approx(effective_fee_one_pool, rel=1e-8):
        raise AssertionError('Fee levels do not match.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_sell_x_vs_single_position(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    sell_quantity = mpf(10)

    agent1 = Agent(holdings={'A': 1000, 'B': 0})
    one_position = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=fee
    ).swap(
        agent1, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )

    agent1_copy = Agent(holdings={'A': 1000, 'B': 0})
    one_position_feeless = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent1_copy, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )

    agent2 = Agent(holdings={'A': 1000, 'B': 0})
    whole_pool = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing=tick_spacing,
        fee=fee
    ).swap(
        agent2, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )

    agent2_copy = Agent(holdings={'A': 1000, 'B': 0})
    whole_pool_feeless = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent2_copy, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )

    effective_fee_one_pool = (agent1_copy.holdings['B'] - agent1.holdings['B']) / (
                agent1_copy.initial_holdings['B'] - agent1_copy.holdings['B'])
    effective_fee_whole_pool = (agent2_copy.holdings['B'] - agent2.holdings['B']) / (
                agent2_copy.initial_holdings['B'] - agent2_copy.holdings['B'])
    if agent1.holdings['A'] != agent2.holdings['A']:
        raise AssertionError('Sell quantity was not applied correctly.')
    if agent1.holdings['B'] != pytest.approx(agent2.holdings['B'], rel=1e-6):
        raise AssertionError('Buy quantity was not calculated correctly.')
    if effective_fee_whole_pool != pytest.approx(effective_fee_one_pool, rel=1e-5):
        raise AssertionError('Fee levels do not match.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_sell_y_vs_single_position(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    sell_quantity = mpf(10)

    agent1 = Agent(holdings={'A': 0, 'B': 1000})
    one_position = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=fee
    ).swap(
        agent1, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )

    agent1_copy = Agent(holdings={'A': 0, 'B': 1000})
    one_position_feeless = ConcentratedLiquidityPosition(
        assets={'A': mpf(10 / price), 'B': mpf(10)},
        min_tick=price_to_tick(price, tick_spacing),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent1_copy, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )

    agent2 = Agent(holdings={'A': 0, 'B': 1000})
    whole_pool = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing=tick_spacing,
        fee=fee
    ).swap(
        agent2, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )

    agent2_copy = Agent(holdings={'A': 0, 'B': 1000})
    whole_pool_feeless = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf.sqrt(one_position.invariant),
        tick_spacing=tick_spacing,
        fee=0
    ).swap(
        agent2_copy, tkn_buy='A', tkn_sell='B', sell_quantity=sell_quantity
    )

    effective_fee_one_pool = (agent1_copy.holdings['A'] - agent1.holdings['A']) / (
                agent1_copy.initial_holdings['A'] - agent1_copy.holdings['A'])
    effective_fee_whole_pool = (agent2_copy.holdings['A'] - agent2.holdings['A']) / (
                agent2_copy.initial_holdings['A'] - agent2_copy.holdings['A'])
    if agent1.holdings['B'] != agent2.holdings['B']:
        raise AssertionError('Sell quantity was not applied correctly.')
    if agent1.holdings['A'] != pytest.approx(agent2.holdings['A'], rel=1e-6):
        raise AssertionError('Buy quantity was not calculated correctly.')
    if effective_fee_whole_pool != pytest.approx(effective_fee_one_pool, rel=1e-6):
        raise AssertionError('Fee levels do not match.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_pool_sell_spot(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    sell_quantity = mpf(1) / 10000000000

    initial_state = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf(1000),
        tick_spacing=tick_spacing,
        fee=fee
    )

    agent = Agent(holdings={'A': 1000, 'B': 0})
    sell_spot = initial_state.sell_spot(tkn_sell='A', tkn_buy='B', fee=fee)
    initial_state.swap(
        agent, tkn_buy='B', tkn_sell='A', sell_quantity=sell_quantity
    )
    ex_price = agent.holdings['B'] / (agent.initial_holdings['A'] - agent.holdings['A'])
    if ex_price != pytest.approx(sell_spot, rel=1e-20):
        raise AssertionError('Sell spot price was not calculated correctly.')


@given(st.integers(min_value=1, max_value=100), fee_strategy)
def test_pool_buy_spot(initial_tick, fee):
    tick_spacing = 100
    price = mpf(tick_to_price(initial_tick * tick_spacing + tick_spacing // 2))
    buy_quantity = mpf(1) / 10000000000

    initial_state = ConcentratedLiquidityPoolState(
        asset_list=['A', 'B'],
        sqrt_price=mpf.sqrt(price),
        liquidity=mpf(1000),
        tick_spacing=tick_spacing,
        fee=fee
    )

    agent = Agent(holdings={'B': 1000, 'A': 0})
    buy_spot = initial_state.buy_spot(tkn_buy='A', tkn_sell='B', fee=fee)
    initial_state.swap(
        agent, tkn_buy='A', tkn_sell='B', buy_quantity=buy_quantity
    )
    ex_price = (agent.initial_holdings['B'] - agent.holdings['B']) / agent.holdings['A']
    if ex_price != pytest.approx(buy_spot, rel=1e-20):
        raise AssertionError('Buy spot price was not calculated correctly.')