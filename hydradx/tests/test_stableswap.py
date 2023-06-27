import pytest
import functools
from hydradx.model.amm import stableswap_amm as stableswap
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.trade_strategies import random_swaps, stableswap_arbitrage
from hydradx.model.amm.global_state import GlobalState
from hydradx.model import run
from hypothesis import given, strategies as st
from mpmath import mp, mpf
from hydradx.tests.strategies_omnipool import stableswap_config
mp.dps = 50

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_quantity_strategy = st.floats(min_value=1000, max_value=1000000)
fee_strategy = st.floats(min_value=0, max_value=0.1, allow_nan=False)
trade_quantity_strategy = st.floats(min_value=-1000, max_value=1000)
asset_number_strategy = st.integers(min_value=2, max_value=5)


def stable_swap_equation(d: float, a: float, n: int, reserves: list):
    """
    this is the equation that should remain true at all times within a stableswap pool
    """
    side1 = a * n ** n * sum(reserves) + d
    side2 = a * n ** n * d + d ** (n + 1) / (n ** n * functools.reduce(lambda i, j: i * j, reserves))
    return side1 == pytest.approx(side2)


@given(stableswap_config(trade_fee=0))
def test_swap_invariant(initial_pool: StableSwapPoolState):
    # print(f'testing with {len(initial_pool.asset_list)} assets')
    initial_state = GlobalState(
        pools={
            'stableswap': initial_pool
        },
        agents={
            'trader': Agent(
                holdings={tkn: 100000 for tkn in initial_pool.asset_list},
                trade_strategy=random_swaps(
                    pool_id='stableswap',
                    amount={tkn: 1000 for tkn in initial_pool.asset_list},
                    randomize_amount=True
                )
            )
        }
    )

    new_state = initial_state.copy()
    d = new_state.pools['stableswap'].calculate_d()
    for n in range(10):
        new_state = new_state.agents['trader'].trade_strategy.execute(new_state, agent_id='trader')
        new_d = new_state.pools['stableswap'].calculate_d()
        if new_d != pytest.approx(d):
            raise AssertionError('Invariant has varied.')
    if initial_state.total_wealth() != pytest.approx(new_state.total_wealth()):
        raise AssertionError('Some assets were lost along the way.')


@given(stableswap_config(trade_fee=0))
def test_round_trip_dy(initial_pool: StableSwapPoolState):
    d = initial_pool.calculate_d()
    asset_a = initial_pool.asset_list[0]
    other_reserves = [initial_pool.liquidity[a] for a in list(filter(lambda k: k != asset_a, initial_pool.asset_list))]
    y = initial_pool.calculate_y(reserves=other_reserves, d=d)
    if y != pytest.approx(initial_pool.liquidity[asset_a]) or y < initial_pool.liquidity[asset_a]:
        raise AssertionError('Round-trip calculation incorrect.')
    modified_d = initial_pool.calculate_d(initial_pool.modified_balances(delta={asset_a: 1}))
    if initial_pool.calculate_y(reserves=other_reserves, d=modified_d) != pytest.approx(y+1):
        raise AssertionError('Round-trip calculation incorrect.')


# commented out because without further work, withdraw_asset and remove_liquidity are not equivalent.
# This is because withdraw_asset was written based remove_liquidity_old, which is not equivalent to remove_liquidity.
#
# @given(stableswap_config(precision=0.000000001))
# def test_remove_asset(initial_pool: StableSwapPoolState):
#     initial_agent = Agent(
#         holdings={tkn: 0 for tkn in initial_pool.asset_list}
#     )
#     # agent holds all the shares
#     tkn_remove = initial_pool.asset_list[0]
#     pool_name = initial_pool.unique_id
#     delta_shares = min(initial_pool.shares / 2, 100)
#     initial_agent.holdings.update({initial_pool.unique_id: delta_shares + 1})
#     withdraw_shares_pool, withdraw_shares_agent = stableswap.remove_liquidity(
#         initial_pool, initial_agent, delta_shares, tkn_remove
#     )
#     delta_tkn = withdraw_shares_agent.holdings[tkn_remove] - initial_agent.holdings[tkn_remove]
#     withdraw_asset_pool, withdraw_asset_agent = stableswap.execute_withdraw_asset(
#         initial_pool.copy(), initial_agent.copy(), delta_tkn, tkn_remove
#     )
#     if (
#         withdraw_asset_agent.holdings[tkn_remove] != pytest.approx(withdraw_shares_agent.holdings[tkn_remove])
#         or withdraw_asset_agent.holdings[pool_name] != pytest.approx(withdraw_shares_agent.holdings[pool_name])
#         or withdraw_shares_pool.liquidity[tkn_remove] != pytest.approx(withdraw_asset_pool.liquidity[tkn_remove])
#         or withdraw_shares_pool.shares != pytest.approx(withdraw_asset_pool.shares)
#     ):
#         raise AssertionError("Asset values don't match.")


@given(stableswap_config(precision=0.000000001))
def test_buy_shares(initial_pool: StableSwapPoolState):
    initial_agent = Agent(
        holdings={tkn: 0 for tkn in initial_pool.asset_list + [initial_pool.unique_id]}
    )
    # agent holds all the shares
    tkn_add = initial_pool.asset_list[0]
    pool_name = initial_pool.unique_id
    delta_tkn = 10
    initial_agent.holdings.update({tkn_add: 10})
    add_liquidity_pool, add_liquidity_agent = stableswap.add_liquidity(
        initial_pool, initial_agent, delta_tkn, tkn_add
    )
    delta_shares = add_liquidity_agent.holdings[pool_name] - initial_agent.holdings[pool_name]
    buy_shares_pool, buy_shares_agent = stableswap.execute_buy_shares(
        initial_pool.copy(), initial_agent.copy(), delta_shares, tkn_add, fail_overdraft=False
    )

    if (
        add_liquidity_agent.holdings[tkn_add] != pytest.approx(buy_shares_agent.holdings[tkn_add])
        or add_liquidity_agent.holdings[pool_name] != pytest.approx(buy_shares_agent.holdings[pool_name])
        or add_liquidity_pool.liquidity[tkn_add] != pytest.approx(buy_shares_pool.liquidity[tkn_add])
        or add_liquidity_pool.shares != pytest.approx(buy_shares_pool.shares)
        or add_liquidity_pool.calculate_d() != pytest.approx(buy_shares_pool.calculate_d())
    ):
        raise AssertionError("Asset values don't match.")


@given(stableswap_config(asset_dict={'R1': 1000000, 'R2': 1000000}, trade_fee=0))
def test_arbitrage(stable_pool):
    initial_state = GlobalState(
        pools={
            'R1/R2': stable_pool
        },
        agents={
            'Trader': Agent(
                holdings={'R1': 1000000, 'R2': 1000000},
                trade_strategy=random_swaps(pool_id='R1/R2', amount={'R1': 10000, 'R2': 10000}, randomize_amount=True)
            ),
            'Arbitrageur': Agent(
                holdings={'R1': 1000000, 'R2': 1000000},
                trade_strategy=stableswap_arbitrage(pool_id='R1/R2', minimum_profit=0, precision=0.000001)
            )
        },
        external_market={
            'R1': 1,
            'R2': 1
        },
        # evolve_function = fluctuate_prices(volatility={'R1': 1, 'R2': 1}, trend = {'R1': 1, 'R1': 1})
    )
    events = run.run(initial_state, time_steps=10, silent=True)
    # print(events[0].pools['R1/R2'].spot_price, events[-1].pools['R1/R2'].spot_price)
    if (
        events[0].pools['R1/R2'].spot_price
        != pytest.approx(events[-1].pools['R1/R2'].spot_price)
    ):
        raise AssertionError(f"Arbitrageur didn't keep the price stable."
                             f"({events[0].pools['R1/R2'].spot_price})"
                             f"{events[-1].pools['R1/R2'].spot_price}")
    if (
        events[0].agents['Arbitrageur'].holdings['R1']
        + events[0].agents['Arbitrageur'].holdings['R2']
        > events[-1].agents['Arbitrageur'].holdings['R1']
        + events[-1].agents['Arbitrageur'].holdings['R2']
    ):
        raise AssertionError("Arbitrageur didn't make money.")


@given(stableswap_config(trade_fee=0))
def test_add_remove_liquidity(initial_pool: StableSwapPoolState):
    lp_tkn = initial_pool.asset_list[0]
    lp = Agent(
        holdings={lp_tkn: 10000}
    )

    add_liquidity_state, add_liquidity_agent = stableswap.add_liquidity(
        initial_pool, old_agent=lp, quantity=10000, tkn_add=lp_tkn
    )
    if not stable_swap_equation(
        add_liquidity_state.calculate_d(),
        add_liquidity_state.amplification,
        add_liquidity_state.n_coins,
        add_liquidity_state.liquidity.values()
    ):
        raise AssertionError('Stableswap equation does not hold after add liquidity operation.')

    remove_liquidity_state, remove_liquidity_agent = add_liquidity_state.remove_liquidity(
        add_liquidity_state,
        add_liquidity_agent,
        quantity=add_liquidity_agent.holdings[initial_pool.unique_id],
        tkn_remove=lp_tkn
    )
    if not stable_swap_equation(
        remove_liquidity_state.calculate_d(),
        remove_liquidity_state.amplification,
        remove_liquidity_state.n_coins,
        remove_liquidity_state.liquidity.values()
    ):
        raise AssertionError('Stableswap equation does not hold after remove liquidity operation.')
    if remove_liquidity_agent.holdings[lp_tkn] != pytest.approx(lp.holdings[lp_tkn]):
        raise AssertionError('LP did not get the same balance back when withdrawing liquidity.')


def test_curve_style_withdraw_fees():
    initial_state = stableswap.StableSwapPoolState(
        tokens={
            'USDA': 1000000,
            'USDB': 1000000,
            'USDC': 1000000,
            'USDD': 1000000,
        }, amplification=100, trade_fee=0.003,
        unique_id='test_pool'
    )
    initial_agent = Agent(
        holdings={'USDA': 100000}
    )
    test_state, test_agent = stableswap.execute_add_liquidity(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        quantity=initial_agent.holdings['USDA'],
        tkn_add='USDA',
    )

    stable_state, stable_agent = stableswap.execute_remove_liquidity(
        state=test_state.copy(),
        agent=test_agent.copy(),
        shares_removed=test_agent.holdings['test_pool'],
        tkn_remove='USDB'
    )
    effective_fee_withdraw = 1 - stable_agent.holdings['USDB'] / initial_agent.holdings['USDA']

    swap_state, swap_agent = stableswap.execute_swap(
        initial_state.copy(),
        initial_agent.copy(),
        tkn_sell='USDA',
        tkn_buy='USDB',
        sell_quantity=initial_agent.holdings['USDA']
    )
    effective_fee_swap = 1 - swap_agent.holdings['USDB'] / initial_agent.holdings['USDA']

    if effective_fee_withdraw <= effective_fee_swap:
        raise AssertionError('Withdraw fee is not higher than swap fee.')

