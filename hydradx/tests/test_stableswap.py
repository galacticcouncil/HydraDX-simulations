import copy
import functools

import pytest
from hypothesis import given, strategies as st
from mpmath import mp, mpf

from hydradx.model import run
from hydradx.model.amm import stableswap_amm as stableswap
from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_swap
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import random_swaps, stableswap_arbitrage
from hydradx.tests.strategies_omnipool import stableswap_config

mp.dps = 50

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_quantity_strategy = st.floats(min_value=1000, max_value=1000000)
fee_strategy = st.floats(min_value=0, max_value=0.1, allow_nan=False)
trade_quantity_strategy = st.floats(min_value=-1000, max_value=1000)
asset_number_strategy = st.integers(min_value=2, max_value=5)


def stable_swap_equation(pool: StableSwapPoolState):  # d: float, a: float, n: int, reserves: list):
    """
    this is the equation that should remain true at all times within a stableswap pool
    """
    a = pool.ann
    d = pool.d
    n = pool.n_coins
    reserves = list(pool.liquidity.values())
    side1 = a * sum(reserves) + d
    side2 = a * d + d ** (n + 1) / (n ** n * functools.reduce(lambda i, j: i * j, reserves))
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


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000)
       )
def test_spot_price_two_assets(token_a: int, token_b: int, amp: int):
    initial_pool = StableSwapPoolState(
        tokens={"A": token_a, "B": token_b},
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap'
    )
    spot_price_initial = initial_pool.spot_price()

    trade_size = 1
    initial_agent = Agent(holdings={"A": 1, "B": 1})
    swap_state, swap_agent = stableswap.simulate_swap(
        old_state=initial_pool,
        old_agent=initial_agent,
        tkn_sell="A", tkn_buy="B", sell_quantity=trade_size
    )
    delta_a = swap_state.liquidity["A"] - initial_pool.liquidity["A"]
    delta_b = initial_pool.liquidity["B"] - swap_state.liquidity["B"]
    exec_price = delta_a / delta_b

    spot_price_final = swap_state.spot_price()

    if spot_price_initial > exec_price and (spot_price_initial - exec_price) / spot_price_initial > 10e-10:
        raise AssertionError('Initial spot price should be lower than execution price.')
    if exec_price > spot_price_final and (exec_price - spot_price_final) / spot_price_final > 10e-10:
        raise AssertionError('Execution price should be lower than final spot price.')


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000),
       st.integers(min_value=1, max_value=3)
       )
def test_spot_price(token_a: int, token_b: int, token_c: int, token_d: int, amp: int, i: int):
    initial_pool = StableSwapPoolState(
        tokens={"A": token_a, "B": token_b, "C": token_c, "D": token_d},
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap'
    )
    tkns = ["A", "B", "C", "D"]
    spot_price_initial = initial_pool.price(tkns[i], "A")

    trade_size = 1
    initial_agent = Agent(holdings={"A": 1.1, "B": 1.1, "C": 1.1, "D": 1.1})
    swap_agent = initial_agent.copy()
    swap_pool = initial_pool.copy().swap(swap_agent, tkn_sell="A", tkn_buy=tkns[i], sell_quantity=trade_size)
    delta_a = swap_pool.liquidity["A"] - initial_pool.liquidity["A"]
    delta_b = initial_pool.liquidity[tkns[i]] - swap_pool.liquidity[tkns[i]]
    exec_price = delta_a / delta_b

    spot_price_final = swap_pool.price(tkns[i], "A")

    if spot_price_initial > exec_price and (spot_price_initial - exec_price) / spot_price_initial > 10e-10:
        raise AssertionError('Initial spot price should be lower than execution price.')
    if exec_price > spot_price_final and (exec_price - spot_price_final) / spot_price_final > 10e-10:
        raise AssertionError('Execution price should be lower than final spot price.')


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000)
       )
def test_share_price(token_a: int, token_b: int, amp: int):
    tokens = {"A": token_a, "B": token_b}
    initial_pool = StableSwapPoolState(
        tokens=tokens,
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap'
    )

    share_price_initial = initial_pool.share_price()

    agent = Agent(holdings={"A": 100000000, "B": 100000000})
    delta_tkn = 1
    shares_initial = initial_pool.shares
    add_pool = initial_pool.copy()
    add_pool.add_liquidity(agent, quantity=delta_tkn, tkn_add="A")
    shares_final = add_pool.shares
    delta_a = add_pool.liquidity["A"] - tokens["A"]
    delta_s = shares_final - shares_initial
    exec_price = delta_a / delta_s

    if share_price_initial > exec_price and (share_price_initial - exec_price) / share_price_initial > 10e-10:
        raise AssertionError('Initial share price should be lower than execution price.')

    # now we test withdraw

    delta_s = agent.holdings['stableswap']
    share_price_initial = add_pool.share_price()
    a_initial = add_pool.liquidity['A']
    withdraw_pool = add_pool.copy()
    withdraw_pool.remove_liquidity(agent, shares_removed=delta_s, tkn_remove='A')
    a_final = withdraw_pool.liquidity['A']
    exec_price = (a_initial - a_final) / delta_s

    if share_price_initial < exec_price and (exec_price - share_price_initial) / share_price_initial > 10e-10:
        raise AssertionError('Initial share price should be higher than execution price.')


@given(stableswap_config(trade_fee=0))
def test_round_trip_dy(initial_pool: StableSwapPoolState):
    d = initial_pool.calculate_d()
    asset_a = initial_pool.asset_list[0]
    other_reserves = [initial_pool.liquidity[a] for a in list(filter(lambda k: k != asset_a, initial_pool.asset_list))]
    y = initial_pool.calculate_y(reserves=other_reserves, d=d)
    if y != pytest.approx(initial_pool.liquidity[asset_a]) or y < initial_pool.liquidity[asset_a]:
        raise AssertionError('Round-trip calculation incorrect.')
    modified_d = initial_pool.calculate_d(initial_pool.modified_balances(delta={asset_a: 1}))
    if initial_pool.calculate_y(reserves=other_reserves, d=modified_d) != pytest.approx(y + 1):
        raise AssertionError('Round-trip calculation incorrect.')


@given(stableswap_config(precision=0.000000001, trade_fee=0))
def test_remove_asset(initial_pool: StableSwapPoolState):
    initial_agent = Agent(
        holdings={tkn: 0 for tkn in initial_pool.asset_list}
    )
    # agent holds all the shares
    tkn_remove = initial_pool.asset_list[0]
    pool_name = initial_pool.unique_id
    delta_shares = min(initial_pool.shares / 2, 100)
    initial_agent.holdings.update({initial_pool.unique_id: delta_shares + 1})
    withdraw_shares_pool, withdraw_shares_agent = stableswap.simulate_remove_liquidity(
        initial_pool, initial_agent, delta_shares, tkn_remove
    )
    delta_tkn = withdraw_shares_agent.holdings[tkn_remove] - initial_agent.holdings[tkn_remove]
    withdraw_asset_pool, withdraw_asset_agent = initial_pool.copy(), initial_agent.copy()
    withdraw_asset_pool.withdraw_asset(
        withdraw_asset_agent, delta_tkn, tkn_remove
    )
    if (
            withdraw_asset_agent.holdings[tkn_remove] != pytest.approx(withdraw_shares_agent.holdings[tkn_remove])
            or withdraw_asset_agent.holdings[pool_name] != pytest.approx(withdraw_shares_agent.holdings[pool_name])
            or withdraw_shares_pool.liquidity[tkn_remove] != pytest.approx(withdraw_asset_pool.liquidity[tkn_remove])
            or withdraw_shares_pool.shares != pytest.approx(withdraw_asset_pool.shares)
    ):
        raise AssertionError("Asset values don't match.")


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

    add_liquidity_pool, add_liquidity_agent = stableswap.simulate_add_liquidity(
        initial_pool, initial_agent, delta_tkn, tkn_add
    )
    delta_shares = add_liquidity_agent.holdings[pool_name] - initial_agent.holdings[pool_name]
    buy_shares_pool, buy_shares_agent = stableswap.simulate_buy_shares(
        initial_pool.copy(), initial_agent.copy(), delta_shares, tkn_add, fail_overdraft=False
    )

    if (
            add_liquidity_agent.holdings[tkn_add] != pytest.approx(buy_shares_agent.holdings[tkn_add], rel=1e-12)
            or add_liquidity_agent.holdings[pool_name] != pytest.approx(buy_shares_agent.holdings[pool_name], rel=1e-12)
            or add_liquidity_pool.liquidity[tkn_add] != pytest.approx(buy_shares_pool.liquidity[tkn_add], rel=1e-12)
            or add_liquidity_pool.shares != pytest.approx(buy_shares_pool.shares, rel=1e-12)
            or add_liquidity_pool.calculate_d() != pytest.approx(buy_shares_pool.calculate_d(), rel=1e-12)
    ):
        raise AssertionError("Asset values don't match.")


@given(
    st.floats(min_value=0.0001, max_value=.9999),
    st.integers(min_value=1, max_value=1000000),
    st.floats(min_value=0.0001, max_value=0.1)
)
def test_arbitrage_profitability(trade_fraction, amp, fee):
    initial_pool = StableSwapPoolState(
        tokens={
            tkn: mpf(10000) for tkn in ['R' + str(n) for n in range(1, 5)]
        },
        amplification=amp,
        trade_fee=fee
    )
    trader = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list}
    )
    arbitrageur = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list},
        trade_strategy=stableswap_arbitrage(
            pool_id='stableswap', minimum_profit=0, precision=1e-6
        )
    )
    swapped_pool: StableSwapPoolState = initial_pool.copy().swap(
        agent=trader,
        tkn_sell='R1', tkn_buy='R2',
        buy_quantity=trade_fraction * initial_pool.liquidity['R1']
    )
    pre_arb_state = GlobalState(
        pools={'stableswap': swapped_pool},
        agents={'Arbitrageur': arbitrageur},
        external_market={'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1}
    )
    arbitrageur.trade_strategy.execute(pre_arb_state, 'Arbitrageur')
    if (
        sum(arbitrageur.holdings.values()) < sum(arbitrageur.initial_holdings.values())
    ):
        raise AssertionError(
            f"Arbitrageur lost money. "
            f"(start: {sum(arbitrageur.initial_holdings.values())}, end: {sum(arbitrageur.holdings.values())})"
        )


@given(
    st.floats(min_value=0.0001, max_value=.9999),
    st.integers(min_value=1, max_value=10000)
)
def test_arbitrage_efficacy(trade_fraction, amp):
    initial_pool = StableSwapPoolState(
        tokens={
            tkn: mpf(10000) for tkn in ['R' + str(n) for n in range(1, 5)]
        },
        amplification=amp,
        trade_fee=0
    )
    trader = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list}
    )
    arbitrageur = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list},
        trade_strategy=stableswap_arbitrage(
            pool_id='stableswap', minimum_profit=0, precision=1e-6
        )
    )
    swapped_pool: StableSwapPoolState = initial_pool.copy().swap(
        agent=trader,
        tkn_sell='R1', tkn_buy='R2',
        buy_quantity=trade_fraction * initial_pool.liquidity['R1']
    )
    pre_arb_state = GlobalState(
        pools={'stableswap': swapped_pool},
        agents={'Arbitrageur': arbitrageur},
        external_market={'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1}
    )
    final_state = arbitrageur.trade_strategy.execute(pre_arb_state.copy(), 'Arbitrageur')
    final_pool = final_state.pools['stableswap']

    if (
        initial_pool.price('R1', 'R2')
        != pytest.approx(final_pool.price('R1', 'R2'), abs=1e-6)
    ):
        raise AssertionError(f"Arbitrageur didn't keep the price stable."
                             f"({initial_pool.price('R1', 'R2')},"
                             f"{final_pool.price('R1', 'R2')}")


@given(stableswap_config(trade_fee=0))
def test_add_remove_liquidity(initial_pool: StableSwapPoolState):
    lp_tkn = initial_pool.asset_list[0]
    lp = Agent(
        holdings={lp_tkn: 10000}
    )

    add_liquidity_state, add_liquidity_agent = stableswap.simulate_add_liquidity(
        initial_pool, old_agent=lp, quantity=10000, tkn_add=lp_tkn
    )
    if not stable_swap_equation(add_liquidity_state):
        raise AssertionError('Stableswap equation does not hold after add liquidity operation.')

    remove_liquidity_state, remove_liquidity_agent = stableswap.simulate_remove_liquidity(
        add_liquidity_state,
        add_liquidity_agent,
        quantity=add_liquidity_agent.holdings[initial_pool.unique_id],
        tkn_remove=lp_tkn
    )
    if not stable_swap_equation(remove_liquidity_state):
        raise AssertionError('Stableswap equation does not hold after remove liquidity operation.')
    if remove_liquidity_agent.holdings[lp_tkn] != pytest.approx(lp.holdings[lp_tkn]):
        raise AssertionError('LP did not get the same balance back when withdrawing liquidity.')


#
# def test_curve_style_withdraw_fees():
#     initial_state = stableswap.StableSwapPoolState(
#         tokens={
#             'USDA': 1000000,
#             'USDB': 1000000,
#             'USDC': 1000000,
#             'USDD': 1000000,
#         }, amplification=100, trade_fee=0.003,
#         unique_id='test_pool'
#     )
#     initial_agent = Agent(
#         holdings={'USDA': 100000}
#     )
#     test_state, test_agent = stableswap.simulate_add_liquidity(
#         old_state=initial_state.copy(),
#         old_agent=initial_agent.copy(),
#         quantity=initial_agent.holdings['USDA'],
#         tkn_add='USDA',
#     )
#
#     stable_state, stable_agent = stableswap.simulate_remove_liquidity(
#         old_state=test_state.copy(),
#         old_agent=test_agent.copy(),
#         quantity=test_agent.holdings['test_pool'],
#         tkn_remove='USDB'
#     )
#     effective_fee_withdraw = 1 - stable_agent.holdings['USDB'] / initial_agent.holdings['USDA']
#
#     swap_state, swap_agent = stableswap.simulate_swap(
#         initial_state.copy(),
#         initial_agent.copy(),
#         tkn_sell='USDA',
#         tkn_buy='USDB',
#         sell_quantity=initial_agent.holdings['USDA']
#     )
#     effective_fee_swap = 1 - swap_agent.holdings['USDB'] / initial_agent.holdings['USDA']
#
#     if effective_fee_withdraw <= effective_fee_swap:
#         raise AssertionError('Withdraw fee is not higher than swap fee.')


@given(
    st.integers(min_value=1, max_value=1000000),
    st.integers(min_value=10000, max_value=10000000),
)
def test_exploitability(initial_lp: int, trade_size: int):
    assets = ['USDA', 'USDB']
    initial_tvl = 1000000

    initial_state = StableSwapPoolState(
        tokens={tkn: mpf(initial_tvl / len(assets)) for tkn in assets},
        amplification=1000,
        trade_fee=0
    )
    initial_agent = Agent(
        holdings={tkn: mpf(initial_lp / len(assets)) for tkn in assets},
    )

    lp_state, lp_agent = initial_state.copy(), initial_agent.copy()
    for tkn in initial_state.asset_list:
        lp_state.add_liquidity(
            agent=lp_agent,
            quantity=lp_agent.holdings[tkn],
            tkn_add=tkn
        )

    trade_state, trade_agent = lp_state.copy(), lp_agent.copy()
    trade_agent.holdings['USDA'] = trade_size
    trade_state.swap(
        agent=trade_agent,
        tkn_sell='USDA',
        tkn_buy='USDB',
        sell_quantity=trade_size
    )

    withdraw_state, withdraw_agent = trade_state.copy(), trade_agent.copy()
    withdraw_state.remove_liquidity(
        agent=withdraw_agent,
        shares_removed=trade_agent.holdings['stableswap'],
        tkn_remove='USDA'
    )

    max_arb_size = trade_size
    min_arb_size = 0

    for i in range(10):
        final_state, final_agent = withdraw_state.copy(), withdraw_agent.copy()
        arb_size = (max_arb_size - min_arb_size) / 2 + min_arb_size
        final_state.swap(
            agent=final_agent,
            tkn_sell='USDB',
            tkn_buy='USDA',
            buy_quantity=arb_size
        )

        profit = sum(final_agent.holdings.values()) - trade_size - initial_lp
        if profit > 0:
            raise AssertionError(f'Agent profited by exploit ({profit}).')

        if initial_state.price('USDA', 'USDB') < final_state.price('USDA', 'USDB'):
            min_arb_size = arb_size
        elif initial_state.price('USDA', 'USDB') > final_state.price('USDA', 'USDB'):
            max_arb_size = arb_size
        else:
            break


@given(
    st.integers(min_value=1, max_value=1000000),
    st.floats(min_value=0.00001, max_value=0.99999)
)
def test_swap_one(amplification, swap_fraction):
    initial_state = StableSwapPoolState(
        tokens={
            'USDA': mpf(1000000),
            'USDB': mpf(1000000),
            'USDC': mpf(1000000),
            'USDD': mpf(1000000),
        }, amplification=amplification, trade_fee=0,
    )
    stablecoin = initial_state.asset_list[-1]
    tkn_sell = initial_state.asset_list[0]
    buy_quantity = initial_state.liquidity[tkn_sell] * swap_fraction
    initial_agent = Agent(
        holdings={tkn_sell: 10000000000000}
    )
    sell_agent = initial_agent.copy()
    sell_state = initial_state.copy().swap_one(
        agent=sell_agent,
        tkn_sell=tkn_sell,
        quantity=buy_quantity
    )
    if not stable_swap_equation(sell_state):
        raise AssertionError('Stableswap equation does not hold after swap operation.')

    for tkn in initial_state.asset_list:
        if (
                sell_state.price(tkn, stablecoin)
                != pytest.approx(initial_state.price(tkn, stablecoin))
                and tkn != tkn_sell
        ):
            raise AssertionError('Spot price changed for non-swapped token.')

    if sell_state.price(tkn_sell, stablecoin) >= initial_state.price(tkn_sell, stablecoin):
        raise AssertionError('Spot price increased for swapped token.')

    if sell_state.d != pytest.approx(initial_state.d):
        raise AssertionError('D changed after sell operation.')

    tkn_buy = sell_state.asset_list[0]
    buy_agent = sell_agent.copy()
    buy_state = sell_state.copy().swap_one(
        agent=buy_agent,
        tkn_buy=tkn_buy,
        quantity=buy_quantity
    )
    if not stable_swap_equation(buy_state):
        raise AssertionError('Stableswap equation does not hold after swap operation.')

    for tkn in initial_state.asset_list:
        if (
                buy_state.price(tkn, stablecoin)
                != pytest.approx(initial_state.price(tkn, stablecoin))
                and tkn != tkn_buy
        ):
            raise AssertionError('Spot price changed for non-swapped token.')

    if buy_state.price(tkn_buy, stablecoin) <= sell_state.price(tkn_buy, stablecoin):
        raise AssertionError('Spot price decreased for swapped token.')

    if buy_state.d != pytest.approx(initial_state.d):
        raise AssertionError('D changed after buy operation.')


# @given(st.integers(min_value=1, max_value=999))
def test_amplification_change_exploit():  # (end_amp):
    start_amp = 1000
    end_amp = 100
    initial_pool = StableSwapPoolState(
        tokens={
            'USDA': mpf(1000000),
            'USDB': mpf(1000000),
            'USDC': mpf(1000000),
            'USDD': mpf(1000000),
        },
        amplification=start_amp,
        trade_fee=0.001,
    )
    initial_agent = Agent(
        holdings={'USDA': 10000000000000, 'USDB': 10000000000000},
        # trade_strategy=stableswap_arbitrage(pool_id='stableswap', minimum_profit=0, precision=0.000001)
    )
    initial_state = GlobalState(
        pools={
            'stableswap': initial_pool
        },
        agents={
            'trader': initial_agent,
            'arbitrageur': Agent(
                holdings={tkn: 10000000000000 for tkn in initial_pool.asset_list},
                trade_strategy=stableswap_arbitrage(pool_id='stableswap', minimum_profit=1, precision=0.0001)
            )
        },
        external_market={
            'USDA': 1,
            'USDB': 1,
            'USDC': 1,
            'USDD': 1,
        }
    )
    sell_quantity = initial_pool.liquidity['USDA'] * 10
    sell_state = initial_state.copy()
    sell_state.pools['stableswap'].swap(
        agent=sell_state.agents['trader'],
        tkn_sell='USDA',
        tkn_buy='USDB',
        sell_quantity=sell_quantity
    )
    duration = int((start_amp - end_amp) / end_amp * 99.9)
    sell_state.pools['stableswap'].set_amplification(end_amp, duration)
    events = run.run(sell_state, time_steps=duration, silent=True)
    final_pool: StableSwapPoolState = events[-1].pools['stableswap']
    final_agent: Agent = events[-1].agents['trader']
    final_pool.swap(
        agent=final_agent,
        tkn_sell='USDB',
        tkn_buy='USDA',
        buy_quantity=sell_quantity
    )
    loss = sum(initial_pool.liquidity.values()) - sum(final_pool.liquidity.values())
    if loss > 0:
        raise AssertionError(F"Pool lost money. loss: {round(loss / sum(initial_pool.liquidity.values()) * 100, 5)}%")


@given(
    liquidity_stepdown=st.integers(min_value=-1000000, max_value=1000000),
    assets_number=st.integers(min_value=2, max_value=4),
    amplification=st.integers(min_value=1, max_value=10000),
    trade_fee=st.floats(min_value=0, max_value=0.1)
)
def test_buy_sell_spot(
        liquidity_stepdown: int,
        assets_number: int,
        amplification: int,
        trade_fee: float
):
    base_liquidity = mpf(10000000)
    initial_state = StableSwapPoolState(
        tokens={
            tkn: base_liquidity + liquidity_stepdown * n
            for n, tkn in enumerate(['R' + str(n) for n in range(1, assets_number + 1)])
        },
        amplification=amplification,
        trade_fee=trade_fee
    )
    tkn_sell = 'R1'
    tkn_buy = 'R2'
    agent = Agent(holdings={tkn: mpf(1000) for tkn in initial_state.asset_list})
    test_state, test_agent = initial_state.copy(), agent.copy()
    buy_quantity = 0.001
    r1_per_r2 = initial_state.buy_spot(tkn_buy='R2', tkn_sell='R1')
    r2_per_r1 = initial_state.sell_spot(tkn_sell='R1', tkn_buy='R2')
    test_state.swap(
        agent=test_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )
    actual_sell_quantity = test_agent.initial_holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    actual_buy_quantity = test_agent.holdings[tkn_buy] - test_agent.initial_holdings[tkn_buy]
    ex_price_r1 = actual_buy_quantity / actual_sell_quantity
    ex_price_r2 = actual_sell_quantity / actual_buy_quantity
    if r2_per_r1 != pytest.approx(ex_price_r1):
        raise AssertionError(f'Sell spot R1 ({r2_per_r1}) != execution price ({ex_price_r1}).')
    if r1_per_r2 != pytest.approx(ex_price_r2):
        raise AssertionError(f'Buy spot R2 ({r1_per_r2}) != execution price ({ex_price_r2}).')


def test_share_prices():
    tokens = {'USDT': mpf(1000000), 'USDC': mpf(1200000)}
    pool = StableSwapPoolState(tokens, mpf(100), trade_fee=mpf(0.01))
    spot = pool.share_price('USDT')
    add_liq_price = pool.add_liquidity_spot('USDT')
    if add_liq_price <= spot:
        raise AssertionError('Add liquidity price should be higher than spot price.')
    buy_shares_price = pool.buy_shares_spot('USDT')
    if buy_shares_price <= spot:
        raise AssertionError('Buy shares price should be higher than spot price.')
    remove_liq_price = pool.remove_liquidity_spot('USDT')
    if remove_liq_price >= spot:
        raise AssertionError('Remove liquidity price should be lower than spot price.')
    withdraw_asset_price = pool.withdraw_asset_spot('USDT')
    if withdraw_asset_price >= spot:
        raise AssertionError('Withdraw asset price should be lower than spot price.')
