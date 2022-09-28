import pytest

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.trade_strategies import random_swaps, stableswap_arbitrage
from hydradx.model.amm.global_state import GlobalState
from hydradx.model import run
from hydradx.model import processing

# # fix value, i.e. fix p_x^y * x + y
# D = 200000000
# x_step_size = 500000
# x_min = 10000000
# x_max = 200000000
# liq_depth = {}
#
# # for A in range(10, 101, 10):
# for A in [5, 10, 20, 50, 500]:
#     ann = A * 4
#
#     print("A is " + str(A))
#
#     liq_depth[A] = [None] * ((x_max - x_min) // x_step_size)
#     prices = [None] * ((x_max - x_min) // x_step_size)
#
#     i = 0
#     p_prev = 0
#     for x in range(x_min, x_max + x_step_size, x_step_size):
#         p_prev = p
#         y = StableSwapPoolState.calculate_y(x, D, ann)
#         p = spot_price([x, y], D, ann)
#         if i > 0:
#             liq_depth[A][i - 1] = x_step_size / (p - p_prev)
#             prices[i - 1] = p
#         # print((x, y, p))
#         # print(liq_depth[A][i-1], p)
#         i += 1
#     # print(sum(liq_depth[A]))
#     s = sum(liq_depth[A])
#     for j in range(len(liq_depth[A])):
#         # liq_depth[A][j] = liq_depth[A][j]/s
#         liq_depth[A][j] = liq_depth[A][j]
#


def testSwapInvariant():
    initial_state = GlobalState(
        pools={
            'R1/R2': StableSwapPoolState(
                tokens={
                    'R1': 100000,
                    'R2': 100000
                },
                amplification=100,
                trade_fee=0
            )
        },
        agents={
            'trader': Agent(
                holdings={'R1': 1000, 'R2': 1000},
                trade_strategy=random_swaps(pool_id='R1/R2', amount={'R1': 200, 'R2': 1000}, randomize_amount=True)
            )
        },
        external_market={'R1': 1, 'R2': 1}
    )

    pool_events = []
    agent_events = []
    new_pool = initial_state.pools['R1/R2'].copy()
    new_agent = initial_state.agents['trader'].copy()
    d = new_pool.calculate_d()
    for n in range(10):
        if n % 2 == 0:
            new_pool, new_agent = new_pool.execute_swap(
                new_agent, 'R1', 'R2', 100, 0
            )
        else:
            new_pool, new_agent = new_pool.execute_swap(
                new_agent, 'R2', 'R1', 0, 100
            )
        new_d = new_pool.calculate_d()
        if d != pytest.approx(d):
            raise AssertionError('Invariant has varied.')
        pool_events.append(new_pool.copy())
        agent_events.append(new_agent.copy())


def test_arbitrage():
    stable_pool = StableSwapPoolState(
        tokens={
            'R1': 500000,
            'R2': 500000
        },
        amplification=10,
        trade_fee=0
    )

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
                trade_strategy=stableswap_arbitrage(pool_id='R1/R2', minimum_profit=0)
            )
        },
        external_market={
            'R1': 1,
            'R2': 1
        },
        # evolve_function = fluctuate_prices(volatility={'R1': 1, 'R2': 1}, trend = {'R1': 1, 'R1': 1})
    )
    events = run.run(initial_state, time_steps=100)
    # print(events[0]['state'].pools['R1/R2'].spot_price, events[-1]['state'].pools['R1/R2'].spot_price)
    if (
        events[0]['state'].pools['R1/R2'].spot_price
        != pytest.approx(events[-1]['state'].pools['R1/R2'].spot_price, rel=1e-2)
    ):
        raise AssertionError("Arbitrageur didn't keep the price stable.")
    if (
        events[0]['state'].agents['Arbitrageur'].holdings['R1']
        + events[0]['state'].agents['Arbitrageur'].holdings['R2']
        > events[-1]['state'].agents['Arbitrageur'].holdings['R1']
        + events[-1]['state'].agents['Arbitrageur'].holdings['R2']
    ):
        raise AssertionError("Arbitrageur didn't make money.")
