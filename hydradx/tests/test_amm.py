import copy
import random

import pytest
from hypothesis import given, strategies as st
from hydradx.tests.test_omnipool_amm import omnipool_config
from hydradx.tests.test_basilisk_amm import constant_product_pool_config
from hydradx.model.amm.global_state import GlobalState, fluctuate_prices
from hydradx.model.amm.agents import Agent

from hydradx.model import run
from hydradx.model import plot_utils as pu
from hydradx.model import processing
from hydradx.model.amm.trade_strategies import random_swaps, invest_all
from hydradx.model.amm.amm import AMM

import sys
import random

sys.path.append('../..')

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=1000, max_value=1000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    return_dict = {
        'HDX': draw(asset_price_strategy),
        'USD': 1
    }
    return_dict.update({
        f"{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}": draw(asset_price_strategy)
        for i in range(token_count - 2)
    })
    return return_dict


@st.composite
def agent_config(
    draw,
    holdings: dict = None,
    asset_list: list = None,
    trade_strategy: any = None
):
    return Agent(
        holdings=holdings or {
            tkn: draw(asset_quantity_strategy)
            for tkn in asset_list
        },
        trade_strategy=trade_strategy
    )


@st.composite
def global_state_config(
        draw,
        asset_dict: dict[str: float] = None,
        pools=None,
        agents=None
) -> GlobalState:
    market_prices = asset_dict or draw(assets_config())
    asset_list = list(market_prices.keys())
    if not pools:
        pools = {}
        # a Basilisk pool for every asset pair
        for x in range(len(asset_list) - 1):
            for y in range(x + 1, len(asset_list)):
                x_quantity = draw(asset_quantity_strategy)
                pools.update({
                    f'{asset_list[x]}/{asset_list[y]}':
                    draw(constant_product_pool_config(
                        asset_dict={
                            asset_list[x]: x_quantity,
                            asset_list[y]: x_quantity * market_prices[asset_list[x]] / market_prices[asset_list[y]]
                        },
                        trade_fee=draw(fee_strategy)
                    ))
                })
        # and an Omnipool
        usd_price_lrna = 1  # draw(asset_price_strategy)
        market_prices.update({'LRNA': usd_price_lrna})
        liquidity = {tkn: 1000 for tkn in asset_list}
        pools.update({
            'omnipool': draw(omnipool_config(
                asset_dict={
                    tkn: {
                        'liquidity': liquidity[tkn],
                        'LRNA': liquidity[tkn] * market_prices[tkn] / usd_price_lrna
                    }
                    for tkn in asset_list
                },
                lrna_fee=draw(fee_strategy),
                asset_fee=draw(fee_strategy)
            ))
        })

    if not agents:
        agents = {
            f'Agent{_}': draw(agent_config(
                asset_list=asset_list
            ))
            for _ in range(5)
        }

    return GlobalState(
        pools=pools,
        agents=agents,
        external_market=market_prices
    )


@given(global_state_config())
def test_simulation(initial_state: GlobalState):

    for a, agent in enumerate(initial_state.agents.values()):
        pool: AMM = initial_state.pools[list(initial_state.pools.keys())[a % len(initial_state.pools)]]
        agent.trade_strategy = [
            random_swaps(pool=pool.unique_id, amount={tkn: 1/initial_state.price(tkn) for tkn in pool.asset_list}),
            invest_all(pool=pool.unique_id)
        ][a % 2]

    # VVV -this would break the property test- VVV
    # initial_state.evolve_function = fluctuate_prices()

    initial_wealth = initial_state.total_wealth()
    events = run.run(initial_state, time_steps=10)
    events = processing.postprocessing(events, optional_params=['pool_val', 'holdings_val', 'impermanent_loss'])

    # pu.plot(events, asset='all')
    # pu.plot(events, agent='Trader', prop=['holdings', 'holdings_val'])

    # property test: is there still the same total wealth held by all pools + agents?
    final_state = events[-1]['state']
    if final_state.total_wealth() != pytest.approx(initial_wealth):
        raise AssertionError('total wealth quantity changed!')


@given(global_state_config())
def test_construction(initial_state: GlobalState):
    # see whether we can just construct a valid global state
    # print(initial_state)
    pass


if __name__ == "__main__":
    pass
