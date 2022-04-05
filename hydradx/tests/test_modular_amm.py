from hypothesis.strategies import builds, composite, floats, integers
import pytest
from hypothesis import given, strategies as st, assume
from hydradx.model.modular_amm.omnipool_amm import *
import random


def test_market_construction():
    # noinspection SpellCheckingInspection
    lrna = amm.Asset(name='LRNA', price=1)
    hdx = amm.Asset(name='HDX', price=0.08)
    usd = amm.Asset(name='USD', price=1)
    doge = amm.Asset(name='DOGE', price=0.000001)
    eth = amm.Asset(name='ETH', price=4000.0)

    omnipool = OmniPool(
        lrna=lrna,
        tvl_cap_usd=1000000,
        lrna_fee=0.001,
        asset_fee=0.002,
        preferred_stablecoin=usd
    )
    omnipool.add_pool(hdx, 1000)
    omnipool.add_pool(usd, 1000)
    omnipool.add_pool(doge, 1000)
    omnipool.add_pool(eth, 1000)

    external_market = Market(omnipool.asset_list, price_denomination='USD')

    agents = [
        OmnipoolAgent(name='LP')
        .add_asset(omnipool, doge, 1000)
        .add_asset(omnipool, hdx, 1000)
        ,
        OmnipoolAgent(name='trader')
        .add_asset(external_market, usd, 1000)
        .add_asset(external_market, eth, 1000)
        ,
        OmnipoolAgent(name='arbitrager')
        .add_asset(external_market, usd, 1000)
        .add_asset(external_market, hdx, 1000)
    ]

    assert omnipool.pool(3) == omnipool.pool(doge) == omnipool.pool('DOGE')
    assert omnipool.B('HDX') == pytest.approx(1/2 * omnipool.S('HDX'))
    assert agents[2].r('USD') == 1000
    assert agents[0].s(doge) == 1000
    assert omnipool.Q(2) == omnipool.R(2) * omnipool.price(2)


asset_price_strategy = st.floats(min_value=0.0001, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=6)
asset_quantity_strategy = st.floats(min_value=1, max_value=1000)


@composite
def assets_config(draw) -> list[amm.Asset]:
    nof_assets = draw(asset_number_strategy)
    return [
        amm.Asset('HDX', draw(asset_price_strategy)),
        amm.Asset('USD', 1)
    ] + [
        amm.Asset(
            name=''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)),
            price=draw(asset_price_strategy))
        for n in range(nof_assets - 2)
    ]


@composite
def omnipool_config(draw, assets_list=None, lrna_fee=None, asset_fee=None, initial_liquidity=0, tvl_cap_usd=0):
    assets_list = assets_list or draw(assets_config())
    omnipool = OmniPool(
        lrna=amm.Asset('LRNA', draw(asset_price_strategy)),
        lrna_fee=lrna_fee or draw(st.floats(min_value=0, max_value=0.1)),
        asset_fee=asset_fee or draw(st.floats(min_value=0, max_value=0.1)),
        preferred_stablecoin=assets_list[1],
        initial_liquidity=initial_liquidity or draw(st.floats(min_value=0, max_value=1000000)),
        tvl_cap_usd=tvl_cap_usd or 10000
    )
    for asset in assets_list:
        omnipool.add_pool(asset, draw(asset_quantity_strategy))

    return omnipool


@given(market_state=omnipool_config(),
       buy_index=st.integers(min_value=0, max_value=5),
       sell_index=st.integers(min_value=0, max_value=5),
       delta_r=asset_quantity_strategy)
def test_swap_asset(market_state, buy_index, sell_index, delta_r):
    assume(sell_index < len(market_state.asset_list))
    assume(buy_index < len(market_state.asset_list))
    old_state = market_state
    external_market = amm.Market(market_state.asset_list)
    old_agents = [
        OmnipoolAgent('trader')
        .add_asset(external_market, old_state.asset(buy_index), 0)
        .add_asset(external_market, old_state.asset(sell_index), 1000000)
    ]
    swap_assets(old_state, old_agents, sell_index, buy_index, trader_id=0, sell_quantity=delta_r)


@given(market_state=omnipool_config(tvl_cap_usd=1000000),
       asset_index=st.integers(min_value=0, max_value=5),
       quantity=asset_quantity_strategy)
def test_add_liquidity(market_state, asset_index, quantity):
    external_market = amm.Market(market_state.asset_list)
    old_state = market_state
    old_agents = [
        OmnipoolAgent('trader')
        .add_asset(external_market, old_state.asset(asset_index), 10000000)
    ]
    add_liquidity(old_state, old_agents, agent_index=0, asset_index=asset_index, delta_r=quantity)
