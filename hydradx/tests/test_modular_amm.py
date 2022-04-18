import pytest
from hypothesis import given, strategies as st, assume
from hydradx.model.modular_amm.omnipool_amm import *
from hydradx.model.modular_amm.amm import Market
import random


def test_market_construction():
    # noinspection SpellCheckingInspection
    lrna = amm.Asset(name='LRNA', price=0.5)
    hdx = amm.Asset(name='HDX', price=0.08)
    usd = amm.Asset(name='USD', price=1)
    doge = amm.Asset(name='DOGE', price=0.001)
    eth = amm.Asset(name='ETH', price=4000.0)

    omnipool = OmniPool(
        tvl_cap_usd=1000000,
        lrna_fee=0.001,
        asset_fee=0.002,
        preferred_stablecoin='USD'
    ).initializeAssetList([lrna, hdx, usd, doge, eth])

    omnipool.add_lrna_pool(eth, 10)
    omnipool.add_lrna_pool(usd, 100)
    omnipool.add_lrna_pool(hdx, 1000)
    omnipool.add_lrna_pool(doge, 100000)

    agents = [
        OmnipoolAgent(name='LP')
        .add_liquidity(omnipool, doge, 1000)
        .add_liquidity(omnipool, hdx, 1000),
        OmnipoolAgent(name='trader')
        .add_position(usd, 1000)
        .add_position(eth, 1000),
        OmnipoolAgent(name='arbitrager')
        .add_position(usd, 1000)
        .add_position(hdx, 1000)
    ]

    assert omnipool.B('HDX') == pytest.approx(1 / 2 * omnipool.S('HDX'))
    assert agents[2].r('USD') == 1000
    assert agents[0].s(doge) == 1000
    assert omnipool.Q(doge) == omnipool.R(doge) * omnipool.price(doge) / omnipool.price(lrna)


asset_price_strategy = st.floats(min_value=0.0001, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=1, max_value=1000000)


@st.composite
def assets_config(draw, asset_count: int = 0) -> list[amm.Asset]:
    asset_count = asset_count or draw(asset_number_strategy)
    amm.Asset.clear()
    return [
               amm.Asset('LRNA', draw(asset_price_strategy)),
               amm.Asset('HDX', draw(asset_price_strategy)),
               amm.Asset('USD', 1)
           ] + [
               amm.Asset(
                   name=''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)),
                   price=draw(asset_price_strategy))
               for _ in range(asset_count - 3)
           ]


@st.composite
def omnipool_config(draw, asset_list=None, asset_count=0, lrna_fee=None, asset_fee=None, tvl_cap_usd=0):
    asset_list = asset_list or draw(assets_config(asset_count))
    Market.reset()
    omnipool = OmniPool(
        lrna_fee=lrna_fee or draw(st.floats(min_value=0, max_value=0.1)),
        asset_fee=asset_fee or draw(st.floats(min_value=0, max_value=0.1)),
        preferred_stablecoin=asset_list[1],
        tvl_cap_usd=tvl_cap_usd or 1000000,
    ).initializeAssetList(asset_list)
    for asset in asset_list:
        if asset != omnipool.lrna:
            omnipool.add_lrna_pool(asset, draw(asset_quantity_strategy))

    return omnipool


@given(market_state=omnipool_config(asset_count=6),
       buy_index=st.integers(min_value=1, max_value=5),
       sell_index=st.integers(min_value=1, max_value=5),
       delta_r=asset_quantity_strategy)
def test_swap_asset(market_state, buy_index, sell_index, delta_r):
    assume(sell_index < len(market_state.pool_list))
    assume(buy_index < len(market_state.pool_list))
    sell_asset = market_state.pool_list[sell_index].asset
    buy_asset = market_state.pool_list[buy_index].asset
    assume(buy_index != sell_index)
    old_state = market_state
    old_agents = [
        OmnipoolAgent('trader')
        .add_position(buy_asset, 0)
        .add_position(sell_asset, 1000000)
    ]
    new_state, new_agents = swap_assets(old_state, old_agents, sell_asset, buy_asset, trader_id=0,
                                        sell_quantity=delta_r)

    # do some algebraic checks
    i = sell_asset.name
    j = buy_asset.name
    delta_L = new_state.L - old_state.L
    delta_Qj = new_state.Q(j) - old_state.Q(j)
    delta_Qi = new_state.Q(i) - old_state.Q(i)
    delta_QH = new_state.Q('HDX') - old_state.Q('HDX')
    if new_state.Q(i) * new_state.R(i) != pytest.approx(old_state.Q(i) * old_state.R(i)):
        raise ValueError('price change in asset {i}')
    if i != 0 and j != 0:
        if delta_L + delta_Qj + delta_Qi + delta_QH != pytest.approx(0, abs=1e10):
            raise ValueError('Some LRNA was lost along the way.')


@given(market_state=omnipool_config(tvl_cap_usd=5000000),
       pool_index=st.integers(min_value=1, max_value=5),
       quantity=asset_quantity_strategy)
def test_add_liquidity(market_state, pool_index, quantity):
    assume(pool_index < len(market_state.pool_list))
    asset_index = market_state.pool_list[pool_index].name
    old_state = market_state
    old_agents = [
        OmnipoolAgent(name='LP')
        .add_position(old_state.asset(asset_index), 10000000)
    ]
    new_state, new_agents = add_liquidity(
        old_state,
        old_agents,
        agent_index=0,
        asset_name=asset_index,
        delta_r=quantity
    )

    i = asset_index
    if pytest.approx(old_state.R(i) / old_state.S(i)) != new_state.R(i) / new_state.S(i):
        raise ValueError("Incorrect ratio of assets to shares.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i)) != new_state.Q(i) / new_state.R(i):
        raise ValueError("Asset price should not change when liquidity is added.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i) * (old_state.Q_total + old_state.L) / old_state.Q_total) != \
            new_state.Q(i) / new_state.R(i) * (new_state.Q_total + new_state.L) / new_state.Q_total:
        # TODO: understand better what this means.
        raise ValueError("Target price has changed.")


@given(market_state=omnipool_config(tvl_cap_usd=5000000),
       pool_index=st.integers(min_value=1, max_value=5),
       quantity=asset_quantity_strategy)
def test_remove_liquidity(market_state, pool_index, quantity):
    assume(pool_index < len(market_state.pool_list))
    asset_index = market_state.pool_list[pool_index].name
    old_state = market_state
    old_agents = [
        OmnipoolAgent(name='LP')
        .add_liquidity(old_state, asset_index, 1000000)
    ]
    new_state, new_agents = add_liquidity(
        old_state,
        old_agents,
        agent_index=0,
        asset_name=asset_index,
        delta_r=quantity
    )
    i = asset_index
    if pytest.approx(old_state.R(i) / old_state.S(i)) != new_state.R(i) / new_state.S(i):
        raise ValueError("Incorrect ratio of assets to shares.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i)) != new_state.Q(i) / new_state.R(i):
        raise ValueError("Asset price should not change when liquidity is added.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i) * (old_state.Q_total + old_state.L) / old_state.Q_total) != \
            new_state.Q(i) / new_state.R(i) * (new_state.Q_total + new_state.L) / new_state.Q_total:
        raise ValueError("Target price has changed.")


@given(market_state=omnipool_config(asset_count=4))
def test_buy_lrna(market_state: OmniPool):
    asset_name = market_state.pool_list[2].name
    agents = [
        OmnipoolAgent(name='trader')
        .add_position(asset_name='LRNA', quantity=1000000)
    ]
    old_state, old_agents = market_state, agents
    buy_state, buy_agents = sell_lrna(
        market_state=old_state,
        agents_list=agents,
        agent_index=0,
        asset_name=asset_name,
        buy_asset_quantity=1000
    )
    delta_q = buy_state.Q(asset_name) - old_state.Q(asset_name)
    if delta_q > 0:
        sell_state, sell_agents = sell_lrna(
            market_state=old_state,
            agents_list=agents,
            agent_index=0,
            asset_name=asset_name,
            sell_lrna_quantity=-delta_q
        )
        assert sell_agents[0].q == pytest.approx(buy_agents[0].q)
        assert sell_agents[0].r(asset_name) == pytest.approx(buy_agents[0].r(asset_name))
        assert sell_state.Q(asset_name) == pytest.approx(buy_state.Q(asset_name))
        assert sell_state.R(asset_name) == pytest.approx(buy_state.R(asset_name))


if __name__ == "__main__":
    test_market_construction()
    test_add_liquidity()
    test_swap_asset()
    test_buy_lrna()
