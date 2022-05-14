import pytest
from hypothesis import given, strategies as st, assume
from hydradx.model.amm.omnipool_amm import *
from hydradx.model.amm.amm import WorldState
import random


def test_market_construction():
    # noinspection SpellCheckingInspection
    lrna = amm.Asset(name='LRNA', price=0.5)
    hdx = amm.Asset(name='HDX', price=0.08)
    usd = amm.Asset(name='USD', price=1)
    doge = amm.Asset(name='DOGE', price=0.001)
    eth = amm.Asset(name='ETH', price=4000.0)

    omnipool = Omnipool(
        tvl_cap_usd=1000000,
        lrna_fee=0.001,
        asset_fee=0.002,
        preferred_stablecoin='USD'
    )
    omnipool.add_lrna_pool(eth, 10)
    omnipool.add_lrna_pool(usd, 100)
    omnipool.add_lrna_pool(hdx, 1000)
    omnipool.add_lrna_pool(doge, 100000)

    agents = [
        OmnipoolAgent(name='LP')
            .add_liquidity(omnipool, 'DOGE', 1000)
            .add_liquidity(omnipool, 'HDX', 1000),
        OmnipoolAgent(name='trader')
            .add_position('USD', 1000)
            .add_position(eth, 1000),
        OmnipoolAgent(name='arbitrager')
            .add_position('USD', 1000)
            .add_position(hdx, 1000)
    ]

    assert omnipool.B('HDX') == pytest.approx(1 / 2 * omnipool.S('HDX'))
    assert agents[2].r('USD') == 1000
    assert agents[0].s(doge) == 1000
    assert omnipool.Q(doge) == omnipool.R(doge) * doge.price / lrna.price
    assert agents[0].holdings(agents[0].pool_asset('DOGE')) == 1000


asset_price_strategy = st.floats(min_value=0.0001, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=1, max_value=1000000)


@st.composite
def assets_config(draw, asset_count: int = 0) -> list[amm.Asset]:
    asset_count = asset_count or draw(asset_number_strategy)
    Asset.clear()
    return [
               Asset('LRNA', price=draw(asset_price_strategy)),
               Asset('HDX', price=draw(asset_price_strategy)),
               Asset('USD', price=1)
           ] + [
               Asset(
                   name=''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)),
                   price=draw(asset_price_strategy))
               for _ in range(asset_count - 3)
           ]


@st.composite
def omnipool_config(draw, asset_list=None, asset_count=0, lrna_fee=None, asset_fee=None, tvl_cap_usd=0) -> Omnipool:
    asset_list = asset_list or draw(assets_config(asset_count))
    omnipool = Omnipool(
        lrna_fee=lrna_fee or draw(st.floats(min_value=0, max_value=0.1)),
        asset_fee=asset_fee or draw(st.floats(min_value=0, max_value=0.1)),
        preferred_stablecoin='USD',
        tvl_cap_usd=tvl_cap_usd or 1000000,
    )
    for asset in asset_list:
        if asset != omnipool.lrna:
            omnipool.add_lrna_pool(asset, draw(asset_quantity_strategy))

    return omnipool


@given(initial_state=omnipool_config(asset_count=6),
       buy_index=st.integers(min_value=1, max_value=5),
       sell_index=st.integers(min_value=1, max_value=5),
       delta_r=asset_quantity_strategy)
def test_swap_asset(initial_state, buy_index, sell_index, delta_r):
    assume(sell_index < len(initial_state.pool_list))
    assume(buy_index < len(initial_state.pool_list))
    sell_asset = initial_state.pool_list[sell_index].asset
    buy_asset = initial_state.pool_list[buy_index].asset
    assume(buy_index != sell_index)
    old_state = initial_state
    old_agents = [
        OmnipoolAgent('trader')
        .add_position(buy_asset, 0)
        .add_position(sell_asset, 1000000)
    ]
    sell_i_state, sell_i_agents = \
        swap_assets(old_state, old_agents, sell_asset, buy_asset, trader_id=0, delta_r=delta_r)

    # do some algebraic checks
    i = sell_asset.name
    j = buy_asset.name
    delta_L = sell_i_state.L - old_state.L
    delta_Qj = sell_i_state.Q(j) - old_state.Q(j)
    delta_Qi = sell_i_state.Q(i) - old_state.Q(i)
    delta_QH = sell_i_state.Q('HDX') - old_state.Q('HDX')
    delta_Rj = sell_i_state.R(j) - old_state.R(j)
    if sell_i_state.Q(i) * sell_i_state.R(i) != pytest.approx(old_state.Q(i) * old_state.R(i)):
        raise ValueError('price change in asset {i}')
    if i != 0 and j != 0:
        if delta_L + delta_Qj + delta_Qi + delta_QH != pytest.approx(0, abs=1e10):
            raise ValueError('Some LRNA was lost along the way.')
    sell_j_state, sell_j_agents = swap_assets(
        old_state, old_agents, sell_asset, buy_asset, trader_id=0, delta_r=delta_Rj
    )
    assert sell_i_state.R(i) == pytest.approx(sell_j_state.R(i))
    assert sell_i_state.R(j) == pytest.approx(sell_j_state.R(j))
    assert sell_i_state.Q(i) == pytest.approx(sell_j_state.Q(i))
    assert sell_i_state.Q(j) == pytest.approx(sell_j_state.Q(j))


@given(initial_state=omnipool_config(tvl_cap_usd=5000000),
       pool_index=st.integers(min_value=1, max_value=5),
       quantity=asset_quantity_strategy)
def test_add_liquidity(initial_state: Omnipool, pool_index: int, quantity: float):
    assume(pool_index < len(initial_state.pool_list))
    asset = initial_state.pool_list[pool_index].asset
    old_state = initial_state
    old_agents = [
        OmnipoolAgent(name='LP')
        .add_position(asset, 10000000)
    ]
    new_state, new_agents = add_liquidity(
        old_state,
        old_agents,
        agent_index=0,
        asset=asset,
        delta_r=quantity
    )

    i = asset.name
    if pytest.approx(old_state.R(i) / old_state.S(i)) != new_state.R(i) / new_state.S(i):
        raise ValueError("Incorrect ratio of assets to shares.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i)) != new_state.Q(i) / new_state.R(i):
        raise ValueError("Asset price should not change when liquidity is added.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i) * (old_state.Q_total + old_state.L) / old_state.Q_total) != \
            new_state.Q(i) / new_state.R(i) * (new_state.Q_total + new_state.L) / new_state.Q_total:
        # TODO: understand better what this means.
        raise ValueError("Target price has changed.")


@given(initial_state=omnipool_config(tvl_cap_usd=5000000),
       pool_index=st.integers(min_value=1, max_value=5),
       quantity=asset_quantity_strategy)
def test_remove_liquidity(initial_state, pool_index, quantity):
    assume(pool_index < len(initial_state.pool_list))
    asset = initial_state.pool_list[pool_index].asset
    old_state = initial_state
    old_agents = [
        OmnipoolAgent(name='LP')
        .add_liquidity(old_state, asset, 1000000)
    ]
    new_state, new_agents = add_liquidity(
        old_state,
        old_agents,
        agent_index=0,
        asset=asset,
        delta_r=quantity
    )
    i = asset.name
    if pytest.approx(old_state.R(i) / old_state.S(i)) != new_state.R(i) / new_state.S(i):
        raise ValueError("Incorrect ratio of assets to shares.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i)) != new_state.Q(i) / new_state.R(i):
        raise ValueError("Asset price should not change when liquidity is added.")

    elif pytest.approx(old_state.Q(i) / old_state.R(i) * (old_state.Q_total + old_state.L) / old_state.Q_total) != \
            new_state.Q(i) / new_state.R(i) * (new_state.Q_total + new_state.L) / new_state.Q_total:
        raise ValueError("Target price has changed.")


@given(initial_state=omnipool_config(asset_count=4))
def test_swap_lrna(initial_state: Omnipool):
    asset = initial_state.pool_list[2].asset
    agents = [
        OmnipoolAgent(name='trader')
        .add_position(asset='LRNA', quantity=1000000)
        .add_position(asset=asset, quantity=1000)
    ]
    old_state, old_agents = initial_state, agents
    sell_r_state, sell_r_agents = swap_lrna(
        market_state=old_state,
        agents_list=old_agents,
        agent_index=0,
        asset=asset,
        delta_r=1000
    )
    delta_q = sell_r_state.Q(asset) - old_state.Q(asset)
    if delta_q > 0:
        buy_q_state, buy_q_agents = swap_lrna(
            market_state=old_state,
            agents_list=old_agents,
            agent_index=0,
            asset=asset,
            delta_q=-delta_q
        )
        assert sell_r_agents[0].q == pytest.approx(buy_q_agents[0].q)
        assert sell_r_agents[0].r(asset) == pytest.approx(buy_q_agents[0].r(asset))
        assert sell_r_state.Q(asset) == pytest.approx(buy_q_state.Q(asset))
        assert sell_r_state.R(asset) == pytest.approx(buy_q_state.R(asset))

    sell_q_state, sell_q_agents = swap_lrna(
        market_state=old_state,
        agents_list=old_agents,
        agent_index=0,
        asset=asset,
        delta_r=1000
    )
    delta_q = sell_q_state.Q(asset) - old_state.Q(asset)
    if delta_q > 0:
        buy_r_state, buy_r_agents = swap_lrna(
            market_state=old_state,
            agents_list=old_agents,
            agent_index=0,
            asset=asset,
            delta_q=-delta_q
        )
        assert sell_q_agents[0].q == pytest.approx(buy_r_agents[0].q)
        assert sell_q_agents[0].r(asset) == pytest.approx(buy_r_agents[0].r(asset))
        assert sell_q_state.Q(asset) == pytest.approx(buy_r_state.Q(asset))
        assert sell_q_state.R(asset) == pytest.approx(buy_r_state.R(asset))

    # test whether a two-part asset-lrna-asset swap can be equivalent to a direct swap
    half_swap_state, half_swap_agents = swap_lrna(
        market_state=old_state,
        agents_list=old_agents,
        agent_index=0,
        asset=asset,
        delta_r=1
    )
    delta_q = half_swap_state.Q(asset) - old_state.Q(asset)
    if delta_q < 0:
        other_asset = initial_state.pool_list[1].name
        swap_state, swap_agents = swap_lrna(
            market_state=half_swap_state,
            agents_list=half_swap_agents,
            agent_index=0,
            asset=other_asset,
            delta_q=-delta_q
        )
        if repr(swap_state) != repr(half_swap_state):
            # i.e. if the transaction was successful
            direct_swap_state, direct_swap_agents = swap_assets(
                market_state=old_state,
                agents_list=old_agents,
                sell_asset=asset,
                buy_asset=other_asset,
                trader_id=0,
                delta_r=1
            )
            assert pytest.approx(swap_agents[0].value_holdings(swap_state)) == \
                   direct_swap_agents[0].value_holdings(direct_swap_state)
            assert direct_swap_state.T_total == pytest.approx(swap_state.T_total)
            assert direct_swap_state.Q_total == pytest.approx(swap_state.Q_total)
            # print("A test passed.")


@given(initial_state=omnipool_config(asset_count=4))
def test_trade_strategies(initial_state: Omnipool):
    def totalWealth(agent: OmnipoolAgent):
        return sum([
            agent.holdings(asset) * asset.price
            for asset in agent.asset_list
        ])

    trader = OmnipoolAgent(
            name='trader',
            trade_strategy=OmnipoolTradeStrategies.random_swaps(amount=50)
        ) \
        .add_position(asset=initial_state.pool_list[1].asset, quantity=1000) \
        .add_position(asset=initial_state.pool_list[2].asset, quantity=1000) \

    arbitrager = OmnipoolAgent(
            name='arbitrager',
            trade_strategy=OmnipoolTradeStrategies.arbitrage(sell_fee=0.01)
        ) \
        .add_position(asset=initial_state.preferred_stablecoin, quantity=1000000) \
        .add_position(asset=initial_state.pool_list[1].asset, quantity=1000) \
        .add_position(asset=initial_state.pool_list[2].asset, quantity=1000) \

    agents = dict(
        arbitrager=arbitrager,
    )
    arbitrager_starting_wealth = totalWealth(arbitrager)
    initial_state = WorldState(
        initial_state, agents
    )
    initial_assets = initial_state.asset_list
    evolving_state = initial_state

    for _ in range(100):
        evolving_state = evolving_state.copy()
        market = evolving_state.exchange
        agents = evolving_state.agents.values()

        # market prices fluctuate
        for asset in market.pool_asset_list:
            if asset != market.preferred_stablecoin:
                asset.price *= random.random() * 0.001 - 0.0005 + 1

        for agent in agents:
            agent.tradeStrategy.execute(agent, market)

    # reset asset prices
    evolving_state.agents['arbitrager'].asset_list = initial_assets
    arbitrager_ending_wealth = totalWealth(evolving_state.agents['arbitrager'])
    profit = arbitrager_ending_wealth - arbitrager_starting_wealth
    assert profit > 0
    # print(f'starting: {arbitrager_starting_wealth}, '
    #       f'ending: {arbitrager_ending_wealth}, '
    #       f'profit: {profit} ({profit / arbitrager_starting_wealth * 100}%)')

    if profit / arbitrager_starting_wealth > 1:
        print('Abnormally high profit margin...')
        print(repr(initial_state.exchange), repr(evolving_state.exchange))


def test_world_state():
    # Dependencies
    import pandas
    from hydradx.model.amm.amm import WorldState

    # Experiments
    from hydradx.model import run

    assets = [
        Asset(name='LRNA', price=1),
        Asset(name='HDX', price=0.08),
        Asset(name='USD', price=1),
        Asset(name='R1', price=3000),
        Asset(name='R2', price=.001)
    ]
    omnipool = (
        Omnipool(
            tvl_cap_usd=100000000,
            lrna_fee=0,
            asset_fee=0,
            preferred_stablecoin='USD'
        )
            .add_lrna_pool('HDX', 1000000)
            .add_lrna_pool('USD', 1000000)
            .add_lrna_pool('R1', 1000)
            .add_lrna_pool('R2', 1000000000)
    )
    agents = {
        "trader": OmnipoolAgent(
            name='trader',
            trade_strategy=OmnipoolTradeStrategies.random_swaps(amount=1000)
        )

            .add_position('R1', 10000)
            .add_position('R2', 1000000000),

        "LP1": OmnipoolAgent(name='liquidity provider 1')
            .add_liquidity(omnipool, 'R1', 1000),

        "LP2": OmnipoolAgent(name='liquidity provider 2')
            .add_liquidity(omnipool, 'R2', 100010000)
    }

    worldState = WorldState(
        exchange=omnipool,
        agents=agents
    )

    timesteps = 5000
    state = {'WorldState': worldState}
    config_dict = {
        'N': 1,  # number of monte carlo runs
        'T': range(timesteps),  # number of timesteps - 147439 is the length of uniswap_events
        'M': {'timesteps': [timesteps]},  # simulation parameters
    }

    pandas.options.mode.chained_assignment = None  # default='warn'
    pandas.options.display.float_format = '{:.2f}'.format

    run.config(config_dict, state)
    events = run.run()

    from hydradx.model import processing
    rdf, agent_df = processing.postprocessing(events, params_to_include=['withdraw_val', 'hold_val', 'pool_val'])


if __name__ == "__main__":
    test_market_construction()
    test_add_liquidity()
    test_swap_asset()
    # test_swap_lrna()
    test_remove_liquidity()
    test_trade_strategies()
