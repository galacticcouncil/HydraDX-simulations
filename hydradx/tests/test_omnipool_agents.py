import copy
import random

import pytest
from hypothesis import given, strategies as st

from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth, invest_all, extra_trade_volume, \
    price_sensitive_trading

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
arb_precision_strategy = st.integers(min_value=1, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
percentage_of_liquidity_strategy = st.floats(min_value=0.0000001, max_value=0.10)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def reasonable_market(draw, token_count: int = 0) -> list:
    token_count = token_count or draw(asset_number_strategy)
    return [draw(asset_price_bounded_strategy) for _ in range(token_count)]


@st.composite
def assets_reasonable_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_bounded_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA_price': usd_price_lrna
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        } for _ in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_reasonable_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        imbalance=None,
) -> oamm.OmnipoolState:
    asset_dict: dict = asset_dict or draw(assets_reasonable_config(token_count))

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
    )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy) if imbalance is None else imbalance
    test_state.update()
    return test_state


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), percentage_of_liquidity_strategy)
def test_back_and_forth_trader_feeless(omnipool: oamm.OmnipoolState, pct: float):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=back_and_forth)
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat = back_and_forth('omnipool', pct)

    old_agent = copy.deepcopy(agent)

    strat.execute(state, 'agent')
    new_agent = state.agents['agent']
    assert new_agent.holdings['LRNA'] == old_agent.holdings['LRNA']  # LRNA holdings should be *exact*
    for asset in omnipool.asset_list:
        assert new_agent.holdings[asset] == pytest.approx(old_agent.holdings[asset], rel=1e-15)


@given(omnipool_reasonable_config(token_count=3), percentage_of_liquidity_strategy)
def test_back_and_forth_trader(omnipool: oamm.OmnipoolState, pct: float):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=back_and_forth)
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat = back_and_forth('omnipool', pct)

    old_agent = copy.deepcopy(agent)

    strat.execute(state, 'agent')
    new_agent = state.agents['agent']
    if new_agent.holdings['LRNA'] != old_agent.holdings['LRNA']:
        raise
    for asset in omnipool.asset_list:
        if new_agent.holdings[asset] > old_agent.holdings[asset]:
            if new_agent.holdings[asset] != pytest.approx(old_agent.holdings[asset], rel=1e-15):
                raise


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), reasonable_market(token_count=3),
       arb_precision_strategy)
def test_omnipool_arbitrager_feeless(omnipool: oamm.OmnipoolState, market: list, arb_precision: int):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage)
    external_market = {omnipool.asset_list[i]: market[i] for i in range(len(omnipool.asset_list))}
    external_market[omnipool.stablecoin] = 1.0
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)
    strat = omnipool_arbitrage('omnipool', arb_precision)

    old_holdings = copy.deepcopy(agent.holdings)

    strat.execute(state, 'agent')
    new_holdings = state.agents['agent'].holdings

    old_value, new_value = 0, 0

    # Trading should result in net zero LRNA trades
    if new_holdings['LRNA'] != pytest.approx(old_holdings['LRNA'], rel=1e-15):
        raise

    for asset in omnipool.asset_list:
        old_value += old_holdings[asset] * external_market[asset]
        new_value += new_holdings[asset] * external_market[asset]

        # Trading should bring pool to market price
        if oamm.usd_price(omnipool, asset) != pytest.approx(external_market[asset], rel=1e-15):
            raise

    # Trading should be profitable
    if old_value > new_value:
        if new_value != pytest.approx(old_value, rel=1e-15):
            raise


@given(omnipool_reasonable_config(token_count=3), reasonable_market(token_count=3), arb_precision_strategy)
def test_omnipool_arbitrager(omnipool: oamm.OmnipoolState, market: list, arb_precision: int):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage)
    external_market = {omnipool.asset_list[i]: market[i] for i in range(len(omnipool.asset_list))}
    external_market[omnipool.stablecoin] = 1.0
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)
    strat = omnipool_arbitrage('omnipool', arb_precision)

    old_holdings = copy.deepcopy(agent.holdings)

    strat.execute(state, 'agent')
    new_holdings = state.agents['agent'].holdings

    old_value, new_value = 0, 0

    # Trading should result in net zero LRNA trades
    if new_holdings['LRNA'] != pytest.approx(old_holdings['LRNA'], rel=1e-15):
        raise

    for asset in omnipool.asset_list:
        old_value += old_holdings[asset] * external_market[asset]
        new_value += new_holdings[asset] * external_market[asset]

        # Trading should bring pool to market price
        # if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
        #     raise

    # Trading should be profitable
    if old_value > new_value:
        if new_value != pytest.approx(old_value, rel=1e-15):
            raise


@given(omnipool_reasonable_config(token_count=3))
def test_omnipool_LP(omnipool: oamm.OmnipoolState):
    holdings = {asset: 10000 for asset in omnipool.asset_list}
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage)
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat = invest_all('omnipool')

    new_state = strat.execute(state, 'agent')
    for asset in omnipool.asset_list:
        if new_state.agents['agent'].holdings[asset] != 0:
            raise
        if new_state.agents['agent'].holdings[('omnipool', asset)] == 0:
            raise


@given(omnipool_reasonable_config(token_count=3))
def test_extra_trade_volume(omnipool: oamm.OmnipoolState):
    holdings = {asset: 100000000 for asset in omnipool.asset_list + ['LRNA']}
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage('omnipool', arb_precision=2))
    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat_1 = extra_trade_volume('omnipool', 1.0)
    strat_2 = extra_trade_volume('omnipool', 2.0)

    initial_state.external_market = {asset: 1.0 for asset in omnipool.asset_list}
    new_state_0 = agent.trade_strategy.execute(initial_state.copy(), 'agent')

    new_state_1 = strat_1.execute(new_state_0.copy(), 'agent')
    new_state_2 = strat_2.execute(new_state_0.copy(), 'agent')

    volume_0 = sum([
        (new_state_0.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_0.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    volume_1 = sum([
        (new_state_1.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_1.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    volume_2 = sum([
        (new_state_2.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_2.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    if volume_0 >= volume_1:
        raise AssertionError('Volume should increase with extra trade volume')
    if volume_1 >= volume_2:
        raise AssertionError('Volume should increase with extra trade volume')


@given(omnipool_reasonable_config(token_count=3, asset_fee=0))
def test_price_sensitive_trading(omnipool: oamm.OmnipoolState):
    holdings = {asset: 100000000 for asset in omnipool.asset_list + ['LRNA']}
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage('omnipool', arb_precision=2))
    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    initial_state.external_market = {asset: oamm.usd_price(omnipool, asset) for asset in omnipool.asset_list}

    tkn_sell = omnipool.asset_list[0]
    tkn_buy = omnipool.asset_list[1]
    strat_1 = price_sensitive_trading(
        'omnipool', max_volume_usd=100, price_sensitivity=0.0, tkn_sell=tkn_sell, tkn_buy=tkn_buy
    )
    strat_2 = price_sensitive_trading(
        'omnipool', max_volume_usd=100, price_sensitivity=1.0, tkn_sell=tkn_sell, tkn_buy=tkn_buy
    )
    new_state_1 = strat_1.execute(initial_state.copy(), 'agent')
    new_state_2 = strat_2.execute(initial_state.copy(), 'agent')

    volume_1 = sum([
        (new_state_1.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_1.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    volume_2 = sum([
        (new_state_2.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_2.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    if volume_1 <= volume_2:
        raise AssertionError('Volume should decrease with higher price sensitivity.')

    new_state_3 = initial_state.copy()
    for tkn in omnipool.asset_list:
        new_state_3.pools['omnipool'].liquidity[tkn] /= 2
        new_state_2.pools['omnipool'].lrna[tkn] /= 2
    strat_2.execute(new_state_3, 'agent')

    volume_3 = sum([
        (new_state_3.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_3.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    if volume_3 >= volume_2:
        raise AssertionError('Volume should decrease with lower liquidity.')

    new_state_4 = initial_state.copy()
    new_state_4.pools['omnipool'].asset_fee = {tkn: 0.01 for tkn in omnipool.asset_list}
    strat_2.execute(new_state_4, 'agent')

    volume_4 = sum([
        (new_state_4.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_4.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    if volume_4 >= volume_2:
        raise AssertionError('Volume should decrease with higher fees.')

    new_state_5 = initial_state.copy()
    new_state_5.pools['omnipool'].liquidity[tkn_buy] *= 2
    strat_2.execute(new_state_5, 'agent')

    volume_5 = sum([
        (new_state_5.pools['omnipool'].current_block.volume_in[tkn]
         + new_state_5.pools['omnipool'].current_block.volume_out[tkn]) * oamm.lrna_price(omnipool, tkn)
        for tkn in omnipool.asset_list
    ])

    if volume_5 <= volume_2:
        raise AssertionError('Volume should increase with favorable price.')
