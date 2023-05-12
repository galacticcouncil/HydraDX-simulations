import copy
import math

import pytest
from hypothesis import given, strategies as st  # , settings

# from hydradx.model import run
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth, invest_all
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, reasonable_market

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
arb_precision_strategy = st.integers(min_value=1, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
percentage_of_liquidity_strategy = st.floats(min_value=0.0000001, max_value=0.10)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


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
    omnipool.trade_limit_per_block = float('inf')
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
    initial_agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage)
    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': initial_agent})

    new_state = invest_all('omnipool').execute(initial_state, 'agent')
    for tkn in omnipool.asset_list:
        if new_state.agents['agent'].holdings[tkn] != 0:
            raise AssertionError(f'Failed to LP {tkn}')
        if new_state.agents['agent'].holdings[('omnipool', tkn)] == 0:
            raise AssertionError(f'Did not receive shares for {tkn}')

    hdx_state = invest_all('omnipool', 'HDX').execute(initial_state.copy(), 'agent')

    if hdx_state.agents['agent'].holdings['HDX'] != 0:
        raise AssertionError('HDX not reinvested.')
    if hdx_state.agents['agent'].holdings[('omnipool', 'HDX')] == 0:
        raise AssertionError('HDX shares not received.')

    for tkn in omnipool.asset_list:
        if tkn == 'HDX':
            continue
        if hdx_state.agents['agent'].holdings[tkn] == 0:
            raise
        if ('omnipool', tkn) in hdx_state.agents['agent'].holdings \
                and hdx_state.agents['agent'].holdings[('omnipool', tkn)] != 0:
            raise
