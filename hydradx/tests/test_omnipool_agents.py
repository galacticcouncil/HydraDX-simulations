import copy

import pytest
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
import random


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


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), reasonable_market(token_count=3))
def test_omnipool_arbitrager_feeless(omnipool: oamm.OmnipoolState, market: list):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=omnipool_arbitrage)
    external_market = {omnipool.asset_list[i]: market[i] for i in range(len(omnipool.asset_list))}
    external_market[omnipool.stablecoin] = 1.0
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)
    strat = omnipool_arbitrage('omnipool')

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
        if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
            raise

    # Trading should be profitable
    if old_value > new_value:
        if new_value != pytest.approx(old_value, rel=1e-15):
            raise


@given(omnipool_reasonable_config(token_count=3), reasonable_market(token_count=3), arb_precision_strategy)
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
        # if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
        #     raise

    # Trading should be profitable
    if old_value > new_value:
        if new_value != pytest.approx(old_value, rel=1e-15):
            raise
