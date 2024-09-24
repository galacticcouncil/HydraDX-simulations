import copy
import math

import pytest
from hypothesis import given, strategies as st  # , settings

# from hydradx.model import run
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth, invest_all, dca_with_lping
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, reasonable_market
from hydradx.model.run import run
from hydradx.tests.utils import randomize_object
from mpmath import mp, mpf
mp.dps = 50

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
        if new_value != pytest.approx(old_value, rel=1e-12):
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
        raise AssertionError(f'Arbbtrageur traded LRNA. old: {old_holdings["LRNA"]}, new: {new_holdings["LRNA"]}')

    for asset in omnipool.asset_list:
        old_value += old_holdings[asset] * external_market[asset]
        new_value += new_holdings[asset] * external_market[asset]

        # Trading should bring pool to market price
        # if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
        #     raise

    # Trading should be profitable
    if old_value > new_value:
        if new_value != pytest.approx(old_value, rel=1e-15):
            raise AssertionError(f'Arbitrageur lost money. old_value: {old_value}, new_value: {new_value}')


@given(omnipool_reasonable_config(token_count=3))
def test_omnipool_LP(omnipool: oamm.OmnipoolState):
    holdings = {asset: 10000 for asset in omnipool.asset_list}
    initial_agent = Agent(holdings=holdings, trade_strategy=invest_all('omnipool', when=4))
    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': initial_agent})

    new_state = invest_all('omnipool').execute(initial_state.copy(), 'agent')
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
            raise AssertionError(f'{tkn} missing from holdings.')
        if ('omnipool', tkn) in hdx_state.agents['agent'].holdings \
                and hdx_state.agents['agent'].holdings[('omnipool', tkn)] != 0:
            raise AssertionError(f'Agent has shares of {tkn}, but should not.')

    events = run(initial_state, time_steps=4, silent=True)
    final_state = events[-1]
    if final_state.agents['agent'].holdings['HDX'] != 0 or events[-2].agents['agent'].holdings['HDX'] == 0:
        raise AssertionError('HDX not invested at the right time.')


def test_agent_copy():
    init_agent = randomize_object(
        Agent(holdings={'HDX': 100, 'USD': 100}, share_prices={'HDX': 1, 'USD': 1})
    )
    copy_agent = init_agent.copy()

    for member in copy_agent.__dict__:
        if (
                getattr(init_agent, member) != getattr(copy_agent, member)
        ):
            raise AssertionError(f'Copy failed for {member}.\n'
                                 f'original: {getattr(init_agent, member)}\n'
                                 f'copy: {getattr(copy_agent, member)}')


@given(
    omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=5),
    st.lists(st.floats(min_value=0.0001, max_value=0.5), min_size=2, max_size=2),
    st.lists(st.floats(min_value=0.9, max_value=1.1), min_size=2, max_size=2),
    st.floats(min_value=0.00001, max_value=1.1),
    st.lists(st.floats(min_value=10000, max_value=100000), min_size=2, max_size=2),
    st.lists(st.booleans(), min_size=2, max_size=2)
)
def test_dca_with_lping(
        omnipool: oamm.OmnipoolState,
        init_lp_pcts: list[float],
        price_mults: list[float],
        shares_mult: float,
        assets: list[float],
        has_assets: list[bool]
):
    buy_tkn = omnipool.asset_list[3]
    sell_tkn = omnipool.asset_list[4]
    trader_id = 'trader'

    for tkn in omnipool.liquidity:
        omnipool.liquidity[tkn] = mpf(omnipool.liquidity[tkn])
        omnipool.lrna[tkn] = mpf(omnipool.lrna[tkn])

    holdings = {
        ('omnipool', sell_tkn): mpf(init_lp_pcts[0] * omnipool.shares[sell_tkn]),
        ('omnipool', buy_tkn): mpf(init_lp_pcts[1] * omnipool.shares[buy_tkn]),
        sell_tkn: mpf(assets[0]) if has_assets[0] else 0,
        buy_tkn: mpf(assets[1]) if has_assets[1] else 0
    }
    max_shares_per_block = holdings[('omnipool', sell_tkn)] * shares_mult
    share_prices = {
        ('omnipool', sell_tkn): omnipool.lrna_price(omnipool, sell_tkn) * price_mults[0],
        ('omnipool', buy_tkn): omnipool.lrna_price(omnipool, buy_tkn) * price_mults[1]
    }

    agent = Agent(holdings=holdings, unique_id=trader_id, share_prices=share_prices)

    state = GlobalState(pools={'omnipool': omnipool}, agents={trader_id: agent})

    strategy = dca_with_lping('omnipool', sell_tkn, buy_tkn, max_shares_per_block)

    init_buy_tkn_lped = agent.holdings[('omnipool', buy_tkn)] if ('omnipool', buy_tkn) in agent.holdings else 0
    init_sell_tkn_lped = agent.holdings[('omnipool', sell_tkn)] if ('omnipool', sell_tkn) in agent.holdings else 0
    init_buy_tkn = agent.holdings[buy_tkn]
    init_sell_tkn = agent.holdings[sell_tkn]

    strategy.execute(state, trader_id)

    if init_sell_tkn_lped > 0:
        if len(agent.nfts) == 0:
            raise AssertionError('Agent does not have shares for buy_tkn.')

        lp_diff = init_sell_tkn_lped
        if ('omnipool', sell_tkn) in agent.holdings:
            lp_diff -= agent.holdings[('omnipool', sell_tkn)]

        if lp_diff != pytest.approx(min(max_shares_per_block, init_sell_tkn_lped), rel=1e-20):
            raise AssertionError('Agent traded incorrect amount of shares.')
    elif (agent.holdings[('omnipool', sell_tkn)] != init_sell_tkn_lped or
            agent.holdings[('omnipool', buy_tkn)] != init_buy_tkn_lped):
        raise AssertionError('Agent LP shares changed.')
    if agent.holdings[sell_tkn] != pytest.approx(init_sell_tkn, rel=1e-20):
        raise AssertionError('Agent sell token changed.')
    if agent.holdings[buy_tkn] != pytest.approx(init_buy_tkn, rel=1e-20):
        raise AssertionError('Agent buy token changed.')
