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


# @given(
#     st.floats(min_value=100000.0, max_value=10000000.0),
#     st.floats(min_value=100000.0, max_value=10000000.0)
# )
# def test_price_manipulation(usd_liquidity, dai_liquidity):
#     omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
#         tokens={
#             'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
#             'WETH': {'liquidity': 2265161, 'LRNA': 2265161},
#             'DAI': {'liquidity': 2254499, 'LRNA': 2254499},
#         },
#         lrna_fee=0.0005,
#         asset_fee=0.0025,
#         preferred_stablecoin='DAI',
#         withdrawal_fee = True,
#         min_withdrawal_fee = 0.0001,
#     )
#
#     agent = Agent(
#         holdings={tkn: omnipool.liquidity[tkn] / 2 for tkn in ['WETH', 'DAI']},
#         trade_strategy=price_manipulation(
#             pool_id='omnipool',
#             asset1='WETH',
#             asset2='DAI'
#         )
#     )
#     agent.holdings['LRNA'] = 0
#
#     state = GlobalState(
#         pools={'omnipool': omnipool},
#         agents={'agent': agent},
#         external_market={tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}
#     )
#
#     start_holdings = copy.deepcopy(agent.holdings)
#     events = run.run(state, 10, silent=True)
#
#     profit = sum(list(events[-1].agents['agent'].holdings.values())) - sum(list(start_holdings.values()))
#     holdings = events[-1].agents['agent'].holdings
#     er = 1
#     if profit > 0:
#         raise AssertionError(f'Profit: {profit}, Holdings: {holdings}')
#     else:
#         print('no profit')


# def test_fuzz_price_manipulation():
#     initial_state: oamm.OmnipoolState = oamm.OmnipoolState(
#         tokens={
#             'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
#             'WETH': {'liquidity': 1000000, 'LRNA': 1000000},
#             'DAI': {'liquidity': 1000000, 'LRNA': 1000000},
#         },
#         lrna_fee=0,
#         asset_fee=0,
#         preferred_stablecoin='DAI'
#     )
#
#     max_profit = (0, 0, 0, 0)
#     for i in range(1000000):
#         agent_holdings_percent = 10
#         # first_trade_factor = random.random()
#         # second_trade_factor = random.random()
#
#         initial_agent = Agent(
#             holdings={'DAI': 10000000000, 'WETH': 10000000000},
#         )
#         initial_agent.holdings['LRNA'] = 0
#         asset1 = 'DAI'
#         asset2 = 'WETH'
#         lp_quantity = initial_state.liquidity["DAI"] * agent_holdings_percent / 100
#         trade_state = initial_state.copy()
#         trade_agent = initial_agent.copy()
#
#         first_trade = float(int(random.random() * 100000000))
#         oamm.execute_swap(
#             state=trade_state,
#             agent=trade_agent,
#             tkn_sell=asset1, tkn_buy=asset2,
#             sell_quantity=first_trade
#         )
#
#         oamm.execute_add_liquidity(
#             state=trade_state,
#             agent=trade_agent,
#             quantity=lp_quantity,
#             tkn_add=asset2
#         )
#         second_trade = float(int(random.random() * 100000000))
#         oamm.execute_swap(
#             state=trade_state,
#             agent=trade_agent,
#             tkn_sell=asset2, tkn_buy=asset1,
#             sell_quantity=second_trade
#         )
#         final_state, final_agent = oamm.execute_remove_liquidity(
#             state=trade_state,
#             agent=trade_agent,
#             quantity=trade_agent.holdings[(initial_state.unique_id, asset2)],
#             tkn_remove=asset2
#         )
#         oamm.execute_swap(
#             state=final_state,
#             agent=final_agent,
#             tkn_sell='LRNA',
#             tkn_buy='DAI',
#             sell_quantity=final_agent.holdings['LRNA']
#         )
#
#         profit = (
#                 oamm.cash_out_omnipool(final_state, final_agent, {tkn: 1 for tkn in initial_state.asset_list})
#                 - oamm.cash_out_omnipool(initial_state, initial_agent, {tkn: 1 for tkn in initial_state.asset_list})
#         )
#         if profit > max_profit[0]:
#             max_profit = (profit, lp_quantity, first_trade, second_trade)
#
#     print(f'agent sells {max_profit[2]} WETH for DAI')
#     print(f'agent LPs {max_profit[1]} DAI.')
#     print(f'agent sells {max_profit[3]} DAI for WETH')
#     print(f'agent withdraws all DAI')
#
#     print(f'agent nets {max_profit[0]}')
#
#
# def test_fuzz_manipulate_withdraw():
#     initial_state: oamm.OmnipoolState = oamm.OmnipoolState(
#         tokens={
#             'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
#             'WETH': {'liquidity': 1000000, 'LRNA': 1000000},
#             'DAI': {'liquidity': 1000000, 'LRNA': 1000000},
#         },
#         lrna_fee=0,
#         asset_fee=0,
#         preferred_stablecoin='DAI'
#     )
#
#     max_profit = (0, 0, 0, 0)
#     for i in range(1000000):
#         agent_holdings_percent = 10
#         # first_trade_factor = random.random()
#         # second_trade_factor = random.random()
#
#         initial_agent = Agent(
#             holdings={'DAI': 10000000000, 'WETH': 10000000000},
#         )
#         initial_agent.holdings['LRNA'] = 0
#         lp_quantity = initial_state.liquidity["DAI"] * agent_holdings_percent / 100
#         LP_state, LP = oamm.execute_add_liquidity(
#             state=initial_state.copy(),
#             agent=initial_agent.copy(),
#             quantity=lp_quantity,
#             tkn_add='DAI'
#         )
#
#         # ^^ that's our initial state right there ^^
#         # trade to manipulate the price
#         first_trade = float(int(random.random() * 100000000))
#         new_state, new_agent = oamm.execute_swap(
#             state=LP_state.copy(),
#             agent=LP.copy(),
#             tkn_sell='WETH',
#             tkn_buy='DAI',
#             sell_quantity=first_trade
#         )
#
#         oamm.execute_remove_liquidity(
#             state=new_state,
#             agent=new_agent,
#             quantity=new_agent.holdings[('omnipool', 'DAI')],
#             tkn_remove='DAI'
#         )
#         second_trade = float(int(random.random() * 100000000))
#         final_state, final_agent = oamm.execute_swap(
#             state=new_state,
#             agent=new_agent,
#             tkn_sell='DAI',
#             tkn_buy='WETH',
#             sell_quantity=second_trade
#         )
#
#         oamm.execute_swap(
#             state=final_state,
#             agent=final_agent,
#             tkn_sell='LRNA',
#             tkn_buy='DAI',
#             sell_quantity=final_agent.holdings['LRNA']
#         )
#
#         profit = (
#                 oamm.cash_out_omnipool(final_state, final_agent, {tkn: 1 for tkn in initial_state.asset_list})
#                 - oamm.cash_out_omnipool(LP_state, LP, {tkn: 1 for tkn in initial_state.asset_list})
#         )
#         if profit > max_profit[0]:
#             max_profit = (profit, lp_quantity, first_trade, second_trade)
#
#     print(f'max profit: agent LPs {max_profit[1]} DAI.')
#     print(f'agent sells {max_profit[2]} WETH for DAI')
#     print(f'agent withdraws all DAI')
#     print(f'agent sells {max_profit[3]} DAI for WETH')
#     print(f'agent nets {max_profit[0]}')


def test_withdraw_manipulation_scenario():
    agent_holdings = {
        'DAI': 1000000000,
        'WETH': 1000000000,
        'LRNA': 0
    }

    tokens = {
        'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
        'WETH': {'liquidity': 1000000, 'LRNA': 1000000},
        'DAI': {'liquidity': 1000000, 'LRNA': 1000000},
    }

    initial_state = oamm.OmnipoolState(
        tokens=tokens,
        lrna_fee=0.0005,
        asset_fee=0.0025,
        preferred_stablecoin='DAI'
    )

    initial_agent = Agent(
        holdings=agent_holdings
    )

    initial_agent = Agent(
        holdings={'DAI': 10000000000, 'WETH': 10000000000},
    )
    initial_agent.holdings['LRNA'] = 0
    lp_quantity = 990000.0
    LP_state, LP = oamm.execute_add_liquidity(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        quantity=lp_quantity,
        tkn_add='DAI'
    )

    # trade to manipulate the price
    first_trade = 5768844.0
    new_state, new_agent = oamm.execute_swap(
        state=LP_state.copy(),
        agent=LP.copy(),
        tkn_sell='WETH',
        tkn_buy='DAI',
        sell_quantity=first_trade
    )

    oamm.execute_remove_liquidity(
        state=new_state,
        agent=new_agent,
        quantity=new_agent.holdings[('omnipool', 'DAI')],
        tkn_remove='DAI'
    )
    second_trade = 1109338.0
    final_state, final_agent = oamm.execute_swap(
        state=new_state,
        agent=new_agent,
        tkn_sell='DAI',
        tkn_buy='WETH',
        sell_quantity=second_trade
    )

    oamm.execute_swap(
        state=final_state,
        agent=final_agent,
        tkn_sell='LRNA',
        tkn_buy='DAI',
        sell_quantity=final_agent.holdings['LRNA']
    )

    profit = (
            oamm.cash_out_omnipool(final_state, final_agent, {tkn: 1 for tkn in initial_state.asset_list})
            - oamm.cash_out_omnipool(LP_state, LP, {tkn: 1 for tkn in initial_state.asset_list})
    )

    er = 1


def test_add_manipulation_scenario():
    agent_holdings = {
        'DAI': 1000000000,
        'WETH': 1000000000,
        'LRNA': 0
    }

    tokens = {
        'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
        'WETH': {'liquidity': 1000000, 'LRNA': 1000000},
        'DAI': {'liquidity': 1000000, 'LRNA': 1000000},
    }

    initial_state = oamm.OmnipoolState(
        tokens=tokens,
        lrna_fee=0,
        asset_fee=0,
        preferred_stablecoin='DAI'
    )

    initial_agent = Agent(
        holdings=agent_holdings
    )

    initial_agent = Agent(
        holdings={'DAI': 10000000000, 'WETH': 10000000000},
    )
    initial_agent.holdings['LRNA'] = 0

    # trade to manipulate the price
    first_trade = 99999123.0
    sell_state, sell_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell='WETH',
        tkn_buy='DAI',
        sell_quantity=first_trade
    )

    lp_quantity = 100000.0
    lp_state, lp_agent = oamm.execute_add_liquidity(
        state=sell_state.copy(),
        agent=sell_agent.copy(),
        quantity=lp_quantity,
        tkn_add='DAI'
    )

    second_trade = 510194.0
    buy_state, buy_agent = oamm.execute_swap(
        state=lp_state,
        agent=lp_agent,
        tkn_sell='DAI',
        tkn_buy='WETH',
        sell_quantity=second_trade
    )

    final_state, final_agent = oamm.execute_remove_liquidity(
        state=buy_state,
        agent=buy_agent,
        quantity=lp_agent.holdings[('omnipool', 'DAI')],
        tkn_remove='DAI'
    )

    oamm.execute_swap(
        state=final_state,
        agent=final_agent,
        tkn_sell='LRNA',
        tkn_buy='DAI',
        sell_quantity=final_agent.holdings['LRNA']
    )

    profit = (
            oamm.cash_out_omnipool(final_state, final_agent, {tkn: 1 for tkn in initial_state.asset_list})
            - oamm.cash_out_omnipool(initial_state, initial_agent, {tkn: 1 for tkn in initial_state.asset_list})
    )

    er = 1


# @settings(max_examples=1000000)
# @settings(max_examples=1)
@given(
    # st.floats(min_value=1000.0, max_value=10000000.0),
    # st.floats(min_value=1000.0, max_value=10000000.0),
    st.floats(min_value=0, max_value=0.01),
    # st.floats(min_value=0.05, max_value=0.05),
    st.floats(min_value=0.000, max_value=0.01),
    st.floats(min_value=4.0, max_value=8.0),
    st.floats(min_value=0.002, max_value=0.02),
    # st.floats(min_value=8.0, max_value=8.0),
)
def test_realistic_withdraw_manipulation(lp_percentage, price_movement, initial_price_DOT, initial_price_HDX):
    # initial_price = 4.446839631289968
    # price_movement = 0.008582536343883918
    # lp_percentage = 0.014641383746118966
    lp_percentage = 0.05
    price_movement = 0.01
    initial_price_HDX = 0.002
    initial_price_DOT = 4
    attack_asset = 'DOT'

    initial_price = initial_price_DOT if attack_asset == 'DOT' else initial_price_HDX

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    buy_quantity = tokens[attack_asset]['liquidity'] - tokens[attack_asset]['liquidity'] / math.sqrt(1 + price_movement)

    initial_state = oamm.OmnipoolState(
        tokens=tokens,
        lrna_fee=0.0005,
        asset_fee=0.0025,
        preferred_stablecoin='DAI',
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    DOT_shares_total = initial_state.shares[attack_asset]
    # lp_percentage = 0.5

    agent_holdings = {
        'DAI': 100000000000,
        'WETH': 10000000000,
        'DOT': 100000000000,
        'WBTC': 10000000000,
        'HDX': 100000000000,
        'LRNA': 0,
        ('omnipool', attack_asset): DOT_shares_total * lp_percentage,
    }

    initial_state.protocol_shares[attack_asset] -= agent_holdings[('omnipool', attack_asset)]

    agent_prices = {
        ('omnipool', attack_asset): initial_price
    }

    agent_delta_r = {
        ('omnipool', attack_asset): 0.0
    }

    initial_agent = Agent(
        holdings=agent_holdings,
        share_prices=agent_prices,
        delta_r=agent_delta_r
    )

    trade_state, trade_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell='DAI',
        tkn_buy=attack_asset,
        # sell_quantity=sell1_quantity
        buy_quantity=buy_quantity
    )

    withdraw_state, withdraw_agent = oamm.execute_remove_liquidity(
        state=trade_state.copy(),
        agent=trade_agent.copy(),
        quantity=trade_agent.holdings[('omnipool', attack_asset)],
        tkn_remove=attack_asset
    )

    sell_lrna_state, sell_lrna_agent = oamm.execute_swap(
        withdraw_state.copy(),
        withdraw_agent.copy(),
        tkn_sell='LRNA',
        tkn_buy='DAI',
        sell_quantity=withdraw_agent.holdings['LRNA']
    )

    final_global_state = GlobalState(
        pools={'omnipool': sell_lrna_state},
        agents={'attacker': sell_lrna_agent},
        external_market=market_prices
    )
    # import math
    # omnipool = sell_lrna_state
    # state = final_global_state
    # asset1 = 'DAI'
    # asset2 = 'DOT'
    # delta_r = (math.sqrt((
    #                  omnipool.lrna[asset2] * omnipool.lrna[asset1] * omnipool.liquidity[asset2] *
    #                  omnipool.liquidity[asset1]
    #          ) / (state.external_market[asset2] / state.external_market[asset1])) - (
    #        omnipool.lrna[asset1] * omnipool.liquidity[asset2]
    # )) / (omnipool.lrna[asset2] + omnipool.lrna[asset1])
    #
    # arbed_pool, arbed_agent = oamm.execute_swap(
    #     state=sell_lrna_state.copy(),
    #     agent=sell_lrna_agent.copy(),
    #     tkn_sell='DOT',
    #     tkn_buy='DAI',
    #     sell_quantity=delta_r
    # )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=final_global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    initial_wealth = oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    no_arb_wealth = oamm.cash_out_omnipool(sell_lrna_state, sell_lrna_agent, market_prices)
    final_wealth = oamm.cash_out_omnipool(arbed_pool, arbed_agent, market_prices)

    no_arb_profit = no_arb_wealth - initial_wealth
    profit = final_wealth - initial_wealth
    if profit > 0:
        raise ValueError('attacker made money')
