import copy
import random

import math
import pytest
from hypothesis import given, strategies as st, settings

from hydradx.model import run
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth, invest_all, price_manipulation, \
    price_manipulation_multiple_blocks, manipulate_and_withdraw
from hydradx.model.processing import pool_val
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, reasonable_market


def test_fuzz_price_manipulation():
    initial_state: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 44000000, 'LRNA': 275143},
            'WETH': {'liquidity': 1400, 'LRNA': 2276599},
            'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
            'DOT': {'liquidity': 88000, 'LRNA': 546461},
            'WBTC': {'liquidity': 47, 'LRNA': 1145210},
        },
        lrna_fee=0,
        asset_fee=0,
        max_withdrawal_per_block=0.05,
        max_lp_per_block=0.05,
        remove_liquidity_volatility_threshold=0.01,
        oracles={'price': 1},
        preferred_stablecoin='DAI'
    )

    max_profit = (0, 0, 0, 0)
    asset1 = 'WETH'
    asset2 = 'DAI'
    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    initial_agent = Agent(
        holdings={
            'HDX': 440000000,
            'WETH': 140000,
            'DAI': 226826200,
            'DOT': 880000,
            'WBTC': 470,
        },
        trade_strategy=omnipool_arbitrage(pool_id='omnipool')
    )

    price_move = 0.01

    for i in range(100000):
        # trade to manipulate the price
        # first_trade = 13906.0
        sell_state, sell_agent = oamm.execute_swap(
            state=initial_state.copy(),
            agent=initial_agent.copy(),
            tkn_sell=asset1,
            tkn_buy=asset2,
            buy_quantity=random.randrange(
                1, int(initial_state.liquidity[asset2] * (1 / math.sqrt(1 - price_move) - 1))
            )
        )
        buy_quantity = sell_agent.holdings[asset2] - initial_agent.holdings[asset2]

        lp_quantity_1 = random.randrange(
            1, int(initial_state.liquidity[asset1] * min(1.0, initial_state.max_lp_per_block))
        )
        lp_state, lp_agent = oamm.execute_add_liquidity(
            state=sell_state.copy(),
            agent=sell_agent.copy(),
            quantity=lp_quantity_1,
            tkn_add=asset1
        )

        lp_quantity_2 = random.randrange(
            1, int(initial_state.liquidity[asset2] * min(1.0, initial_state.max_lp_per_block))
        )
        lp_state, lp_agent = oamm.execute_add_liquidity(
            state=lp_state,
            agent=lp_agent,
            quantity=lp_quantity_2,
            tkn_add=asset2
        )

        # close arb
        glob = lp_agent.trade_strategy.execute(
            state=GlobalState(
                agents={
                    'agent': lp_agent.copy(),
                },
                pools={
                    'omnipool': lp_state.copy(),
                },
                external_market=market_prices
            ), agent_id='agent'
        )

        final_state: oamm.OmnipoolState = glob.pools['omnipool']
        final_agent: Agent = glob.agents['agent']

        final_diff = {
            tkn: final_agent.holdings[tkn] - (initial_agent.holdings[tkn] if tkn in initial_agent.holdings else 0) for
            tkn in final_agent.holdings}

        profit = (
                oamm.cash_out_omnipool(final_state, final_agent, market_prices)
                - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
        )

        if profit > max_profit[0]:
            max_profit = (profit, buy_quantity, lp_quantity_1, lp_quantity_2)

    print()
    print(f'# agent buys {max_profit[1]} {asset2} with {asset1}')
    print(f'# agent LPs {max_profit[2]} {asset1}.')
    print(f'# agent LPs {max_profit[3]} {asset2}.')
    print(f'# agent arbitrages all assets and withdraws all liquidity.')
    print(f'# agent nets {max_profit[0]}')


@given(
    st.floats(min_value=100000.0, max_value=10000000.0),
    st.floats(min_value=100000.0, max_value=10000000.0)
)
def test_price_manipulation(usd_liquidity, dai_liquidity):
    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
            'WETH': {'liquidity': 2265161, 'LRNA': 2265161},
            'DAI': {'liquidity': 2254499, 'LRNA': 2254499},
        },
        lrna_fee=0,
        asset_fee=0,
        preferred_stablecoin='DAI',
        # remove_liquidity_volatility_threshold=0  # test will fail if this is uncommented
    )

    initial_agent = Agent(
        holdings={tkn: omnipool.liquidity[tkn] / 2 for tkn in ['WETH', 'DAI']},
        trade_strategy=price_manipulation(
            pool_id='omnipool',
            asset1='WETH',
            asset2='DAI'
        )
    )
    initial_agent.holdings['LRNA'] = 0

    state = GlobalState(
        pools={'omnipool': omnipool},
        agents={'agent': initial_agent},
        external_market={tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}
    )

    start_holdings = copy.deepcopy(initial_agent.holdings)
    events = run.run(state, 10, silent=True)

    profit = (
        oamm.cash_out_omnipool(events[-1].pools['omnipool'], events[-1].agents['agent'], state.external_market)
        - oamm.cash_out_omnipool(omnipool, initial_agent, state.external_market)
    )

    holdings = events[-1].agents['agent'].holdings
    er = 1
    if profit > 0:
        raise AssertionError(f'Profit: {profit}, Holdings: {holdings}')


@given(
    omnipool_reasonable_config(),
    st.floats(min_value=0.01, max_value=2),
    st.integers(min_value=1, max_value=100)
)
def test_fuzz_manipulate_withdraw(initial_state: oamm.OmnipoolState, price_ratio, token_index):
    # initial_state.: oamm.OmnipoolState = oamm.OmnipoolState(
    # tokens={
    #     'HDX': {'liquidity': 44000000, 'LRNA': 275143},
    #     'WETH': {'liquidity': 1400, 'LRNA': 2276599},
    #     'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
    #     'DOT': {'liquidity': 88000, 'LRNA': 546461},
    #     'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    #     # 'USD': {'liquidity': 1, 'LRNA': 1}
    # },
    initial_state.lrna_fee = 0
    initial_state.asset_fee = 0
    initial_state.preferred_stablecoin = 'DAI'
    initial_state.trade_limit_per_block = 0.05
    initial_state.remove_liquidity_volatility_threshold = 0.01
    initial_state.max_withdrawal_per_block = 0.05

    max_profit = (0, 0, 0, 0, 0)
    for i in range(10):
        lp_percentage = 50  # random.random() * 100

        tkn_buy_index = (token_index + i) % len(initial_state.asset_list)
        tkn_sell_index = (token_index + i) * 2 % len(initial_state.asset_list)
        if tkn_sell_index == tkn_buy_index:
            tkn_sell_index = (tkn_sell_index + 1) % len(initial_state.asset_list)
        tkn_sell = initial_state.asset_list[tkn_sell_index]
        tkn_buy = initial_state.asset_list[tkn_buy_index]

        shares_total = initial_state.shares[tkn_buy]
        lp_quantity = shares_total * lp_percentage / 100

        agent_holdings = {
            tkn: 1000000 / oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list
        }
        agent_holdings[('omnipool', tkn_buy)] = lp_quantity

        initial_state.protocol_shares[tkn_buy] -= agent_holdings[('omnipool', tkn_buy)]
        initial_price = oamm.lrna_price(initial_state, tkn_buy) * price_ratio

        agent_prices = {
            ('omnipool', tkn_buy): initial_price
        }

        initial_agent = Agent(
            holdings=agent_holdings,
            share_prices=agent_prices,
        )

        # trade to manipulate the price
        first_sell = initial_state.liquidity[tkn_buy] * min(
            initial_state.trade_limit_per_block,
            initial_state.max_withdrawal_per_block,
            initial_state.remove_liquidity_volatility_threshold
        )

        new_state, new_agent = oamm.execute_swap(
            state=initial_state.copy(),
            agent=initial_agent.copy(),
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=first_sell
        )
        first_buy = new_agent.holdings[tkn_buy] - new_agent.holdings[tkn_buy]

        oamm.execute_remove_liquidity(
            state=new_state,
            agent=new_agent,
            quantity=new_agent.holdings[('omnipool', tkn_buy)],
            tkn_remove=tkn_buy
        )

        tkn_buy, tkn_sell = tkn_sell, tkn_buy
        market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}
        second_sell = (math.sqrt((
            new_state.lrna[tkn_sell] * new_state.lrna[tkn_buy]
            * new_state.liquidity[tkn_sell] * new_state.liquidity[tkn_buy]
        ) / (market_prices[tkn_buy] / market_prices[tkn_sell])) - (
            new_state.lrna[tkn_buy] * new_state.liquidity[tkn_sell]
        )) / (new_state.lrna[tkn_sell] + new_state.lrna[tkn_buy])

        final_state, final_agent = oamm.execute_swap(
            state=new_state,
            agent=new_agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=second_sell
        )
        second_buy = final_agent.holdings[tkn_buy] - new_agent.holdings[tkn_buy]

        profit = (
                oamm.cash_out_omnipool(final_state, final_agent, market_prices)
                - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
        )

        if profit > max_profit[0]:
            max_profit = (profit, lp_quantity, initial_price, first_buy, second_buy, tkn_buy, tkn_sell)

    if max_profit[0] > 5:
        attack_asset = max_profit[6]
        trade_asset = max_profit[5]
        print(f'max profit:')
        print(f'agent LPs {max_profit[1]} {attack_asset} ({max_profit[1] / initial_state.liquidity[attack_asset] * 100}'
              f'%) at {max_profit[2]} USD spot price.')
        print(f'agent buys {max_profit[3]} {attack_asset} with {trade_asset}')
        print(f'agent withdraws all {attack_asset}')
        print(f'agent buys {max_profit[4]} {trade_asset} with {attack_asset}')
        print(f'agent nets {max_profit[0]}')
        print(f'initial omnipool state: {initial_state}')


# @settings(max_examples=10000)
def test_withdraw_manipulation_scenario():

    tokens = {
        "HDX": {'liquidity': 4933171.633861665, 'LRNA': 4244267.0263093775},
        "USD": {'liquidity': 3046151.331137664, 'LRNA': 2256201.600780461},
        "myn": {'liquidity': 7702657.221815255, 'LRNA': 5847872.44683839}
    }

    initial_state = oamm.OmnipoolState(
        tokens=tokens,
        lrna_fee=0,
        asset_fee=0,
        preferred_stablecoin='USD',
        max_withdrawal_per_block=0.05,
        remove_liquidity_volatility_threshold=0.01,
        max_lp_per_block=0.05,
        trade_limit_per_block=0.05
    )

    agent_holdings = {
        tkn: 1000000 / oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list
    }

    initial_agent = Agent(
        holdings=agent_holdings
    )

    lp_percent = 0.50
    lp_token = 'myn'
    trade_token = 'USD'
    lp_quantity = int(initial_state.liquidity[lp_token] * lp_percent)

    initial_agent.holdings[('omnipool', lp_token)] = lp_quantity
    initial_agent.share_prices[('omnipool', lp_token)] = 0.037835736649170025

    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    # trade to manipulate the price
    first_trade = 77026.57221815255
    trade_state, trade_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell=trade_token,
        tkn_buy=lp_token,
        sell_quantity=first_trade
    )

    withdraw_state, withdraw_agent = oamm.execute_remove_liquidity(
        state=trade_state.copy(),
        agent=trade_agent.copy(),
        quantity=trade_agent.holdings[('omnipool', lp_token)],
        tkn_remove=lp_token
    )

    # second_trade = 250000.0
    tkn_buy = trade_token
    tkn_sell = lp_token
    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}
    second_trade = (math.sqrt((
        withdraw_state.lrna[tkn_sell] * withdraw_state.lrna[tkn_buy]
        * withdraw_state.liquidity[tkn_sell] * withdraw_state.liquidity[tkn_buy]
    ) / (market_prices[tkn_buy] / market_prices[tkn_sell])) - (
        withdraw_state.lrna[tkn_buy] * withdraw_state.liquidity[tkn_sell]
    )) / (withdraw_state.lrna[tkn_sell] + withdraw_state.lrna[tkn_buy])

    final_state, final_agent = oamm.execute_swap(
        state=withdraw_state.copy(),
        agent=withdraw_agent.copy(),
        tkn_sell=lp_token,
        tkn_buy=trade_token,
        sell_quantity=second_trade
    )

    profit = (
            oamm.cash_out_omnipool(final_state, final_agent, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    print(f'profit analysis:')
    print(f'agent LPs {lp_quantity} {lp_token} ({lp_percent * 100}%) '
          f'at {initial_agent.share_prices["omnipool", lp_token]} USD spot price.')
    print(f'agent sells {first_trade} {trade_token} for {lp_token}')
    print(f'agent withdraws all {lp_token}')
    print(f'agent sells {second_trade} {lp_token} for {trade_token}')
    print(f'agent nets {profit}')

    if profit > 0:
        raise AssertionError(f'profit should be negative, but is {profit}')


def test_double_add_manipulation_scenario():
    agent_holdings = {
        'HDX': 4400000000,
        'WETH': 140000,
        'DAI': 226826200,
        'DOT': 8800000,
        'WBTC': 4700,
    }

    omnipool_assets = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    initial_state = oamm.OmnipoolState(
        tokens=omnipool_assets,
        lrna_fee=0,
        asset_fee=0,
        preferred_stablecoin='DAI',
        max_withdrawal_per_block=0.05,
        max_lp_per_block=0.05,
        remove_liquidity_volatility_threshold=0.01,
        # oracles={'price': 1},
    )

    initial_agent = Agent(
        holdings=agent_holdings
    )

    # agent buys 10929 DAI with WETH
    # agent LPs 37 WETH.
    # agent LPs 113402 DAI.
    # agent arbitrages all assets and withdraws all liquidity.
    # agent nets 486.6858757138252

    asset1 = 'WETH'
    asset2 = 'DAI'
    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}
    # trade to manipulate the price
    # first_trade = 13906.0
    sell_state, sell_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell=asset1,
        tkn_buy=asset2,
        buy_quantity=10929  # 1081304.128411442
    )
    asset2_bought = sell_agent.holdings[asset2] - initial_agent.holdings[asset2]
    first_diff = {tkn: sell_agent.holdings[tkn] - (initial_agent.holdings[tkn] if tkn in initial_agent.holdings else 0)
                  for tkn in sell_agent.holdings}
    actual_price_move = (
        1 - oamm.lrna_price(initial_state, 'DAI') / oamm.lrna_price(sell_state, 'DAI')
    )

    lp_quantity = 37  # 1111.0
    lp_state, lp_agent = oamm.execute_add_liquidity(
        state=sell_state.copy(),
        agent=sell_agent.copy(),
        quantity=lp_quantity,
        tkn_add=asset1
    )

    lp_quantity = 113402  # 1134131.0
    lp_state, lp_agent = oamm.execute_add_liquidity(
        state=lp_state,
        agent=lp_agent,
        quantity=lp_quantity,
        tkn_add=asset2
    )
    second_diff = {tkn: lp_agent.holdings[tkn] - (initial_agent.holdings[tkn] if tkn in initial_agent.holdings else 0)
                   for tkn in lp_agent.holdings}

    # close arb
    glob = omnipool_arbitrage(pool_id='omnipool').execute(
        state=GlobalState(
            agents={
                'agent': lp_agent.copy(),
            },
            pools={
                'omnipool': lp_state.copy(),
            },
            external_market=market_prices
        ), agent_id='agent'
    )

    final_state = glob.pools['omnipool']
    final_agent = glob.agents['agent']

    final_diff = {tkn: final_agent.holdings[tkn] - (initial_agent.holdings[tkn] if tkn in initial_agent.holdings else 0) for
            tkn in final_agent.holdings}

    profit = (
            oamm.cash_out_omnipool(final_state, final_agent, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    loss = (
            oamm.value_assets(market_prices, final_state.liquidity)
            - oamm.value_assets(market_prices, initial_state.liquidity)
    )

    er = 1
