import copy
from datetime import timedelta
import os
from hypothesis import given, strategies as st, settings, Phase

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.arbitrage_agent import calculate_profit, calculate_arb_amount_bid, calculate_arb_amount_ask, \
    process_next_swap, execute_arb, get_arb_swaps, get_arb_swaps_simple, combine_swaps, flatten_swaps
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
from hydradx.model.amm.omnipool_amm import OmnipoolState, lrna_price
from hydradx.model.processing import get_omnipool_data, get_omnipool_data_from_file, get_centralized_market, \
    get_orderbooks_from_file  # , get_stableswap_data, get_unique_name
from hydradx.model.amm.arbitrage_agent_general import get_arb_swaps as get_arb_swaps_general, \
    execute_arb as execute_arb_general, flatten_swaps as flatten_swaps_general
from mpmath import mp, mpf


def test_calculate_profit():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    calculated_profit = calculate_profit(init_agent, agent)
    for tkn in init_agent.holdings:
        assert calculated_profit[tkn] == profits[tkn]


def test_calculate_profit_with_mapping():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000, 'DAI': 4000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0, 'DAI': 100}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DAI'] += profits['DAI']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    mapping = {'DAI': 'USD', 'USDT': 'USD'}
    calculated_profit = calculate_profit(init_agent, agent, mapping)

    assert calculated_profit['USD'] == 200
    assert calculated_profit['DOT'] == 200
    assert calculated_profit['HDX'] == 0


# @settings(max_examples=1)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=1.1, max_value=10.0),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
)
def test_calculate_arb_amount_bid(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    orig_price = initial_state.price(initial_state, 'DOT', 'USDT')
    buy_spot = orig_price / ((1 - lrna_fee) * (1 - asset_fee))
    bid_price = buy_spot / (1 - cex_fee) * price_mult

    tkn = 'DOT'
    numeraire = 'USDT'
    bid = [bid_price, 100000]
    p = 1e-10
    amt = calculate_arb_amount_bid(initial_state, tkn, numeraire, bid, cex_fee, precision=p, max_iters=1000)
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    buy_spot = test_price / ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = bid[0] * (1 - cex_fee)

    if abs(buy_spot - cex_price) / cex_price > p and abs(buy_spot - cex_price) > p and amt != bid[1]:
        raise

    if amt == bid[1] and buy_spot > cex_price:
        raise

    agent.holdings[tkn] -= amt
    agent.holdings[numeraire] += amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0


# @settings(max_examples=1)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=1.1, max_value=10.0),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
    max_trade=st.floats(min_value=1, max_value=100),
)
def test_calculate_arb_amount_bid_max_liquidity(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float,
        max_trade: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    orig_price = initial_state.price(initial_state, 'DOT', 'USDT')
    buy_spot = orig_price / ((1 - lrna_fee) * (1 - asset_fee))
    bid_price = buy_spot / (1 - cex_fee) * price_mult

    tkn = 'DOT'
    numeraire = 'USDT'
    bid = [bid_price, 100000]
    p = 1e-10
    amt = calculate_arb_amount_bid(initial_state, tkn, numeraire, bid, cex_fee, max_liq_tkn=max_trade,
                                   max_liq_num=max_trade, precision=p, max_iters=1000)
    init_holding = 1000000
    agent = Agent(holdings={'USDT': init_holding, 'DOT': init_holding, 'HDX': init_holding}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    buy_spot = test_price / ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = bid[0] * (1 - cex_fee)

    if (abs(init_holding - agent.holdings[tkn]) - max_trade) / init_holding > 1e-10:
        raise
    if (abs(init_holding - agent.holdings[numeraire]) - max_trade) / init_holding > 1e-10:
        raise

    # checks if the cex price and spot price have been brought into alignment
    if abs(buy_spot - cex_price) / cex_price > p and abs(buy_spot - cex_price) > p and amt != bid[1]:
        # if cex price and spot price aren't in alignment, it should be because of trade size limit
        if ((max_trade - abs(init_holding - agent.holdings[tkn])) / init_holding) > 1e-10:
            if ((max_trade - abs(init_holding - agent.holdings[numeraire])) / init_holding) > 1e-10:
                raise

    if amt == bid[1] and buy_spot > cex_price:
        raise

    agent.holdings[tkn] -= amt
    agent.holdings[numeraire] += amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=0.1, max_value=0.95),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
)
def test_calculate_arb_amount_ask(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    orig_price = initial_state.price(initial_state, 'DOT', 'USDT')
    sell_spot = orig_price * ((1 - lrna_fee) * (1 - asset_fee))
    ask_price = sell_spot * (1 - cex_fee) * price_mult

    tkn = 'DOT'
    numeraire = 'USDT'
    ask = [ask_price, 100000]
    p = 1e-10
    amt = calculate_arb_amount_ask(initial_state, tkn, numeraire, ask, cex_fee, precision=p, max_iters=1000)
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    sell_spot = test_price * ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = ask[0] * (1 + cex_fee)

    if abs(sell_spot - cex_price) / cex_price > p and abs(sell_spot - cex_price) > p and amt != ask[1]:
        raise

    if amt == ask[1] and cex_price > sell_spot:
        raise

    agent.holdings[tkn] += amt
    agent.holdings[numeraire] -= amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=0.1, max_value=0.95),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
    max_trade=st.floats(min_value=1, max_value=100),
)
def test_calculate_arb_amount_ask_max_liquidity(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float,
        max_trade: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    orig_price = initial_state.price(initial_state, 'DOT', 'USDT')
    sell_spot = orig_price * ((1 - lrna_fee) * (1 - asset_fee))
    ask_price = sell_spot * (1 - cex_fee) * price_mult

    tkn = 'DOT'
    numeraire = 'USDT'
    ask = [ask_price, 100000]
    p = 1e-10
    amt = calculate_arb_amount_ask(initial_state, tkn, numeraire, ask, cex_fee, max_liq_tkn=max_trade,
                                   max_liq_num=max_trade, precision=p, max_iters=1000)
    init_holding = 1000000
    agent = Agent(holdings={'USDT': init_holding, 'DOT': init_holding, 'HDX': init_holding}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    sell_spot = test_price * ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = ask[0] * (1 + cex_fee)

    if (abs(init_holding - agent.holdings[tkn]) - max_trade) / init_holding > 1e-10:
        raise
    if (abs(init_holding - agent.holdings[numeraire]) - max_trade) / init_holding > 1e-10:
        raise

    # checks if the cex price and spot price have been brought into alignment
    if abs(sell_spot - cex_price) / cex_price > p and abs(sell_spot - cex_price) > p and amt != ask[1]:
        # if cex price and spot price aren't in alignment, it should be because of trade size limit
        if ((max_trade - abs(init_holding - agent.holdings[tkn])) / init_holding) > 1e-10:
            if ((max_trade - abs(init_holding - agent.holdings[numeraire])) / init_holding) > 1e-10:
                raise

    if amt == ask[1] and cex_price > sell_spot:
        raise

    agent.holdings[tkn] += amt
    agent.holdings[numeraire] -= amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
)
def test_process_next_swap(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):
    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price(op_state, 'DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price(op_state, 'HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price(op_state, 'HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                       [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX', 'DOT'): hdx_dot_order_book_obj
    }

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')

    test_state = op_state.copy()
    test_agent = agent.copy()
    test_cex = cex.copy()
    buffer = 0.0
    init_max_liquidity = {
        'dex': {'USDT': 1000, 'DOT': 100, 'HDX': 100000},
        'kraken': {'USDT': 1000, 'DOT': 100, 'HDX': 100000},
        'binance': {'USDT': 1000, 'DOT': 100, 'HDX': 100000}
    }
    max_liquidity = copy.deepcopy(init_max_liquidity)
    iters = 20
    tkn_pair = ('DOT', 'USDT')

    swap = process_next_swap(test_state, test_agent, test_cex, tkn_pair, tkn_pair, buffer, max_liquidity['dex'], max_liquidity['kraken'], iters)
    if swap:
        cex_swap, dex_swap = swap['cex'], swap['dex']
        dex_spot = op_state.price(op_state, 'DOT', 'USDT')
        if cex_swap['buy_asset'] != dex_swap['sell_asset'] or cex_swap['sell_asset'] != dex_swap['buy_asset']:
            raise  # check that trades match
        cex_numeraire_amt = cex_swap['amount'] * cex_swap['price']
        if dex_swap['trade'] == 'sell':
            dex_numeraire_amt = dex_swap['min_buy']
            if dex_numeraire_amt < cex_numeraire_amt:  # check profitability
                raise
            if dex_swap['min_buy'] / dex_swap['amount'] > dex_spot:  # check dex slippage direction
                raise
            if cex_swap['price'] < dex_swap['price']:  # check cex slippage direction
                raise
        elif dex_swap['trade'] == 'buy':
            dex_numeraire_amt = dex_swap['max_sell']
            if dex_numeraire_amt > cex_numeraire_amt:  # check profitability
                raise
            if dex_swap['max_sell'] / dex_swap['amount'] < dex_spot:  # check dex slippage direction
                raise
            if cex_swap['price'] > dex_swap['price']:  # check cex slippage direction
                raise

        swap['exchange'] = 'exchange_name'

        arb_swaps = [swap]

        initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
        agent = initial_agent.copy()

        execute_arb(op_state, {'exchange_name': cex}, agent, arb_swaps)

        profit = calculate_profit(initial_agent, agent)
        for tkn in profit:
            if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
                raise

        for tkn in op_state.asset_list:
            if tkn in max_liquidity:
                if test_state.liquidity[tkn] - op_state.liquidity[tkn] != init_max_liquidity['dex'][tkn] - \
                        max_liquidity['dex'][tkn]:
                    raise
        for tkn in cex.asset_list:
            if tkn in max_liquidity:
                if test_cex.liquidity[tkn] - cex.liquidity[tkn] != init_max_liquidity['cex'][tkn] - \
                        max_liquidity['cex'][tkn]:
                    raise


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
)
def test_get_arb_swaps_simple(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):
    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price(op_state, 'DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price(op_state, 'HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price(op_state, 'HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                       [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX', 'DOT'): hdx_dot_order_book_obj
    }

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    cfg = [
        {"tkn_pair": ("DOT", "USDT"), "exchange": "exchange_name", "order_book": ("DOT", "USDT"), "buffer": 0.0},
        {"tkn_pair": ("HDX", "USDT"), "exchange": "exchange_name", "order_book": ("HDX", "USDT"), "buffer": 0.0},
        {"tkn_pair": ("HDX", "DOT"), "exchange": "exchange_name", "order_book": ("HDX", "DOT"), "buffer": 0.0}
    ]

    arb_swaps = get_arb_swaps_simple(op_state, {'exchange_name': cex}, cfg)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, {'exchange_name': cex}, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    buffer_ls=st.lists(st.floats(min_value=0.0002, max_value=0.0100), min_size=3, max_size=3),
)
def test_get_arb_swaps_simple_with_buffer(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list,
        buffer_ls: list
):
    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price(op_state, 'DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price(op_state, 'HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price(op_state, 'HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                       [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX', 'DOT'): hdx_dot_order_book_obj
    }

    cfg = [
        {"tkn_pair": k, "exchange": "exchange_name", "order_book": k, "buffer": buffer_ls[i]}
        for i, k in enumerate(order_book)
    ]

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    arb_swaps = get_arb_swaps_simple(op_state, {'exchange_name': cex}, cfg)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, {'exchange_name': cex}, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
)
def test_get_arb_swaps(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):
    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price(op_state, 'DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price(op_state, 'HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price(op_state, 'HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                       [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX', 'DOT'): hdx_dot_order_book_obj
    }

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    cfg = [{"tkn_pair": k, "exchange": "exchange_name", "order_book": k, "buffer": 0.0} for k in order_book]
    arb_swaps = get_arb_swaps(op_state, {"exchange_name": cex}, cfg)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, {"exchange_name": cex}, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise


def test_combine_step():

    cfg = [
        {"tkn_ids": (0, 10), "exchange": "kraken", "order_book": ("HDX", "USD")},
        {"tkn_ids": (5, 10), "exchange": "kraken", "order_book": ("DOT", "USDT")},
        {"tkn_ids": (20, 10), "exchange": "kraken", "order_book": ("ETH", "USDT")},
        {"tkn_ids": (20, 18), "exchange": "kraken", "order_book": ("ETH", "DAI")},
        {"tkn_ids": (5, 20), "exchange": "kraken", "order_book": ("DOT", "ETH")},
        {"tkn_ids": (19, 10), "exchange": "kraken", "order_book": ("BTC", "USDT")},
        {"tkn_ids": (11, 10), "exchange": "kraken", "order_book": ("BTC", "USDT")},
        {"tkn_ids": (19, 18), "exchange": "kraken", "order_book": ("BTC", "DAI")},
        {"tkn_ids": (11, 18), "exchange": "kraken", "order_book": ("BTC", "DAI")},
        {"tkn_ids": (5, 19), "exchange": "kraken", "order_book": ("DOT", "BTC")},
        {"tkn_ids": (5, 11), "exchange": "kraken", "order_book": ("DOT", "BTC")},
        {"tkn_ids": (20, 19), "exchange": "kraken", "order_book": ("ETH", "BTC")},
        {"tkn_ids": (20, 11), "exchange": "kraken", "order_book": ("ETH", "BTC")},
        {"tkn_ids": (9, 10), "exchange": "kraken", "order_book": ("ASTR", "USD")},
        {"tkn_ids": (13, 10), "exchange": "kraken", "order_book": ("CFG", "USD")},
        {"tkn_ids": (14, 10), "exchange": "kraken", "order_book": ("BNC", "USD")},
        {"tkn_ids": (16, 10), "exchange": "kraken", "order_book": ("GLMR", "USD")},
        {"tkn_ids": (17, 10), "exchange": "kraken", "order_book": ("INTR", "USD")},
        {"tkn_ids": (5, 10), "exchange": "binance", "order_book": ("DOT", "USDT")},
        {"tkn_ids": (5, 20), "exchange": "binance", "order_book": ("DOT", "ETH")},
        {"tkn_ids": (5, 19), "exchange": "binance", "order_book": ("DOT", "BTC")},
        {"tkn_ids": (5, 11), "exchange": "binance", "order_book": ("DOT", "BTC")},
        {"tkn_ids": (20, 10), "exchange": "binance", "order_book": ("ETH", "USDT")},
        {"tkn_ids": (20, 18), "exchange": "binance", "order_book": ("ETH", "DAI")},
        {"tkn_ids": (20, 19), "exchange": "binance", "order_book": ("ETH", "BTC")},
        {"tkn_ids": (20, 11), "exchange": "binance", "order_book": ("ETH", "BTC")},
        {"tkn_ids": (19, 10), "exchange": "binance", "order_book": ("BTC", "USDT")},
        {"tkn_ids": (11, 10), "exchange": "binance", "order_book": ("BTC", "USDT")},
        {"tkn_ids": (19, 18), "exchange": "binance", "order_book": ("BTC", "DAI")},
        {"tkn_ids": (11, 18), "exchange": "binance", "order_book": ("BTC", "DAI")},
        {"tkn_ids": (9, 10), "exchange": "binance", "order_book": ("ASTR", "USDT")},
        {"tkn_ids": (16, 10), "exchange": "binance", "order_book": ("GLMR", "USDT")}
    ]

    #
    # asset_list, asset_numbers, tokens, fees = get_omnipool_data(rpc='wss://rpc.hydradx.cloud', archive=False)
    #
    # kraken = get_centralized_market(config=cfg, exchange_name='kraken', trade_fee=0.0016, archive=False)
    # binance = get_centralized_market(config=cfg, exchange_name='binance', trade_fee=0.001, archive=False)
    # cex = {
    #     'kraken': kraken,
    #     'binance': binance
    # }
    # uncomment above to test with live data, below for archived data
    #
    input_path = './data/'
    if not os.path.exists(input_path):
        input_path = 'hydradx/tests/data/'
    asset_list, asset_numbers, tokens, fees = get_omnipool_data_from_file(input_path)

    cex = {}
    for exchange in ('kraken', 'binance'):
        cex[exchange] = CentralizedMarket(
            order_book=get_orderbooks_from_file(input_path=input_path)[exchange],
            unique_id=exchange,
            trade_fee={'kraken': 0.0016, 'binance': 0.001}[exchange]
        )
    kraken = cex['kraken']
    binance = cex['binance']

    # get arb cfg
    for arb_cfg in cfg:
        arb_cfg['tkn_pair'] = (asset_numbers[arb_cfg['tkn_ids'][0]], asset_numbers[arb_cfg['tkn_ids'][1]])
        arb_cfg['buffer'] = 0.001

    dex = OmnipoolState(
        tokens=tokens,
        lrna_fee={asset: fees[asset]['protocol_fee'] for asset in asset_list},
        asset_fee={asset: fees[asset]['asset_fee'] for asset in asset_list},
        preferred_stablecoin='USDT',
        unique_id='dex'
    )

    asset_map = {
        'WETH': 'ETH',
        'XETH': 'ETH',
        'XXBT': 'BTC',
        'WBTC': 'BTC',
        'ZUSD': 'USD',
        'USDT': 'USD',
        'USDC': 'USD',
        'DAI': 'USD',
        'USDT001': 'USD',
        'DAI001': 'USD',
        'WETH001': 'ETH',
        'WBTC001': 'BTC',
        'iBTC': 'BTC',
        'XBT': 'BTC'
    }
    exchanges_per_tkn = {
        tkn: sum(1 if tkn in exchange.asset_list else 0 for exchange in (dex, kraken, binance))
        for tkn in dex.asset_list + kraken.asset_list + binance.asset_list + list(asset_numbers.values())
    }
    initial_agent = Agent(
        holdings={
            tkn: 1000000 * exchanges_per_tkn[tkn]
            / (
                (binance.buy_spot(tkn, 'USD') or kraken.buy_spot(tkn, 'USD') or dex.buy_spot(tkn, 'USD')) or 1
            )
            for tkn in dex.asset_list + kraken.asset_list + binance.asset_list + list(asset_numbers.values())
        }
    )
    max_liquidity = {
        'dex': {tkn: initial_agent.holdings[tkn] / exchanges_per_tkn[tkn] for tkn in initial_agent.holdings},
        'cex': {
            ex.unique_id:
            {tkn: initial_agent.holdings[tkn] / exchanges_per_tkn[tkn] for tkn in initial_agent.holdings}
            for ex in (binance, kraken)
        }
    }
    arb_swaps = get_arb_swaps(
        dex, cex, cfg, max_liquidity=max_liquidity
    )

    test_dex, test_kraken, test_binance, test_agent = \
        dex.copy(), kraken.copy(), binance.copy(), initial_agent.copy()
    execute_arb(test_dex, {'kraken': test_kraken, 'binance': test_binance}, test_agent, arb_swaps)
    profit = calculate_profit(initial_agent, test_agent, asset_map)
    profit_total = test_binance.value_assets(profit, asset_map)
    print('profit: ', profit)
    print('total: ', profit_total)

    combine_dex, combine_kraken, combine_binance, combine_agent = \
        dex.copy(), kraken.copy(), binance.copy(), initial_agent.copy()
    combined_swaps = combine_swaps(
        dex=combine_dex,
        cex={'kraken': combine_kraken, 'binance': combine_binance},
        agent=combine_agent,
        all_swaps=arb_swaps,
        asset_map=asset_map,
        max_liquidity=max_liquidity
    )
    execute_arb(combine_dex, {'kraken': combine_kraken, 'binance': combine_binance}, combine_agent, combined_swaps)
    combined_profit = calculate_profit(initial_agent, combine_agent, asset_map)
    combined_profit_total = combine_binance.value_assets(combined_profit, asset_map)

    # see if iterating on that can get any extra profit
    iter_dex, iter_kraken, iter_binance, iter_agent = (
        combine_dex.copy(), combine_kraken.copy(), combine_binance.copy(), combine_agent.copy()
    )
    max_liquidity = {
        'dex': {tkn: iter_agent.holdings[tkn] / exchanges_per_tkn[tkn] for tkn in iter_agent.holdings},
        'cex': {
            ex.unique_id:
            {tkn: iter_agent.holdings[tkn] / exchanges_per_tkn[tkn] for tkn in iter_agent.holdings}
            for ex in (binance, kraken)
        }
    }
    optimized_arb_swaps = get_arb_swaps(
        iter_dex, {'kraken': iter_kraken, 'binance': iter_binance}, cfg, max_liquidity=max_liquidity
    )
    itered_swaps = combine_swaps(
        iter_dex, {'kraken': iter_kraken, 'binance': iter_binance}, iter_agent, optimized_arb_swaps, asset_map,
        max_liquidity=max_liquidity
    )
    execute_arb(iter_dex, {'kraken': iter_kraken, 'binance': iter_binance}, iter_agent, itered_swaps)
    iter_profit = calculate_profit(initial_agent, iter_agent, asset_map)
    iter_profit_total = iter_binance.value_assets(iter_profit, asset_map)

    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    for tkn in combined_profit:
        if combined_profit[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    if profit_total > combined_profit_total:
        raise AssertionError('Loss detected.')
    else:
        print(
            f"extra profit obtained by combining swaps: {combined_profit_total - profit_total}"
            f" ({(combined_profit_total - profit_total) / profit_total * 100}%)"
        )
        if iter_profit_total > combined_profit_total:
            print(f'Second iteration also gained {iter_profit_total - combined_profit_total}.')
        else:
            print('Iteration did not improve profit.')


def test_generalized_arb():
    exchanges = {
        'kraken': CentralizedMarket(
            order_book={
                ('DOT', 'USD'): OrderBook(
                    bids=[[0.999, 10000], [0.99, 10000], [0.9, 10000], [0.8, 10000]],
                    asks=[[1.001, 10000], [1.01, 10000], [1.1, 10000], [1.2, 10000]]
                ),
                ('HDX', 'USD'): OrderBook(
                    bids=[[0.0999, 10000], [0.099, 10000], [0.09, 10000], [0.085, 10000], [0.08, 10000]],
                    asks=[[0.1001, 10000], [0.101, 10000], [0.11, 10000], [0.12, 10000]]
                ),
                ('HDX', 'DOT'): OrderBook(
                    bids=[[0.099909, 10000], [0.099, 10000], [0.09, 10000], [0.08, 10000]],
                    asks=[[0.1001, 10000], [0.101, 10000], [0.11, 10000], [0.12, 10000]]
                )
            },
            unique_id='kraken',
            trade_fee=0.0016
        ),
        'binance': CentralizedMarket(
            order_book={
                ('DOT', 'USDT'): OrderBook(
                    bids=[[0.999, 10000], [0.99, 10000], [0.9, 10000], [0.8, 10000]],
                    asks=[[1.001, 10000], [1.01, 10000], [1.1, 10000], [1.2, 10000]]
                ),
                ('HDX', 'USDT'): OrderBook(
                    bids=[[0.0999, 10000], [0.099, 10000], [0.09, 10000], [0.085, 10000], [0.08, 10000]],
                    asks=[[0.1001, 10000], [0.101, 10000], [0.11, 10000], [0.12, 10000]]
                ),
                ('HDX', 'DOT'): OrderBook(
                    bids=[[0.09991, 10000], [0.099, 10000], [0.09, 10000], [0.08, 10000]],
                    asks=[[0.1001, 10000], [0.101, 10000], [0.11, 10000], [0.12, 10000]]
                )
            },
            unique_id='binance',
            trade_fee=0.001
        ),
        'omnipool': OmnipoolState(
            tokens={
                'USDT': {
                    'liquidity': 2062772,
                    'LRNA': 2062772
                },
                'DOT': {
                    'liquidity': 350000,
                    'LRNA': 350000
                },
                'HDX': {
                    'liquidity': 100000000,
                    'LRNA': 9910000
                }
            },
            lrna_fee=0.0005,
            asset_fee=0.0025,
            preferred_stablecoin='USDT',
            unique_id='omnipool'
        )
    }
    config = [
        {"exchanges": {"kraken": ("HDX", "USD"), "omnipool": ("HDX", "USDT")}, "buffer": 0.00},
        {"exchanges": {"binance": ("HDX", "USDT"), "omnipool": ("HDX", "USDT")}, "buffer": 0.00},
        {"exchanges": {"kraken": ("HDX", "DOT"), "omnipool": ("HDX", "DOT")}, "buffer": 0.00},
        {"exchanges": {"binance": ("HDX", "DOT"), "omnipool": ("HDX", "DOT")}, "buffer": 0.00},
        {"exchanges": {"kraken": ("DOT", "USD"), "omnipool": ("DOT", "USDT")}, "buffer": 0.00},
        {"exchanges": {"binance": ("DOT", "USDT"), "omnipool": ("DOT", "USDT")}, "buffer": 0.00},
    ]
    print(
        f"Kraken HDX/USDT buy price: "
        f"{exchanges['kraken'].buy_spot(tkn_buy='USDT', tkn_sell='HDX')}"
    )
    print(f"Omnipool HDX/USDT sell price: {exchanges['omnipool'].sell_spot(tkn_sell='HDX', tkn_buy='USDT')}")
    max_liquidity = {
        'omnipool': {'HDX': 100000, 'USDT': 100000, 'DOT': 100000},
        'kraken': {'HDX': 100000, 'USD': 100000, 'DOT': 100000},
        'binance': {'HDX': 100000, 'USDT': 100000, 'DOT': 100000}
    }
    arb_swaps_general = get_arb_swaps_general(
        exchanges={ex_name: ex.copy() for ex_name, ex in exchanges.items()},
        config=config,
        max_liquidity=max_liquidity
    )
    agent = Agent(
        holdings={
            'HDX': 200000,
            'USDT': 200000,
            'USD': 200000,
            'DOT': 200000
        }
    )
    general_arb_exchanges = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}
    execute_arb_general(
        exchanges=general_arb_exchanges,
        agent=agent,
        all_swaps=arb_swaps_general
    )
    agent_profit = calculate_profit(Agent(holdings=agent.initial_holdings), agent)
    print(agent_profit)
    print(f"Agent profit: {exchanges['binance'].value_assets(agent_profit, equivalency_map={'USDT': 'USD'})}")

    # try with the original agent for comparison
    original_agent = Agent(
        holdings={
            'HDX': 200000,
            'USDT': 200000,
            'USD': 200000,
            'DOT': 200000
        }
    )
    arb_swaps = get_arb_swaps(
        op_state=exchanges['omnipool'],
        cex_dict={'kraken': exchanges['kraken'], 'binance': exchanges['binance']},
        config=[
            {"tkn_pair": ('HDX', 'USDT'), "exchange": "kraken", "order_book": ("HDX", "USD"), "buffer": 0.00},
            {"tkn_pair": ('HDX', 'USDT'), "exchange": "binance", "order_book": ("HDX", "USDT"), "buffer": 0.00},
            {"tkn_pair": ('HDX', 'DOT'), "exchange": "kraken", "order_book": ("HDX", "DOT"), "buffer": 0.00},
            {"tkn_pair": ('HDX', 'DOT'), "exchange": "binance", "order_book": ("HDX", "DOT"), "buffer": 0.00},
            {"tkn_pair": ('DOT', 'USDT'), "exchange": "kraken", "order_book": ("DOT", "USD"), "buffer": 0.00},
            {"tkn_pair": ('DOT', 'USDT'), "exchange": "binance", "order_book": ("DOT", "USDT"), "buffer": 0.00},
        ],
        max_liquidity={
            'dex': {'HDX': 100000, 'USDT': 100000, 'DOT': 100000},
            'cex': {
                'kraken': {'HDX': 100000, 'USD': 100000, 'DOT': 100000},
                'binance': {'HDX': 100000, 'USDT': 100000, 'DOT': 100000}
            }
        }
    )
    original_arb_exchanges = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}
    execute_arb(
        dex=original_arb_exchanges['omnipool'],
        cex_dict={'kraken': original_arb_exchanges['kraken'], 'binance': original_arb_exchanges['binance']},
        agent=original_agent,
        all_swaps=arb_swaps
    )
    original_agent_profit = calculate_profit(Agent(holdings=original_agent.initial_holdings), original_agent)
    print(original_agent_profit)
    print(
        f"Original agent profit:"
        f" {exchanges['binance'].value_assets(original_agent_profit, equivalency_map={'USDT': 'USD'})}"
    )
    print('-------generalized swaps:-------')
    for i, swap in enumerate(flatten_swaps_general(arb_swaps_general)):
        print(i, swap)
    print('-------original swaps:-------')
    for i, swap in enumerate(flatten_swaps(arb_swaps)):
        print(i, swap)
    print('done.')
