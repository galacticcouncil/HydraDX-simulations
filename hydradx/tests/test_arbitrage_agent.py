import copy
from datetime import timedelta

from hypothesis import given, strategies as st, settings, Phase

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.arbitrage_agent import calculate_profit, calculate_arb_amount_bid, calculate_arb_amount_ask, \
    process_next_swap
from hydradx.model.amm.arbitrage_agent import get_arb_swaps_simple, execute_arb, get_arb_swaps
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
from hydradx.model.amm.omnipool_amm import OmnipoolState


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

    order_book_map = {k: k for k in order_book}

    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')

    test_state = op_state.copy()
    test_agent = agent.copy()
    test_cex = cex.copy()
    buffer = 0.0
    init_max_liquidity = {'cex': {'USDT': 1000, 'DOT': 100, 'HDX': 100000},
                          'dex': {'USDT': 1000, 'DOT': 100, 'HDX': 100000}}
    max_liquidity = copy.deepcopy(init_max_liquidity)
    iters = 20
    tkn_pair = ('DOT', 'USDT')

    swap = process_next_swap(test_state, test_agent, test_cex, tkn_pair, tkn_pair, buffer, max_liquidity, iters)
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

        arb_swaps = [swap]

        initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
        agent = initial_agent.copy()

        execute_arb(op_state, cex, agent, arb_swaps)

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

    buffer = {(('DOT', 'USDT'), ('DOT', 'USDT')): 0.0,
              (('HDX', 'USDT'), ('HDX', 'USDT')): 0.0,
              (('HDX', 'DOT'), ('HDX', 'DOT')): 0.0}

    arb_swaps = get_arb_swaps_simple(op_state, cex, buffer)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, cex, agent, arb_swaps)

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
        buffer_ls: float
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

    buffer = {(k, k): buffer_ls[i] for i, k in enumerate(order_book)}

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    arb_swaps = get_arb_swaps_simple(op_state, cex, buffer)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, cex, agent, arb_swaps)

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

    buffer = {(k, k): 0 for k in order_book}
    arb_swaps = get_arb_swaps(op_state, cex, buffer)
    initial_agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb(op_state, cex, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise
