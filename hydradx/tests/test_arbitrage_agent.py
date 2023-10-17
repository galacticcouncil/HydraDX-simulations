from hypothesis import given, strategies as st, settings, reproduce_failure

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.arbitrage_agent import calculate_profit, calculate_arb_amount_bid, calculate_arb_amount_ask
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


# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=1.1, max_value=10.0),
)
def test_calculate_arb_amount_bid(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float
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

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

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
    bid = {'price': bid_price, 'amount': 100000}
    p = 1e-10
    amt = calculate_arb_amount_bid(initial_state, tkn, numeraire, bid, cex_fee, precision=p)
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    buy_spot = test_price / ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = bid['price'] * (1 - cex_fee)

    if abs(buy_spot - cex_price) > p and amt != bid['amount']:
        raise

    if amt == bid['amount'] and buy_spot > cex_price:
        raise

    agent.holdings[tkn] -= amt
    agent.holdings[numeraire] += amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0


# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=0.1, max_value=0.95),
)
def test_calculate_arb_amount_ask(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float
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

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

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
    ask = {'price': ask_price, 'amount': 100000}
    p = 1e-10
    amt = calculate_arb_amount_ask(initial_state, tkn, numeraire, ask, cex_fee, precision=p)
    agent = Agent(holdings={'USDT': 1000000000, 'DOT': 1000000000, 'HDX': 1000000000}, unique_id='bot')
    init_agent = agent.copy()
    initial_state.swap(agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    test_price = initial_state.price(initial_state, tkn, numeraire)
    sell_spot = test_price * ((1 - lrna_fee) * (1 - asset_fee))
    cex_price = ask['price'] / (1 - cex_fee)

    if abs(sell_spot - cex_price) > p and amt != ask['amount']:
        raise

    if amt == ask['amount'] and cex_price > sell_spot:
        raise

    agent.holdings[tkn] += amt
    agent.holdings[numeraire] -= amt * cex_price

    profit = calculate_profit(init_agent, agent)
    for tkn in profit:
        assert profit[tkn] >= 0
