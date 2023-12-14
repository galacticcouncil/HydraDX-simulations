from hypothesis import given, strategies as st
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
import pytest
from mpmath import mp, mpf
mp.dps = 100

base_price_strat = st.floats(min_value=0.001, max_value=1000)
fee_strat = st.floats(min_value=0, max_value=0.1)

@st.composite
def order_book_strategy(draw, base_price: float = 0, book_depth: float = 0, price_points: int = 0):
    if not base_price:
        base_price = draw(base_price_strat)
    price_increment = draw(st.floats(min_value=0.000001, max_value=0.1))
    price_points = price_points or draw(st.integers(min_value=1, max_value=10))
    book_depth = book_depth or draw(st.integers(min_value=1, max_value=1000))
    return OrderBook(
        bids=[
            [base_price - i * base_price * price_increment, book_depth / price_points]
            for i in range(1, price_points + 1)
        ],
        asks=[
            [base_price + i * base_price * price_increment, book_depth / price_points]
            for i in range(1, price_points + 1)
        ]
    )


initial_agent = Agent(
    holdings={
        'DAI': mpf(10000000000),
        'ETH': mpf(10000000000)
    }
)


@given(
    sell_quantity=st.floats(min_value=0.1, max_value=100),
    order_book=order_book_strategy(book_depth=1000000),
    trade_fee=fee_strat
)
def test_sell_quote(sell_quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                },
                trade_fee=trade_fee
            )
        },
        agents={'agent': initial_agent}
    )
    tkn_buy = 'ETH'
    tkn_sell = 'DAI'
    sell_state = initial_state.copy()
    sell_state.execute_swap(
        agent_id='agent',
        pool_id='Kraken',
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity
    )

    value_bought = sum(
        [sell_quantity * price for (price, sell_quantity)
         in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [sell_quantity * price for (price, sell_quantity)
         in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - sell_state.agents['agent'].holdings[tkn_sell]

    if value_bought != pytest.approx(quantity_sold):
        raise AssertionError('Central market sell trade failed to execute correctly.')

    quantity_bought = sell_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    value_sold = sum(
        [sell_quantity for (price, sell_quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [sell_quantity for (price, sell_quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    if value_sold * (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_bought):
        raise AssertionError('Fee was not applied correctly.')

    if quantity_sold != pytest.approx(sell_quantity):
        raise AssertionError('Sell quantity was not applied correctly.')


@given(
    sell_quantity=st.floats(min_value=0.1, max_value=100),
    order_book=order_book_strategy(book_depth=1000000),
    trade_fee=fee_strat
)
def test_sell_base(sell_quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                },
                trade_fee=trade_fee
            )
        },
        agents={'agent': initial_agent}
    )
    tkn_buy = 'DAI'
    tkn_sell = 'ETH'
    buy_state = initial_state.copy()
    buy_state.execute_swap(
        pool_id='Kraken',
        agent_id='agent',
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    value_bought = sum(
        [sell_quantity for (price, sell_quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [sell_quantity for (price, sell_quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    if value_bought != pytest.approx(quantity_sold):
        raise AssertionError('Central market sell trade failed to execute correctly.')

    value_sold = sum(
        [sell_quantity * price for (price, sell_quantity)
         in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [sell_quantity * price for (price, sell_quantity)
         in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if value_sold * (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_bought):
        raise AssertionError('Fee was not applied correctly.')

    if quantity_sold != pytest.approx(sell_quantity):
        raise AssertionError('Sell quantity was not applied correctly.')


@given(
    buy_quantity=st.floats(min_value=0.1, max_value=100),
    order_book=order_book_strategy(book_depth=1000000),
    trade_fee=fee_strat
)
def test_buy_quote(buy_quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                },
                trade_fee=trade_fee
            )
        },
        agents={'agent': initial_agent}
    )
    tkn_buy = 'DAI'
    tkn_sell = 'ETH'
    buy_state = initial_state.copy()
    buy_state.execute_swap(
        pool_id='Kraken',
        agent_id='agent',
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )

    value_sold = sum(
        [buy_quantity * price for (price, buy_quantity)
         in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [buy_quantity * price for (price, buy_quantity)
         in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if quantity_bought != pytest.approx(buy_quantity):
        raise AssertionError('Buy quantity was not applied correctly.')

    if value_sold != pytest.approx(quantity_bought):
        raise AssertionError('Central market buy trade failed to execute correctly.')

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    value_bought = sum(
        [buy_quantity for (price, buy_quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [buy_quantity for (price, buy_quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    if value_bought * (1 + initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_sold):
        raise AssertionError('Fee was not applied correctly.')

    if quantity_bought != pytest.approx(buy_quantity):
        raise AssertionError('Buy quantity was not applied correctly.')


@given(
    buy_quantity=st.floats(min_value=0.1, max_value=100),
    order_book=order_book_strategy(book_depth=1000000),
    trade_fee=fee_strat
)
def test_buy_base(buy_quantity: float, order_book: OrderBook, trade_fee):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                },
                trade_fee=trade_fee
            )
        },
        agents={'agent': initial_agent}
    )
    tkn_buy = 'ETH'
    tkn_sell = 'DAI'
    buy_state = initial_state.copy()
    buy_state.execute_swap(
        agent_id='agent',
        pool_id='Kraken',
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if quantity_bought != pytest.approx(buy_quantity):
        raise AssertionError('Buy quantity was not applied correctly.')

    value_sold = sum(
        [buy_quantity for (price, buy_quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [buy_quantity for (price, buy_quantity) in buy_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    if value_sold != pytest.approx(quantity_bought):
        raise AssertionError('Central market buy trade failed to execute correctly.')

    value_bought = sum(
        [buy_quantity * price for (price, buy_quantity)
         in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [buy_quantity * price for (price, buy_quantity)
         in buy_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    if value_bought * (1 + initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_sold):
        raise AssertionError('Fee was not applied correctly.')

    if quantity_bought != pytest.approx(buy_quantity):
        raise AssertionError('Buy quantity was not applied correctly.')

        
@given(
    orderbook=order_book_strategy()
)
def test_orderbook(orderbook):
    last_bid = orderbook.bids[0][0]
    for bid in orderbook.bids:
        if bid[0] > last_bid:
            raise AssertionError('Bids are not sorted correctly.')
    last_ask = orderbook.asks[0][0]
    for ask in orderbook.asks:
        if ask[0] < last_ask:
            raise AssertionError('Asks are not sorted correctly.')


@given(
    buy_quantity=st.floats(min_value=0.01, max_value=100),
    order_book=order_book_strategy(book_depth=10000),
    trade_fee=fee_strat
)
def test_calculate_sell_from_buy(order_book: OrderBook, buy_quantity: float, trade_fee):
    initial_cex = CentralizedMarket(
        order_book={
            ('ETH', 'DAI'): order_book
        },
        trade_fee=trade_fee
    )
    tkn_sell = 'ETH'
    tkn_buy = 'DAI'
    sell_quantity = initial_cex.calculate_sell_from_buy(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity,
    )
    agent = Agent(
        holdings={'ETH': mpf(10000000)},
    )
    test_cex = initial_cex.copy()
    test_cex.swap(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity,
        agent=agent
    )
    actual_sell_quantity = agent.initial_holdings[tkn_sell] - agent.holdings[tkn_sell]
    if sell_quantity != pytest.approx(actual_sell_quantity):
        raise AssertionError('Loss detected.')


@given(
    sell_quantity=st.floats(min_value=0.01, max_value=100),
    order_book=order_book_strategy(book_depth=10000)
)
def test_calculate_buy_from_sell(order_book: OrderBook, sell_quantity: float):
    initial_cex = CentralizedMarket(
        order_book={
            ('ETH', 'DAI'): order_book
        },
    )
    tkn_sell = 'ETH'
    tkn_buy = 'DAI'
    buy_quantity = initial_cex.calculate_buy_from_sell(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity,
    )
    agent = Agent(
        holdings={'ETH': mpf(10000000), 'DAI': mpf(0)},
    )
    test_cex = initial_cex.copy()
    test_cex.swap(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity,
        agent=agent
    )
    actual_buy_quantity = agent.holdings[tkn_buy] - agent.initial_holdings[tkn_buy]
    if buy_quantity != pytest.approx(actual_buy_quantity):
        raise AssertionError('Loss detected.')


@given(
    order_book=order_book_strategy(book_depth=10000, price_points=1),
    trade_fee=fee_strat
)
def test_buy_quote_price(order_book: OrderBook, trade_fee: float):
    cex = CentralizedMarket(
        order_book={
            ('ETH', 'DAI'): order_book
        },
        trade_fee=trade_fee
    )
    test_cex = cex.copy()
    test_agent = initial_agent.copy()
    eth_per_dai = cex.buy_spot(tkn_buy='DAI', tkn_sell='ETH')
    dai_per_eth = cex.sell_spot(tkn_sell='ETH', tkn_buy='DAI')
    test_cex.swap(
        tkn_buy='DAI',
        tkn_sell='ETH',
        buy_quantity=1,
        agent=test_agent
    )
    ex_price_dai = (
        (test_agent.holdings['ETH'] - test_agent.initial_holdings['ETH'])
        / (test_agent.initial_holdings['DAI'] - test_agent.holdings['DAI'])
    )
    ex_price_eth = (
        (test_agent.initial_holdings['DAI'] - test_agent.holdings['DAI'])
        / (test_agent.holdings['ETH'] - test_agent.initial_holdings['ETH'])
    )
    if dai_per_eth != pytest.approx(ex_price_eth, rel=1e-20):
        raise AssertionError('buy spot gave incorrect price for quote asset')
    if eth_per_dai != pytest.approx(ex_price_dai, rel=1e-20):
        raise AssertionError('sell spot gave incorrect price for base asset')


@given(
    order_book=order_book_strategy(book_depth=10000, price_points=1),
    trade_fee=fee_strat
)
def test_sell_quote_price(order_book: OrderBook, trade_fee: float):
    cex = CentralizedMarket(
        order_book={
            ('ETH', 'DAI'): order_book
        },
        trade_fee=trade_fee
    )
    test_cex = cex.copy()
    test_agent = initial_agent.copy()
    eth_per_dai = cex.sell_spot(tkn_sell='DAI', tkn_buy='ETH')
    dai_per_eth = cex.buy_spot(tkn_buy='ETH', tkn_sell='DAI')
    test_cex.swap(
        tkn_sell='DAI',
        tkn_buy='ETH',
        sell_quantity=1,
        agent=test_agent
    )
    ex_price_dai = (
        (test_agent.holdings['ETH'] - test_agent.initial_holdings['ETH'])
        / (test_agent.initial_holdings['DAI'] - test_agent.holdings['DAI'])
    )
    ex_price_eth = (
        (test_agent.initial_holdings['DAI'] - test_agent.holdings['DAI'])
        / (test_agent.holdings['ETH'] - test_agent.initial_holdings['ETH'])
    )
    if dai_per_eth != pytest.approx(ex_price_eth):
        raise AssertionError('sell spot gave incorrect price')
    if eth_per_dai != pytest.approx(ex_price_dai):
        raise AssertionError('sell spot gave incorrect price')


@given(
    order_book_strategy(book_depth=100, price_points=2),
)
def test_buy_sell_limit(order_book: OrderBook):
    initial_cex = CentralizedMarket(
        order_book={
            ('ETH', 'DAI'): order_book
        },
    )
    test_buy_agent = initial_agent.copy()
    test_buy_cex = initial_cex.copy()
    test_buy_cex.swap(
        tkn_buy='DAI',
        tkn_sell='ETH',
        buy_quantity=test_buy_cex.buy_limit(tkn_buy='DAI', tkn_sell='ETH'),
        agent=test_buy_agent
    )
    test_buy_cex.swap(
        tkn_buy='ETH',
        tkn_sell='DAI',
        buy_quantity=test_buy_cex.buy_limit(tkn_buy='ETH', tkn_sell='DAI'),
        agent=test_buy_agent
    )
    # this should exactly exhaust the first price point (or at least almost)
    if (
            len(test_buy_cex.order_book[('ETH', 'DAI')].bids) != 1
            and test_buy_cex.order_book[('ETH', 'DAI')].bids[0][1] > 1e-10
    ):
        raise AssertionError('quote, base buy limit not correct')
    if len(test_buy_cex.order_book[('ETH', 'DAI')].asks) != 1 or (
            test_buy_cex.order_book[('ETH', 'DAI')].asks[0][1] != initial_cex.order_book[('ETH', 'DAI')].asks[1][1]
    ):
        raise AssertionError('base, quote buy limit not correct')
    test_sell_cex = initial_cex.copy()
    test_sell_agent = initial_agent.copy()
    test_sell_cex.swap(
        tkn_sell='DAI',
        tkn_buy='ETH',
        sell_quantity=test_sell_cex.sell_limit(tkn_sell='DAI', tkn_buy='ETH'),
        agent=test_sell_agent
    )
    test_sell_cex.swap(
        tkn_sell='ETH',
        tkn_buy='DAI',
        sell_quantity=test_sell_cex.sell_limit(tkn_sell='ETH', tkn_buy='DAI'),
        agent=test_sell_agent
    )
    # this should exactly exhaust the first price point (or at least almost)
    if (
            len(test_sell_cex.order_book[('ETH', 'DAI')].asks) != 1
            and test_sell_cex.order_book[('ETH', 'DAI')].asks[0][1] > 1e-10
    ):
        raise AssertionError('quote, base sell limit not correct')
    if len(test_sell_cex.order_book[('ETH', 'DAI')].bids) != 1 or (
            test_sell_cex.order_book[('ETH', 'DAI')].bids[0][1] != initial_cex.order_book[('ETH', 'DAI')].bids[1][1]
    ):
        raise AssertionError('base, quote sell limit not correct')
