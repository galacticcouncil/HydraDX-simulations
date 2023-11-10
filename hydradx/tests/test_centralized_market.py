from hypothesis import given, strategies as st
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
import pytest


base_price_strat = st.floats(min_value=0.001, max_value=1000)
fee_strat = st.floats(min_value=0, max_value=0.1)

@st.composite
def order_book_strategy(draw, base_price: float = 0, book_depth: float = 0):
    if not base_price:
        base_price = draw(base_price_strat)
    price_increment = draw(st.floats(min_value=0.000001, max_value=0.1))
    price_points = draw(st.integers(min_value=1, max_value=10))
    book_depth = book_depth or draw(st.integers(min_value=1, max_value=1000))
    return OrderBook(
        bids=[[base_price - i * price_increment, book_depth / price_points] for i in range(1, price_points + 1)],
        asks=[[base_price + i * price_increment, book_depth / price_points] for i in range(1, price_points + 1)]
    )


initial_agent = Agent(
    holdings={
        'DAI': 1000000,
        'ETH': 1000
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
        [sell_quantity * price for (price, sell_quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [sell_quantity * price for (price, sell_quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
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
        [sell_quantity * price for (price, sell_quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [sell_quantity * price for (price, sell_quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
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
        [buy_quantity * price for (price, buy_quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [buy_quantity * price for (price, buy_quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

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

    value_sold = sum(
        [buy_quantity for (price, buy_quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [buy_quantity for (price, buy_quantity) in buy_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    if value_sold != pytest.approx(quantity_bought):
        raise AssertionError('Central market buy trade failed to execute correctly.')

    value_bought = sum(
        [buy_quantity * price for (price, buy_quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [buy_quantity * price for (price, buy_quantity) in buy_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
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
            ('DOT', 'USD'): order_book
        },
        trade_fee=trade_fee
    )
    tkn_sell = 'DOT'
    tkn_buy = 'USD'
    sell_quantity = initial_cex.calculate_sell_from_buy(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity,
    )
    agent = Agent(
        holdings={'DOT': 10000000},
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
    order_book=order_book_strategy(book_depth=100)
)
def test_buy_spot(order_book: OrderBook):
    cex = CentralizedMarket(
        order_book={
            ('DOT', 'USD'): order_book
        },
    )
    agent = Agent(
        holdings={'USD': 10000000, 'DOT': 10000000},
    )
    test_cex = cex.copy()
    test_agent = agent.copy()
    buy_spot = cex.buy_spot('DOT', 'USD')
    test_cex.swap(
        tkn_sell='USD',
        tkn_buy='DOT',
        buy_quantity=1,
        agent=test_agent
    )
    ex_price = (
            (test_agent.initial_holdings['USD'] - test_agent.holdings['USD'])
            / (test_agent.holdings['DOT'] - test_agent.initial_holdings['DOT'])
    )
    if buy_spot != pytest.approx(ex_price):
        raise AssertionError('buy spot gave incorrect price')

    test_agent = agent.copy()
    buy_spot = cex.buy_spot('USD', 'DOT')
    test_cex.swap(
        tkn_sell='DOT',
        tkn_buy='USD',
        buy_quantity=1,
        agent=test_agent
    )
    ex_price = (
            (test_agent.initial_holdings['DOT'] - test_agent.holdings['DOT'])
            / (test_agent.holdings['USD'] - test_agent.initial_holdings['USD'])
    )
    if buy_spot != pytest.approx(ex_price):
        raise AssertionError('buy spot gave incorrect price')


@given(
    order_book=order_book_strategy(book_depth=100)
)
def test_sell_spot(order_book: OrderBook):
    cex = CentralizedMarket(
        order_book={
            ('DOT', 'USD'): order_book
        },
    )
    agent = Agent(
        holdings={'USD': 10000000, 'DOT': 10000000},
    )
    test_cex = cex.copy()
    test_agent = agent.copy()
    sell_spot = cex.sell_spot('DOT', 'USD')
    test_cex.swap(
        tkn_sell='USD',
        tkn_buy='DOT',
        sell_quantity=1,
        agent=test_agent
    )
    ex_price = (
            (test_agent.initial_holdings['USD'] - test_agent.holdings['USD'])
            / (test_agent.holdings['DOT'] - test_agent.initial_holdings['DOT'])
    )
    if sell_spot != pytest.approx(ex_price):
        raise AssertionError('sell spot gave incorrect price')

    test_agent = agent.copy()
    sell_spot = cex.sell_spot('USD', 'DOT')
    test_cex.swap(
        tkn_sell='DOT',
        tkn_buy='USD',
        sell_quantity=1,
        agent=test_agent
    )
    ex_price = (
            (test_agent.initial_holdings['DOT'] - test_agent.holdings['DOT'])
            / (test_agent.holdings['USD'] - test_agent.initial_holdings['USD'])
    )
    if sell_spot != pytest.approx(ex_price):
        raise AssertionError('sell spot gave incorrect price')
