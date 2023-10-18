from hypothesis import given, strategies as st
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
import pytest


@st.composite
def order_book_strategy(draw, base_price: float = 1000):
    price_increment = draw(st.floats(min_value=0.000001, max_value=0.1))
    price_points = draw(st.integers(min_value=1, max_value=10))
    book_depth = draw(st.integers(min_value=1, max_value=1000))
    return OrderBook(
        bids=[[base_price - i * price_increment, book_depth / price_points] for i in range(price_points)],
        asks=[[base_price + i * price_increment, book_depth / price_points] for i in range(price_points)]
    )


initial_agent = Agent(
    holdings={
        'DAI': 1000000,
        'ETH': 1000
    }
)


@given(
    quantity=st.floats(min_value=100, max_value=1000000),
    trade_fee=st.floats(min_value=0, max_value=0.1),
    order_book=order_book_strategy()
)
def test_sell_quote(quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                }, trade_fee=trade_fee
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
        sell_quantity=quantity
    )

    value_bought = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    ) - sum(
        [quantity * price for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - sell_state.agents['agent'].holdings[tkn_sell]

    if value_bought != pytest.approx(quantity_sold):
        raise AssertionError('Central market sell trade failed to execute correctly.')

    quantity_bought = sell_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    value_sold = sum(
        [quantity for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    ) - sum(
        [quantity for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    )

    if value_sold * (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_bought):
        raise AssertionError('Fee was not applied correctly.')


@given(
    quantity=st.floats(min_value=0.1, max_value=1000),
    trade_fee=st.floats(min_value=0, max_value=0.1),
    order_book=order_book_strategy()
)
def test_sell_base(quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                }, trade_fee=trade_fee
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
        sell_quantity=quantity
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    value_bought = sum(
        [quantity for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    ) - sum(
        [quantity for (price, quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    )

    if value_bought != pytest.approx(quantity_sold):
        raise AssertionError('Central market sell trade failed to execute correctly.')

    value_sold = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    ) - sum(
        [quantity * price for (price, quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if value_sold * (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_bought):
        raise AssertionError('Fee was not applied correctly.')


@given(
    quantity=st.floats(min_value=100, max_value=1000000),
    trade_fee=st.floats(min_value=0, max_value=0.1),
    order_book=order_book_strategy()
)
def test_buy_quote(quantity: float, trade_fee: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                }, trade_fee=trade_fee
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
        buy_quantity=quantity
    )

    value_sold = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    ) - sum(
        [quantity * price for (price, quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if value_sold != pytest.approx(quantity_bought):
        raise AssertionError('Central market buy trade failed to execute correctly.')

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    value_bought = sum(
        [quantity for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    ) - sum(
        [quantity for (price, quantity) in buy_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].asks]
    )

    if value_bought / (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_sold):
        raise AssertionError('Fee was not applied correctly.')


@given(
    quantity=st.floats(min_value=0.1, max_value=1000),
    order_book=order_book_strategy()
)
def test_buy_base(quantity: float, order_book: OrderBook):
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): order_book
                },
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
        buy_quantity=quantity
    )

    quantity_bought = sell_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    value_sold = sum(
        [quantity for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    ) - sum(
        [quantity for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    )

    if value_sold != pytest.approx(quantity_bought):
        raise AssertionError('Central market buy trade failed to execute correctly.')

    value_bought = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    ) - sum(
        [quantity * price for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - sell_state.agents['agent'].holdings[tkn_sell]

    if value_bought / (1 - initial_state.pools['Kraken'].trade_fee) != pytest.approx(quantity_sold):
        raise AssertionError('Fee was not applied correctly.')


def test_price():
    initial_state = GlobalState(
        pools={
            'Kraken': CentralizedMarket(
                order_book={
                    ('ETH', 'DAI'): OrderBook(
                        bids=[[1000, 1], [999, 1]],
                        asks=[[1001, 1], [1002, 1]]
                    )
                },
            )
        },
        agents={'agent': initial_agent}
    )
    eth_price = initial_state.pools['Kraken'].price('ETH', 'DAI')
    if eth_price != 1000:
        raise AssertionError('External market price incorrect.')

    dai_price = initial_state.pools['Kraken'].price('DAI', 'ETH')
    if dai_price != 1/1001:
        raise AssertionError('External market price incorrect.')
