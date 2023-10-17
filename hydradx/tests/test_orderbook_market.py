from hypothesis import given, strategies as st
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
import pytest

initial_state = GlobalState(
    pools={
        'Kraken': CentralizedMarket(
            order_book={
                ('ETH', 'DAI'): OrderBook(
                    bids=[[1000, 50], [999, 75], [998, 100], [997, 200]],
                    asks=[[1000, 50], [1001, 75], [1002, 100], [1003, 200]]
                )
            }
        )
    },
    agents={
        'agent': Agent(
            holdings={
                'DAI': 1000000,
                'ETH': 1000
            }
        )
    },
)


@given(
    quantity=st.floats(min_value=100, max_value=1000000)
)
def test_sell_quote(quantity: float):
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
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    ) - sum(
        [quantity * price for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_sell, tkn_buy)].bids]
    )

    if value_bought != pytest.approx(quantity):
        raise AssertionError('Central market sell trade failed to execute correctly.')


@given(
    quantity=st.floats(min_value=0.1, max_value=1000)
)
def test_sell_base(quantity: float):
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

    value_bought = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    ) - sum(
        [quantity * price for (price, quantity) in buy_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].asks]
    )

    quantity_bought = buy_state.agents['agent'].holdings[tkn_buy] - initial_state.agents['agent'].holdings[tkn_buy]

    if value_bought != pytest.approx(quantity_bought):
        raise AssertionError('External market buy trade failed to execute correctly.')


@given(
    quantity=st.floats(min_value=100, max_value=1000000)
)
def test_buy_quote(quantity: float):
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

    if value_sold != pytest.approx(quantity):
        raise AssertionError('External market buy trade failed to execute correctly.')


@given(
    quantity=st.floats(min_value=0.1, max_value=1000)
)
def test_buy_base(quantity: float):
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

    if quantity > 225:
        er = 1

    value_sold = sum(
        [quantity * price for (price, quantity) in initial_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    ) - sum(
        [quantity * price for (price, quantity) in sell_state.pools['Kraken'].order_book[(tkn_buy, tkn_sell)].bids]
    )

    quantity_sold = initial_state.agents['agent'].holdings[tkn_sell] - sell_state.agents['agent'].holdings[tkn_sell]

    if value_sold != pytest.approx(quantity_sold):
        raise AssertionError('External market sell trade failed to execute correctly.')