from hypothesis import given

from hydradx.model.amm.global_state import value_assets, GlobalState
from hydradx.tests.strategies_omnipool import reasonable_market_dict, reasonable_holdings
from hydradx.model.amm.agents import Agent
from hypothesis import strategies as st
import pytest


# def value_assets(prices: dict, assets: dict) -> float:
@given(reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_value_assets(market: dict, holdings: list):
    asset_list = list(market.keys())
    assets = {asset_list[i]: holdings[i] for i in range(5)}
    value = value_assets(market, assets)
    if value != sum([holdings[i] * market[asset_list[i]] for i in range(5)]):
        raise


@given(
    quantity=st.floats(min_value=100, max_value=1000000)
)
def test_external_market_sale(quantity: float):
    initial_state = GlobalState(
        pools={},
        agents={
            'agent': Agent(
                holdings={
                    'DAI': 1000000,
                    'ETH': 1000
                }
            )
        },
        external_market={
            'DAI': {1: 100000, 1.01: 100000, 1.02: 800000},
            'ETH': {1000: 100, 1002: 100, 1004: 800}
        }
    )

    tkn_buy = 'ETH'
    tkn_sell = 'DAI'
    sell_state = initial_state.copy()
    sell_state.external_market_trade(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=quantity,
        agent_id='agent'
    )
    # print(new_state.agents['agent'].holdings)

    tkns_bought = {
        price: initial_state.external_market[tkn_buy][price]
        - sell_state.external_market[tkn_buy][price]
        for price in initial_state.external_market[tkn_buy]
    }

    if sum([quantity * price for (price, quantity) in tkns_bought.items()]) != pytest.approx(quantity):
        raise AssertionError('External market sell trade failed to execute correctly.')

    tkn_buy = 'DAI'
    tkn_sell = 'ETH'
    buy_state = initial_state.copy()
    buy_state.external_market_trade(
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=quantity,
        agent_id='agent'
    )
    # print(new_state.agents['agent'].holdings)

    tkns_bought = {
        price: initial_state.external_market[tkn_buy][price]
        - buy_state.external_market[tkn_buy][price]
        for price in initial_state.external_market[tkn_buy]
    }

    sell_quantity = initial_state.agents['agent'].holdings[tkn_sell] - buy_state.agents['agent'].holdings[tkn_sell]

    if (
            sum([quantity * price for (price, quantity) in tkns_bought.items()])
            != pytest.approx(sell_quantity * buy_state.price(tkn_sell))
    ):
        raise AssertionError('External market buy trade failed to execute correctly.')
