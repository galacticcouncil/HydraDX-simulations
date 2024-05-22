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


def test_external_oracle():
    state = GlobalState(
        agents={},
        pools={},
        external_oracle={('ETH', 'USD'): 3000, ('BTC', 'USD'): 60000, ('BTC', 'ETH'): 21}
    )

    assert state.external_oracle[('ETH', 'USD')] == 3000
    assert state.external_oracle[('BTC', 'USD')] == 60000
    assert state.external_oracle[('BTC', 'ETH')] == 21


def test_adding_usd_to_external_market():
    state = GlobalState(agents={}, pools={}, external_market={'USD': 1, 'DOT': 10})
    state2 = GlobalState(agents={}, pools={}, external_market={'DOT': 10})
    assert state.external_market == state2.external_market
