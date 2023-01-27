import pytest, random, copy
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, value_assets, cash_out_omnipool
from hydradx.tests.strategies_omnipool import reasonable_market_dict, omnipool_reasonable_config, reasonable_holdings
from hydradx.tests.strategies_omnipool import reasonable_pct


@given(reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_value_assets(market: dict, holdings: list):
    asset_list = list(market.keys())
    assets = {asset_list[i]: holdings[i] for i in range(5)}
    value = value_assets(market, assets)
    if value != sum([holdings[i] * market[asset_list[i]] for i in range(5)]):
        raise


@given(omnipool_reasonable_config(token_count=5), reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_cash_out_no_liquidity(omnipool: OmnipoolState, market: dict, holdings: list):
    asset_list = list(market.keys())
    holdings_dict = {tkn: holdings[i] for i, tkn in enumerate(asset_list)}
    agent = Agent(holdings=holdings_dict)
    cash = cash_out_omnipool(omnipool, agent, market)
    if cash != sum([holdings_dict[tkn] * market[tkn] for tkn in asset_list]):
        raise


@given(omnipool_reasonable_config(token_count=5), reasonable_pct(token_count=5))
def test_cash_out_only_liquidity_at_spot(omnipool: OmnipoolState, pct_list: list):
    asset_list = omnipool.asset_list
    holdings = {
        (omnipool.unique_id, tkn): omnipool.liquidity[tkn] * pct_list[i] for i, tkn in enumerate(asset_list)
    }
    market = {tkn: omnipool.usd_price(tkn) for tkn in asset_list}
    agent = Agent(holdings=holdings, share_prices={(omnipool.unique_id, tkn): omnipool.price(tkn) for tkn in asset_list})
    cash = cash_out_omnipool(omnipool, agent, market)
    if cash != sum([pct_list[i] * omnipool.liquidity[tkn] * market[tkn] for i, tkn in enumerate(asset_list)]):
        raise