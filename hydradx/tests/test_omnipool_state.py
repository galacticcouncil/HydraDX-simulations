import pytest, random, copy
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, value_assets, cash_out_omnipool
from hydradx.tests.strategies_omnipool import reasonable_market_dict, omnipool_reasonable_config, reasonable_holdings


@given(reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_value_assets(market: dict, holdings: list):
    asset_list = list(market.keys())
    assets = {asset_list[i]: holdings[i] for i in range(5)}
    value = value_assets(market, assets)
    if value != sum([holdings[i] * market[asset_list[i]] for i in range(5)]):
        raise


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), reasonable_market_dict(token_count=3))
def test_cash_out_usd_omnipool(omnipool: OmnipoolState, market: dict):
    usd_amount = 100000
    holdings = {omnipool.stablecoin: usd_amount}
    agent = Agent(holdings=holdings)
    cash = cash_out_omnipool(omnipool, agent, market)
    if cash != usd_amount:
        raise
