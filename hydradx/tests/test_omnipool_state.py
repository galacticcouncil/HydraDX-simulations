import math

import pytest, random, copy
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, value_assets, cash_out_omnipool
from hydradx.tests.strategies_omnipool import reasonable_market_dict, omnipool_reasonable_config, reasonable_holdings
from hydradx.tests.strategies_omnipool import reasonable_pct, asset_number_strategy


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


@given(omnipool_reasonable_config(token_count=5), reasonable_pct(token_count=2), asset_number_strategy)
def test_cash_out_one_asset_only_liquidity(omnipool: OmnipoolState, pct_list: list, trade_size_denom: int):
    asset_list = omnipool.asset_list
    held_asset = None
    for asset in asset_list:
        if asset in ['HDX', 'LRNA', omnipool.stablecoin]:
            continue
        held_asset = asset
        break
    if held_asset is None:
        raise

    initial_lp = omnipool.liquidity[held_asset] * pct_list[0]
    initial_usd_lp = omnipool.liquidity[omnipool.stablecoin] * pct_list[1]
    lp_holdings = {held_asset: initial_lp}
    usdlp_holdings = {omnipool.stablecoin: initial_usd_lp}
    trade_size = omnipool.liquidity[held_asset] / trade_size_denom
    usd_trade_size = omnipool.liquidity[omnipool.stablecoin] / trade_size_denom
    trader_holdings = {held_asset: trade_size, omnipool.stablecoin: usd_trade_size}

    initial_price = omnipool.price(held_asset)
    initial_usd_price = omnipool.price(omnipool.stablecoin)

    trader = Agent(holdings=trader_holdings)
    lp_agent = Agent(holdings=lp_holdings)
    usdlp_agent = Agent(holdings=usdlp_holdings)

    oamm.execute_add_liquidity(omnipool, lp_agent, lp_agent.holdings[held_asset], held_asset)
    oamm.execute_swap(omnipool, trader, "HDX", held_asset, sell_quantity=trade_size)

    market = {tkn: omnipool.usd_price(tkn) for tkn in asset_list}
    cash = cash_out_omnipool(omnipool, lp_agent, market)
    cash_usdlp = cash_out_omnipool(omnipool, usdlp_agent, market)

    final_price = omnipool.price(held_asset)
    final_usd_price = omnipool.price(omnipool.stablecoin)

    # change ratio for TKN price denominated in LRNA
    k = final_price / initial_price
    k_usd = final_usd_price / initial_usd_price

    # xyk pool IL formula * initial assets LPed
    value_target = 2 * math.sqrt(k) / (k + 1) * initial_lp
    usd_price = omnipool.usd_price(held_asset)  # Need to convert from USD to TKN
    if cash / usd_price != pytest.approx(value_target, 1e-12):
        raise

    usd_value_target = 2 * math.sqrt(k_usd) / (k_usd + 1) * initial_usd_lp
    if cash_usdlp != pytest.approx(usd_value_target, 1e-12):
        raise