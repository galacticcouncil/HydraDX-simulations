import math

import pytest
from hypothesis import given, strategies as st

from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, value_assets, cash_out_omnipool, dynamicadd_asset_fee
from hydradx.tests.strategies_omnipool import reasonable_market_dict, omnipool_reasonable_config, reasonable_holdings
from hydradx.tests.strategies_omnipool import reasonable_pct, asset_number_strategy


def test_omnipool_constructor_dynamic_fee_dict_works():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000 / .05, 'LRNA': 1000000 / 20},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000 / 20},
            'DOT': {'liquidity': 1000000 / 5, 'LRNA': 1000000 / 20},
        },
        oracles={'fee_raise': 50},  # , 'fee_lower': 7200},
        lrna_fee=0.0005,
        asset_fee={
            'HDX': 0.0030,
            'USD': dynamicadd_asset_fee(
                minimum=0.0031,
                amplification=0.2,
                raise_oracle_name='fee_raise',
                decay=0.00005,
                fee_max=0.4,
            ),
            'DOT': dynamicadd_asset_fee(
                minimum=0.0032,
                amplification=0.2,
                raise_oracle_name='fee_raise',
                decay=0.00005,
                fee_max=0.4,
            ),
        },
    )

    assert omnipool.last_lrna_fee['HDX'] == 0.0
    assert omnipool.last_lrna_fee['USD'] == 0.0
    assert omnipool.last_lrna_fee['DOT'] == 0.0
    assert omnipool.last_fee['HDX'] == 0.0
    assert omnipool.last_fee['USD'] == 0.0
    assert omnipool.last_fee['DOT'] == 0.0

    assert omnipool.lrna_fee['HDX'].compute() == 0.0005
    assert omnipool.lrna_fee['USD'].compute() == 0.0005
    assert omnipool.lrna_fee['DOT'].compute() == 0.0005
    assert omnipool.asset_fee['HDX'].compute() == 0.0030
    assert omnipool.asset_fee['USD'].compute() == 0.0031
    assert omnipool.asset_fee['DOT'].compute() == 0.0032


def test_omnipool_constructor_last_fee_works():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000 / .05, 'LRNA': 1000000 / 20},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000 / 20},
            'DOT': {'liquidity': 1000000 / 5, 'LRNA': 1000000 / 20},
        },
        oracles={'fee_raise': 50},
        lrna_fee=0.0005,
        asset_fee={
            'HDX': 0.0030,
            'USD': dynamicadd_asset_fee(
                minimum=0.0031,
                amplification=0.2,
                raise_oracle_name='fee_raise',
                decay=0.00005,
                fee_max=0.4,
            ),
            'DOT': dynamicadd_asset_fee(
                minimum=0.0032,
                amplification=0.2,
                raise_oracle_name='fee_raise',
                decay=0.00005,
                fee_max=0.4,
            ),
        },
        last_lrna_fee=0.0005,
        last_asset_fee={
            'HDX': 0.0035,
            'USD': 0.0036,
            'DOT': 0.0037,
        },
    )

    assert omnipool.last_lrna_fee['HDX'] == 0.0005
    assert omnipool.last_lrna_fee['USD'] == 0.0005
    assert omnipool.last_lrna_fee['DOT'] == 0.0005
    assert omnipool.last_fee['HDX'] == 0.0035
    assert omnipool.last_fee['USD'] == 0.0036
    assert omnipool.last_fee['DOT'] == 0.0037


def test_constructor_oracle_from_block_works():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000 / .05, 'LRNA': 1000000 / 20},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000 / 20},
            'DOT': {'liquidity': 1000000 / 5, 'LRNA': 1000000 / 20},
        },
        oracles={'fee_raise': 50},
        lrna_fee=0.0005,
        asset_fee=0.0025,
    )

    assert omnipool.oracles['fee_raise'].liquidity['HDX'] == pytest.approx(1000000 / .05, rel=1e-10)
    assert omnipool.oracles['fee_raise'].price['HDX'] == pytest.approx(0.05 / 20, rel=1e-10)
    assert omnipool.oracles['fee_raise'].volume_in['HDX'] == 0.0
    assert omnipool.oracles['fee_raise'].volume_out['HDX'] == 0.0


def test_constructor_last_oracle_values_works():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000 / .05, 'LRNA': 1000000 / 20},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000 / 20},
            'DOT': {'liquidity': 1000000 / 5, 'LRNA': 1000000 / 20},
        },
        oracles={'fee_raise': 50, 'test2': 100},
        lrna_fee=0.0005,
        asset_fee=0.0025,
        last_oracle_values={
            'fee_raise': {
                'liquidity': {'HDX': 5000000, 'USD': 500000, 'DOT': 100000},
                'volume_in': {'HDX': 10000, 'USD': 10000, 'DOT': 10000},
                'volume_out': {'HDX': 10000, 'USD': 10000, 'DOT': 10000},
                'price': {'HDX': 0.05, 'USD': 1, 'DOT': 5},
            },
            'test2': {
                'liquidity': {'HDX': 5000000 * 1.1, 'USD': 500000 * 1.1, 'DOT': 100000 * 1.1},
                'volume_in': {'HDX': 10000 * 1.1, 'USD': 10000 * 1.1, 'DOT': 10000 * 1.1},
                'volume_out': {'HDX': 10000 * 1.1, 'USD': 10000 * 1.1, 'DOT': 10000 * 1.1},
                'price': {'HDX': 0.05 * 1.1, 'USD': 1 * 1.1, 'DOT': 5 * 1.1},
            }
        }
    )

    assert omnipool.oracles['fee_raise'].liquidity['HDX'] == 5000000
    assert omnipool.oracles['test2'].volume_in['USD'] == 10000 * 1.1
    assert omnipool.oracles['fee_raise'].volume_out['DOT'] == 10000
    assert omnipool.oracles['test2'].price['HDX'] == 0.05 * 1.1


@given(reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_value_assets(market: dict, holdings: list):
    asset_list = list(market.keys())
    assets = {asset_list[i]: holdings[i] for i in range(5)}
    value = value_assets(market, assets)
    if value != sum([holdings[i] * market[asset_list[i]] for i in range(5)]):
        raise


@given(omnipool_reasonable_config(token_count=5), reasonable_market_dict(token_count=5),
       reasonable_holdings(token_count=5))
def test_cash_out_no_liquidity(omnipool: OmnipoolState, market: dict, holdings: list):
    asset_list = list(market.keys())
    holdings_dict = {tkn: holdings[i] for i, tkn in enumerate(asset_list)}
    agent = Agent(holdings=holdings_dict, )
    cash = cash_out_omnipool(omnipool, agent, market)
    if cash != sum([holdings_dict[tkn] * market[tkn] for tkn in asset_list]):
        raise AssertionError('Cash out failed: {} != {}'.format(
            cash, sum([holdings_dict[tkn] * market[tkn] for tkn in asset_list])))


@given(omnipool_reasonable_config(token_count=5), reasonable_pct(token_count=5))
def test_cash_out_only_liquidity_at_spot(omnipool: OmnipoolState, pct_list: list):
    asset_list = omnipool.asset_list
    holdings = {
        (omnipool.unique_id, tkn): omnipool.liquidity[tkn] * pct_list[i] for i, tkn in enumerate(asset_list)
    }
    market = {tkn: oamm.usd_price(omnipool, tkn) for tkn in asset_list}
    agent = Agent(holdings=holdings,
                  share_prices={(omnipool.unique_id, tkn): oamm.price(omnipool, tkn) for tkn in asset_list})
    cash = cash_out_omnipool(omnipool, agent, market)
    min_withdrawal_fee = 0.0001
    res = sum([pct_list[i] * omnipool.liquidity[tkn] * market[tkn] * (1 - min_withdrawal_fee) for i, tkn in
               enumerate(asset_list)])
    if cash != pytest.approx(res, 10e-10):
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

    initial_price = oamm.price(omnipool, held_asset)
    initial_usd_price = oamm.price(omnipool, omnipool.stablecoin)

    trader = Agent(holdings=trader_holdings)
    lp_agent = Agent(holdings=lp_holdings)
    usdlp_agent = Agent(holdings=usdlp_holdings)

    oamm.execute_add_liquidity(omnipool, lp_agent, lp_agent.holdings[held_asset], held_asset)
    oamm.execute_swap(omnipool, trader, "HDX", held_asset, sell_quantity=trade_size)

    market = {tkn: oamm.usd_price(omnipool, tkn) for tkn in asset_list}
    cash = cash_out_omnipool(omnipool, lp_agent, market)
    cash_usdlp = cash_out_omnipool(omnipool, usdlp_agent, market)

    final_price = oamm.price(omnipool, held_asset)
    final_usd_price = oamm.price(omnipool, omnipool.stablecoin)

    # change ratio for TKN price denominated in LRNA
    k = final_price / initial_price
    k_usd = final_usd_price / initial_usd_price

    # xyk pool IL formula * initial assets LPed
    value_target = 2 * math.sqrt(k) / (k + 1) * initial_lp
    usd_price = oamm.usd_price(omnipool, held_asset)  # Need to convert from USD to TKN
    # if cash / usd_price != pytest.approx(value_target, 1e-12):
    #     raise

    if cash / usd_price > value_target:
        raise

    usd_value_target = 2 * math.sqrt(k_usd) / (k_usd + 1) * initial_usd_lp
    # if cash_usdlp != pytest.approx(usd_value_target, 1e-12):
    #     raise

    if cash_usdlp > usd_value_target:
        raise


@given(omnipool_reasonable_config())
def test_cash_out_lrna(omnipool: OmnipoolState):
    initial_lrna = 1000
    agent = Agent(holdings={'LRNA': initial_lrna})
    market = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}
    cash = cash_out_omnipool(omnipool, agent, market)
    if cash == 0:
        raise ValueError("Cash out should not be zero")
    # minus slippage, cash should be 1 / lrna_price('USD') * initial_lrna
    if cash > 1 / oamm.lrna_price(omnipool, 'USD') * 1000:
        raise ValueError("Cash out should not be greater than initial value")

    omnipool.add_token(
        tkn='newcoin',
        liquidity=1000,
        lrna=1000,
        shares=1000,
        protocol_shares=1000
    )

    market['newcoin'] = oamm.usd_price(omnipool, 'newcoin')

    if cash_out_omnipool(omnipool, agent, market) <= cash:
        raise ValueError("Cash out should be higher after adding new token")


@given(
    omnipool_reasonable_config(),
    st.floats(min_value=0.01, max_value=2),
    st.integers(min_value=1, max_value=99)
)
def test_cash_out_accuracy(omnipool: oamm.OmnipoolState, share_price_ratio, lp_index):
    lp_asset = omnipool.asset_list[lp_index % len(omnipool.asset_list)]
    agent = Agent(
        holdings={('omnipool', tkn): omnipool.shares[tkn] / 10 for tkn in omnipool.asset_list}
    )
    agent.holdings.update({tkn: 1 for tkn in omnipool.asset_list})
    for tkn in omnipool.asset_list:
        agent.share_prices[('omnipool', tkn)] = oamm.lrna_price(omnipool, tkn) * share_price_ratio

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}
    cash_out = oamm.cash_out_omnipool(omnipool, agent, market_prices)

    withdraw_state, withdraw_agent = omnipool.copy(), agent.copy()
    for tkn in omnipool.asset_list:
        withdraw_state, withdraw_agent = oamm.execute_remove_liquidity(
            state=withdraw_state,
            agent=withdraw_agent,
            tkn_remove=tkn,
            quantity=withdraw_agent.holdings[('omnipool', tkn)]
        )
        del withdraw_agent.holdings[('omnipool', tkn)]

    if 'LRNA' in withdraw_agent.holdings:
        lrna_sells = {
            tkn: withdraw_agent.holdings['LRNA'] * withdraw_state.lrna[tkn] / withdraw_state.lrna_total
            for tkn in omnipool.asset_list
        }
        lrna_profits = dict()
        for tkn, delta_q in lrna_sells.items():
            agent_holdings = withdraw_agent.holdings[tkn]
            oamm.execute_swap(
                state=withdraw_state,
                agent=withdraw_agent,
                tkn_sell='LRNA',
                tkn_buy=tkn,
                sell_quantity=delta_q
            )
            lrna_profits[tkn] = withdraw_agent.holdings[tkn] - agent_holdings

        er = 1

    del withdraw_agent.holdings['LRNA']
    # del withdraw_agent.holdings[('omnipool', lp_asset)]
    cash_count = sum([market_prices[tkn] * withdraw_agent.holdings[tkn] for tkn in withdraw_agent.holdings])
    if cash_count != pytest.approx(cash_out, rel=1e-15):
        raise AssertionError('Cash out calculation is not accurate.')
