import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity

from hydradx.model.amm.liquidations import CDP, liquidate_cdp
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.global_state import find_partial_liquidation_amount


def test_liquidate_cdp():

    liquidate_pct = 0.7
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 200
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, init_collat_amt, True)
    init_debt_holdings = 10000
    init_collat_holdings = 10000
    agent = Agent(holdings={"USDT": init_debt_holdings, "DOT": init_collat_holdings})

    liquidate_cdp(cdp, agent, init_debt_amt*liquidate_pct, init_collat_amt*liquidate_pct)
    if agent.holdings[collateral_asset] + cdp.collateral_amt != init_collat_holdings + init_collat_amt:
        raise
    if agent.holdings[debt_asset] - cdp.debt_amt != init_debt_holdings - init_debt_amt:
        raise
    if cdp.debt_amt != pytest.approx(init_debt_amt * (1 - liquidate_pct)):
        raise
    if cdp.collateral_amt != pytest.approx(init_collat_amt * (1 - liquidate_pct)):
        raise


def test_liquidate_cdp_fails():

    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 200
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, init_collat_amt, True)
    init_debt_holdings = 10000
    init_collat_holdings = 10000
    agent = Agent(holdings={"USDT": init_debt_holdings, "DOT": init_collat_holdings})

    with pytest.raises(Exception):
        liquidate_cdp(cdp, agent, init_debt_amt*1.2, init_collat_amt*0.7)
    with pytest.raises(Exception):
        liquidate_cdp(cdp, agent, init_debt_amt*0.7, init_collat_amt*1.2)
    liquidate_cdp(cdp, agent, init_debt_amt * 0.7, init_collat_amt * 0.7)


@given(st.floats(min_value=4, max_value=6))
def test_find_partial_liquidation_amount_full(collat_ratio: float):
    prices = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}

    assets = {
        'DOT': {'usd price': prices['DOT'], 'weight': 0.40},
        'HDX': {'usd price': prices['HDX'], 'weight': 0.10},
        'USDT': {'usd price': prices['USDT'], 'weight': 0.30},
        'WETH': {'usd price': prices['WETH'], 'weight': 0.10},
        'iBTC': {'usd price': prices['iBTC'], 'weight': 0.10}
    }

    lrna_price_usd = 35
    initial_omnipool_tvl = 20000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    omnipool = OmnipoolState(
            tokens={
                tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
            },
            preferred_stablecoin='USDT',
        )

    collateral_amt = 200
    debt_amt = collat_ratio * collateral_amt
    cdp = CDP('USDT', 'DOT', debt_amt, collateral_amt, True)
    penalty = 0.01

    liquidation_amount = find_partial_liquidation_amount(omnipool, cdp, penalty)

    if liquidation_amount != pytest.approx(debt_amt * (1 + penalty)):
        raise


@given(st.floats(min_value=6, max_value=6.9))
def test_find_partial_liquidation_amount_partial(collat_ratio: float):
    prices = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}

    assets = {
        'DOT': {'usd price': prices['DOT'], 'weight': 0.40},
        'HDX': {'usd price': prices['HDX'], 'weight': 0.10},
        'USDT': {'usd price': prices['USDT'], 'weight': 0.30},
        'WETH': {'usd price': prices['WETH'], 'weight': 0.10},
        'iBTC': {'usd price': prices['iBTC'], 'weight': 0.10}
    }

    lrna_price_usd = 35
    initial_omnipool_tvl = 20000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    omnipool = OmnipoolState(
            tokens={
                tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
            },
            preferred_stablecoin='USDT',
        )

    collateral_amt = 200000
    debt_amt = collat_ratio * collateral_amt
    cdp = CDP('USDT', 'DOT', debt_amt, collateral_amt, True)
    penalty = 0.01

    liquidation_amount = find_partial_liquidation_amount(omnipool, cdp, penalty, 100)

    if liquidation_amount == pytest.approx(debt_amt * (1 + penalty)):
        raise

    agent = Agent(holdings={"USDT": 0, "DOT": collateral_amt})
    omnipool.swap(agent, tkn_buy="USDT", tkn_sell="DOT", buy_quantity=liquidation_amount)

    # exec_price = agent.holdings["USDT"] * (1 - penalty) / (collateral_amt - agent.holdings["DOT"])
    exec_price = agent.holdings["USDT"] / (collateral_amt - agent.holdings["DOT"]) / (1 + penalty)

    if exec_price != pytest.approx(collat_ratio):
        raise
