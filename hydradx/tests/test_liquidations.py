import pytest

from hydradx.model.amm.liquidations import CDP, liquidate_cdp
from hydradx.model.amm.agents import Agent


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
