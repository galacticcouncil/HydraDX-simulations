import pytest

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.otc import OTC


def test_sell():
    # price is USDT/DOT
    otc = OTC('DOT', 'USDT', 100, 7)
    agent = Agent(holdings={"USDT": 1000, "DOT": 100})
    otc.sell(agent, 10)  # should sell 10 DOT for 70 USDT
    if otc.buy_amount != 10:
        raise
    if otc.sell_amount != 30:
        raise
    if agent.holdings["USDT"] != 1070:
        raise
    if agent.holdings["DOT"] != 90:
        raise


def test_sell_fails():
    otc = OTC('DOT', 'USDT', 100, 7)
    agent = Agent(holdings={"USDT": 1000, "DOT": 100})
    with pytest.raises(Exception):
        otc.sell(agent, 100)  # should fail, too big
    with pytest.raises(Exception):
        otc.sell(agent, -1)
