import pytest
from mpmath import mp, mpf

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import omnipool_settle_otc
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.otc import OTC

mp.dps = 50


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


def test_omnipool_settle_otc():
    tokens = {
        "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(100000)},
        "USDT": {'liquidity': mpf(100000), 'LRNA': mpf(100000)},
        "DOT": {'liquidity': mpf(10000), 'LRNA': mpf(100000)},
    }

    init_state = OmnipoolState(
        tokens=tokens,
        asset_fee=0.0025,
        lrna_fee=0.0005,
        preferred_stablecoin='USDT'
    )

    # fully execute the OTC order
    omnipool = init_state.copy()
    init_sell_amt = 120
    sell_amt = 10
    init_price = 12
    otc = OTC('DOT', 'USDT', init_sell_amt, init_price)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, sell_amt)
    if otc.buy_amount != sell_amt:
        raise
    if otc.sell_amount != init_sell_amt - init_price * sell_amt:
        raise
    # DOT conserved
    if (omnipool.liquidity['DOT'] + otc.buy_amount + treasury.holdings['DOT'] != pytest.approx(
            init_state.liquidity['DOT'], rel=1e-15)):
        raise
    # USDT conserved
    if (omnipool.liquidity['USDT'] != pytest.approx(init_state.liquidity['USDT'] + init_sell_amt, rel=1e-15)):
        raise
    if treasury.holdings['DOT'] == 0:  # treasury should make profit
        raise
    for asset in treasury.holdings:  # treasury should not lose any asset
        if treasury.holdings[asset] < 0:
            raise

    # partially execute the OTC order
    omnipool = init_state.copy()
    init_sell_amt = 120
    sell_amt = 5
    init_price = 12
    otc = OTC('DOT', 'USDT', init_sell_amt, init_price)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, sell_amt)
    if otc.buy_amount != sell_amt:
        raise
    if otc.sell_amount != init_sell_amt - init_price * sell_amt:
        raise
    # DOT conserved
    if (omnipool.liquidity['DOT'] + otc.buy_amount + treasury.holdings['DOT'] != pytest.approx(
            init_state.liquidity['DOT'], rel=1e-15)):
        raise
    # USDT conserved
    if (omnipool.liquidity['USDT'] + otc.sell_amount != pytest.approx(init_state.liquidity['USDT'] + init_sell_amt,
                                                                      rel=1e-15)):
        raise
    if treasury.holdings['DOT'] == 0:  # treasury should make profit
        raise
    for asset in treasury.holdings:  # treasury should not lose any asset
        if treasury.holdings[asset] < 0:
            raise

    # try amount too large for OTC order
    omnipool = init_state.copy()
    otc = OTC('DOT', 'USDT', 120, 12)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    with pytest.raises(Exception):
        omnipool_settle_otc(omnipool, otc, treasury, 100)
