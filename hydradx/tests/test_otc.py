import pytest
from mpmath import mp, mpf
from hypothesis import given, strategies as st, assume, settings, Verbosity

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import omnipool_settle_otc, find_partial_otc_sell_amount, GlobalState, \
    settle_otc_against_omnipool
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


def test_buy():
    # price is USDT/DOT
    otc = OTC('DOT', 'USDT', 100, 7)
    agent = Agent(holdings={"USDT": 1000, "DOT": 100})
    otc.buy(agent, 70)  # should buy 70 USDT for 10 DOT
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


def test_buy_fails():
    otc = OTC('DOT', 'USDT', 100, 7)
    agent = Agent(holdings={"USDT": 1000, "DOT": 100})
    with pytest.raises(Exception):
        otc.buy(agent, 110)  # should fail, too big
    with pytest.raises(Exception):
        otc.buy(agent, -1)


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
    sell_amt = 120  # USDT we are selling to the Omnipool
    init_price = 12
    otc = OTC('DOT', 'USDT', init_sell_amt, init_price)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, sell_amt)
    if otc.buy_amount != sell_amt / init_price:
        raise
    if otc.sell_amount != init_sell_amt - sell_amt:
        raise
    # DOT conserved
    if (omnipool.liquidity['DOT'] + otc.buy_amount + treasury.holdings['DOT'] != pytest.approx(
            init_state.liquidity['DOT'], rel=1e-15)):
        raise
    # USDT conserved
    if omnipool.liquidity['USDT'] + treasury.holdings['USDT'] != pytest.approx(init_state.liquidity['USDT'] + init_sell_amt, rel=1e-15):
        raise
    if treasury.holdings['DOT'] <= 0:  # treasury should make profit
        raise
    for asset in treasury.holdings:  # treasury should not lose any asset
        if asset != 'DOT' and treasury.holdings[asset] < 0:
            raise

    # partially execute the OTC order
    omnipool = init_state.copy()
    init_sell_amt = 120
    sell_amt = 5
    init_price = 12
    otc = OTC('DOT', 'USDT', init_sell_amt, init_price)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, sell_amt)
    if otc.buy_amount != sell_amt / init_price:
        raise
    if otc.sell_amount != init_sell_amt - sell_amt:
        raise
    # DOT conserved
    if (omnipool.liquidity['DOT'] + otc.buy_amount + treasury.holdings['DOT'] != pytest.approx(
            init_state.liquidity['DOT'], rel=1e-15)):
        raise
    # USDT conserved
    if (omnipool.liquidity['USDT'] + otc.sell_amount + treasury.holdings['USDT'] != pytest.approx(init_state.liquidity['USDT'] + init_sell_amt,
                                                                      rel=1e-15)):
        raise
    if treasury.holdings['DOT'] <= 0:  # treasury should make profit
        raise
    for asset in treasury.holdings:  # treasury should not lose any asset
        if asset != 'DOT' and treasury.holdings[asset] < 0:
            raise

    # try amount too large for OTC order
    omnipool = init_state.copy()
    otc = OTC('DOT', 'USDT', 120, 12)
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    with pytest.raises(Exception):
        omnipool_settle_otc(omnipool, otc, treasury, 1000)


def test_find_partial_otc_sell_amount():
    # find_partial_otc_amount(omnipool, otc)
    # returns suggested amount to buy from Omnipool, sell into OTC order
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

    # OTC not partially fillable, but can be fully filled
    otc = OTC('DOT', 'USDT', 120, 12, partially_fillable=False)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    if partial_otc_amount != 120:
        raise

    # OTC not partially fillable, not fillable due to fees
    otc = OTC('DOT', 'USDT', 120, 10.01, partially_fillable=False)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    if partial_otc_amount != 0:
        raise

    # OTC partially fillable, fully fillable at spot price, but fees make it not fillable
    otc = OTC('DOT', 'USDT', 120, 10.01, partially_fillable=True)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    if partial_otc_amount != 0:
        raise

    # OTC partially fillable
    otc = OTC('DOT', 'USDT', 120, 10.04, partially_fillable=True)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    if partial_otc_amount == 0:
        raise
    omnipool = init_state.copy()
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, mpf(partial_otc_amount))
    spot_after = omnipool.buy_spot('DOT', 'USDT')
    if spot_after > 10.04:  # OTC is only arbed against Omnipool up to profitability
        raise
    if (10.04 - spot_after)/10.04 > 1e-6:  # Very little arb profit left between OTC, Omnpool
        raise


def test_find_partial_otc_amount_slippage_prevents_fully_filling_order():
    tokens = {
        "HDX": {'liquidity': mpf(10000), 'LRNA': mpf(100000)},
        "USDT": {'liquidity': mpf(1000), 'LRNA': mpf(100000)},
        "DOT": {'liquidity': mpf(100), 'LRNA': mpf(100000)},
    }

    initial_state = OmnipoolState(
        tokens=tokens,
        asset_fee=0.0,
        lrna_fee=0.0,
        preferred_stablecoin='USDT'
    )

    # OTC not partially fillable, could be partially but not fully filled
    otc = OTC('DOT', 'USDT', 120, 10.001, partially_fillable=False)
    partial_otc_amount = find_partial_otc_sell_amount(initial_state, otc)
    if partial_otc_amount != 0:
        raise

    # OTC partially fillable
    otc = OTC('DOT', 'USDT', 120, 10.001, partially_fillable=True)
    partial_otc_amount = find_partial_otc_sell_amount(initial_state, otc)
    if partial_otc_amount == 0:
        raise
    omnipool = initial_state.copy()
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, partial_otc_amount)
    spot_after = omnipool.buy_spot('DOT', 'USDT')
    if spot_after > 10.001:  # OTC is only arbed against Omnipool up to profitability
        raise
    if (10.001 - spot_after)/10.001 > 1e-6:  # Very little arb profit left between OTC, Omnpool
        raise


@given(
    st.floats(min_value=8.0, max_value=12.0),
    st.floats(min_value=1.0, max_value=10000.0)
)
def test_find_partial_otc_sell_amount_fuzz(price, sell_amount):
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
    spot_before = init_state.buy_spot('DOT', 'USDT')

    # OTC partially fillable
    otc = OTC('DOT', 'USDT', sell_amount, price, partially_fillable=True)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    omnipool = init_state.copy()
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, partial_otc_amount)
    spot_after = omnipool.buy_spot('DOT', 'USDT')

    partially_filled = False
    # if OTC order is not filled at all, initial spot price > OTC price
    # if partial_otc_amount == 0 and (otc.price - spot_before)/otc.price > 1e-6:
    if partial_otc_amount == 0 and spot_before < otc.price:
        raise
    # if OTC order is fully filled, final spot price < OTC price
    if otc.sell_amount == 0 and spot_after > otc.price:
        raise
    # if OTC order is partially filled, final spot price should be close to OTC price, and below it
    if otc.sell_amount != 0 and otc.buy_amount != 0:
        partially_filled = True
        if spot_after > otc.price:
            raise
        if (otc.price - spot_after)/otc.price > 1e-6:
            raise

    # OTC not partially fillable
    otc = OTC('DOT', 'USDT', sell_amount, price, partially_fillable=False)
    partial_otc_amount = find_partial_otc_sell_amount(init_state, otc)
    omnipool = init_state.copy()
    treasury = Agent(holdings={"USDT": 0, "DOT": 0})
    omnipool_settle_otc(omnipool, otc, treasury, partial_otc_amount)
    spot_after = omnipool.buy_spot('DOT', 'USDT')

    # if OTC order is fully filled, final spot price < OTC price
    if otc.sell_amount == 0 and spot_after > otc.price:
        raise
    # if OTC order was partially filled when that was possible, now it is not filled at all
    if partially_filled and partial_otc_amount != 0:
        raise


def test_settle_otc_against_omnipool():
    # settle_otc_against_omnipool(pool_id: str, agent_id: str)
    # returns transform(state: GlobalState) -> GlobalState
    # this transform function is how GlobalState is evolved

    tokens = {
        "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(100000)},
        "USDT": {'liquidity': mpf(100000), 'LRNA': mpf(100000)},
        "DOT": {'liquidity': mpf(10000), 'LRNA': mpf(100000)},
    }
    init_op = OmnipoolState(
        tokens=tokens,
        asset_fee=0.0025,
        lrna_fee=0.0005,
        preferred_stablecoin='USDT'
    )
    otcs = [
        OTC('DOT', 'USDT', 120, 12, partially_fillable=True),
        OTC('HDX', 'USDT', 100, 0.12, partially_fillable=True),
        OTC('HDX', 'DOT', 120, .008, partially_fillable=True),
    ]
    agents = {"treasury": Agent(holdings={"USDT": 0, "DOT": 0})}
    init_state = GlobalState(agents=agents, pools={'omnipool': init_op}, otcs=otcs)
    state = init_state.copy()

    transform_fn = settle_otc_against_omnipool("omnipool", "treasury")
    transform_fn(state)

    for otc in state.otcs:
        if otc.sell_amount == 0:  # these OTCs should have been removed from state
            raise
        if otc.partially_fillable and otc.price > state.pools['omnipool'].buy_spot(otc.sell_asset, otc.buy_asset):
            raise  # these OTCs should have been partially filled
