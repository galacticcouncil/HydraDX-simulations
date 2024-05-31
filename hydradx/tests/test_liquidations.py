import pytest
from hypothesis import given, strategies as st, settings
from mpmath import mp, mpf

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import find_partial_liquidation_amount, omnipool_liquidate_cdp, GlobalState, \
    liquidate_against_omnipool, liquidate_against_omnipool_and_settle_otc
from hydradx.model.amm.liquidations import CDP, money_market
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.otc import OTC

mp.dps = 50


# CDP tests

def test_cdp_validate():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 200
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, init_collat_amt)
    if not cdp.validate():
        raise
    cdp.debt_amt = -1
    if cdp.validate():
        raise
    cdp.debt_amt = init_debt_amt
    cdp.collateral_amt = -1
    if cdp.validate():
        raise
    cdp.collateral_amt = init_collat_amt
    cdp.debt_asset = collateral_asset
    if cdp.validate():
        raise

    cdp = CDP(debt_asset, collateral_asset, 0, init_collat_amt)
    if not cdp.validate():
        raise
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, 0)
    if not cdp.validate():  # note that toxic debt does not fail validation
        raise


# money_market tests

def test_get_oracle_price():
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10}
    )
    if mm.get_oracle_price("DOT", "USDT") != 10:
        raise
    if mm.get_oracle_price("USDT", "DOT") != 0.1:
        raise
    with pytest.raises(Exception):
        mm.get_oracle_price("DOT", "ETH")


def test_is_liquidatable():
    agent = Agent()
    cdp = CDP("USDT", "DOT", 2000*0.7-0.00001, 200)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10},
        cdps=[(agent, cdp)],
        liquidation_threshold=0.7
    )
    if mm.is_liquidatable(cdp):
        raise
    cdp.debt_amt = 2000*0.7+0.00001
    if not mm.is_liquidatable(cdp):
        raise


def test_is_fully_liquidatable():
    agent = Agent()
    cdp = CDP("USDT", "DOT", 2000*0.8-0.00001, 200)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10},
        cdps=[(agent, cdp)],
        liquidation_threshold=0.7,
        full_liquidation_threshold=0.8
    )
    if mm.is_fully_liquidatable(cdp):
        raise
    cdp.debt_amt = 2000*0.8+0.00001
    if not mm.is_fully_liquidatable(cdp):
        raise


def test_get_liquidate_collateral_amt():
    debt_amt = 1000
    collateral_amt = 200
    spot_price = 10
    penalty = 0.01
    collat_liquidated_exp = 101
    cdp = CDP("USDT", "DOT", debt_amt, collateral_amt)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): spot_price},
        liquidation_threshold=0.1,  # very low to ensure liquidation
        liquidation_penalty=penalty
    )

    collat_liquidated = mm.get_liquidate_collateral_amt(cdp, debt_amt)
    if collat_liquidated != pytest.approx(collat_liquidated_exp, rel=1e-15):
        raise


def test_liquidate():
    debt_amt = 1000
    collat_amt = 100
    borrow_agent = Agent()
    cdp = CDP("USDT", "DOT", debt_amt, collat_amt)
    collat_price = 11
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): collat_price},
        cdps=[(borrow_agent, cdp)],
        liquidation_threshold=0.7,
        liquidation_penalty=0.01,
    )
    liquidate_amt = debt_amt / 2
    collat_sold_expected = liquidate_amt / collat_price * 1.01
    agent = Agent(holdings={"USDT": liquidate_amt})
    mm.liquidate(cdp, agent, liquidate_amt)
    if debt_amt - liquidate_amt != cdp.debt_amt:
        raise
    if agent.holdings["USDT"] != 0:
        raise
    if agent.holdings["DOT"] != collat_sold_expected:
        raise
    if cdp.collateral_amt != collat_amt - collat_sold_expected:
        raise
    if not mm.validate():
        raise


def test_liquidate_not_liquidatable():
    debt_amt = 1000
    collat_amt = 500
    borrow_agent = Agent()
    cdp = CDP("USDT", "DOT", debt_amt, collat_amt)
    collat_price = 11
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): collat_price},
        cdps=[(borrow_agent, cdp)],
        liquidation_threshold=0.7,
        liquidation_penalty=0.01,
    )
    liquidate_amt = debt_amt / 2
    agent = Agent(holdings={"USDT": liquidate_amt})
    mm.liquidate(cdp, agent, liquidate_amt)
    if debt_amt != cdp.debt_amt:
        raise
    if agent.holdings["USDT"] != liquidate_amt:
        raise
    if agent.is_holding("DOT"):
        raise
    if cdp.collateral_amt != collat_amt:
        raise
    if not mm.validate():
        raise


def test_liquidate_fails():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 100
    borrow_agent = Agent()
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, init_collat_amt)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 11},
        cdps=[(borrow_agent, cdp.copy())]
    )
    agent = Agent(holdings={"USDT": 10000, "DOT": 10000})

    with pytest.raises(Exception):  # trying to liquidate more than debt amount, should fail
        mm.liquidate(cdp, agent, init_debt_amt * 2)


def test_liquidate_undercollateralized():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 10
    borrow_agent = Agent()
    penalty = 0.01

    # not enough collateral, should liquidate all collateral and have debt left over
    cdp = CDP(debt_asset, collateral_asset, init_debt_amt, init_collat_amt)
    price = 11
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): price},
        cdps=[(borrow_agent, cdp.copy())],
        liquidation_penalty=penalty
    )
    agent = Agent(holdings={"USDT": 10000, "DOT": 0})

    mm.liquidate(cdp, agent, init_debt_amt)
    if cdp.collateral_amt != 0:
        raise
    debt_liq = 110 / (1 + penalty)
    if cdp.debt_amt != init_debt_amt - debt_liq:
        raise
    if agent.holdings["USDT"] != 10000 - debt_liq:
        raise
    if agent.holdings["DOT"] != init_collat_amt:
        raise


# @settings(max_examples=1)
@given(
    st.floats(min_value=1.0, max_value=1.0),
    st.floats(min_value=0.1, max_value=1.0),
    st.floats(min_value=0.01, max_value=0.05),
    st.floats(min_value=0.6, max_value=0.75),
    st.floats(min_value=0.75, max_value=0.9),
    st.floats(min_value=0.2, max_value=0.8),

)
def test_liquidate_fuzz_ltv(ltv_ratio: float, liq_pct: float, penalty: float, liq_threshold: float,
                            full_liq_threshold: float, partial_liq_pct: float):

    # ltv_ratio = 1 / (1.01) + 0.000001
    # liq_pct = 1.0

    # penalty = mpf(0.01)
    # liq_threshold = mpf(0.7)
    # full_liq_threshold = mpf(0.8)
    # partial_liq_pct = mpf(0.5)

    collat_amt = mpf(100)
    collat_price = mpf(11)

    debt_amt = collat_amt * collat_price * ltv_ratio
    borrow_agent = Agent()
    cdp = CDP("USDT", "DOT", debt_amt, collat_amt)
    mm = money_market(
        liquidity={"USDT": mpf(1000000), "DOT": mpf(1000000)},
        oracles={("DOT", "USDT"): collat_price},
        cdps=[(borrow_agent, cdp)],
        liquidation_threshold=liq_threshold,
        full_liquidation_threshold=full_liq_threshold,
        partial_liquidation_pct=partial_liq_pct,
        liquidation_penalty=penalty,
    )
    liquidate_amt = debt_amt * liq_pct
    collat_sold_expected = liquidate_amt / collat_price * (1 + penalty)
    agent = Agent(holdings={"USDT": liquidate_amt})
    mm.liquidate(cdp, agent, liquidate_amt)

    if debt_amt - liquidate_amt == cdp.debt_amt:  # case 1: we liquidated
        if ltv_ratio < liq_threshold:
            raise
        if ltv_ratio < full_liq_threshold and liq_pct > partial_liq_pct:
            raise
        if cdp.collateral_amt != pytest.approx(collat_amt - collat_sold_expected, rel=1e-15):
            raise
    elif cdp.collateral_amt == 0:  # case 2: we liquidated but left some toxic debt
        if ltv_ratio <= 1 / (1+penalty):
            raise
    else:
        if cdp.debt_amt != debt_amt:  # shouldn't liquidate some partial amount
            raise
        if ltv_ratio >= full_liq_threshold:
            raise
        if ltv_ratio >= liq_threshold and liq_pct <= partial_liq_pct:
            raise


def test_borrow():
    borrow_asset = "USDT"
    collateral_asset = "DOT"
    borrow_amt = 500
    collat_amt = 100
    agent = Agent(holdings={collateral_asset: collat_amt})
    mm = money_market(
        liquidity={borrow_asset: 1000000, collateral_asset: 1000000},
        oracles={("DOT", "USDT"): 11},
        liquidation_threshold=0.7,
    )
    mm.borrow(agent, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    if agent.holdings[borrow_asset] != borrow_amt:
        raise
    if agent.is_holding(collateral_asset):
        raise
    if len(mm.cdps) != 1:
        raise
    cdp = mm.cdps[0][1]
    if cdp.debt_amt != borrow_amt:
        raise
    if cdp.collateral_amt != collat_amt:
        raise
    if cdp.debt_asset != borrow_asset:
        raise
    if cdp.collateral_asset != collateral_asset:
        raise
    if not mm.validate():
        raise

    agent2 = Agent(holdings={collateral_asset: collat_amt})
    mm.borrow(agent2, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    if len(mm.cdps) != 2:
        raise
    if not mm.validate():
        raise


def test_borrow_fails():
    borrow_asset = "USDT"
    collateral_asset = "DOT"
    collat_amt = 100
    agent = Agent(holdings={collateral_asset: collat_amt})
    mm = money_market(
        liquidity={borrow_asset: 1000000, collateral_asset: 1000000},
        oracles={("DOT", "USDT"): 10, ("ETH", "UST"): 3000, ("ETH", "DOT"): 300},
        liquidation_threshold=0.7,
        min_ltv=0.6
    )
    borrow_amt = collat_amt * 10 * 0.65
    with pytest.raises(Exception):  # should fail because LTV is too low
        mm.borrow(agent, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    borrow_amt = collat_amt * 10 * 0.50
    with pytest.raises(Exception):  # should fail because debt asset == collateral asset
        mm.borrow(agent, collateral_asset, collateral_asset, borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because collateral asset is not in agent holdings
        mm.borrow(agent, borrow_asset, "ETH", borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because borrow asset is not in liquidity
        mm.borrow(agent, "ETH", collateral_asset, borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because of missing oracle
        mm.borrow(agent, "BTC", collateral_asset, borrow_amt, collat_amt)


def test_repay():
    agent = Agent(holdings={"USDT": 1000})
    cdp = CDP("USDT", "DOT", 1000, 200)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)]
    )
    mm.repay(agent, 0)
    if agent.holdings["USDT"] != 0:
        raise
    if mm.borrowed["USDT"] != 0:
        raise
    if not mm.validate():
        raise


def test_repay_fails():
    agent = Agent(holdings={"USDT": 500})
    cdp = CDP("USDT", "DOT", 1000, 200)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)]
    )
    with pytest.raises(Exception):  # should fail because agent does not have enough funds
        mm.repay(agent, 0)


def test_add_collateral():
    agent = Agent(holdings={"DOT": 500})
    cdp = CDP("USDT", "DOT", 1000, 200)
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): 10},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)]
    )
    mm.add_collateral(0, 100)
    if cdp.collateral_amt != 300:
        raise


def omnipool_setup_for_liquidation_testing() -> OmnipoolState:
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
        liquidity[tkn] = mpf(initial_omnipool_tvl * info['weight'] / info['usd price'])
        lrna[tkn] = mpf(initial_omnipool_tvl * info['weight'] / lrna_price_usd)

    omnipool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0025,
        lrna_fee=0.0005
    )

    return omnipool


@given(
    st.floats(min_value=200, max_value=2000)
)
def test_omnipool_liquidate_cdp_oracle_equals_spot_small_cdp(collateral_amt: float):

    omnipool = omnipool_setup_for_liquidation_testing()
    init_pool = omnipool.copy()
    dot_price = omnipool.price(omnipool, "DOT", "USDT")

    collat_ratio = 0.75

    debt_amt = collat_ratio * dot_price * collateral_amt
    init_cdp = CDP('USDT', 'DOT', mpf(debt_amt), mpf(collateral_amt))
    cdp = init_cdp.copy()
    penalty = 0.01

    agent = Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): dot_price},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)],
        min_ltv=0.6
    )

    liquidation_amount = debt_amt
    init_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    treasury_agent = init_agent.copy()
    omnipool_liquidate_cdp(omnipool, mm, 0, treasury_agent, liquidation_amount)

    before_DOT = init_agent.holdings['DOT'] + init_cdp.collateral_amt + init_pool.liquidity['DOT']
    before_USDT = init_agent.holdings['USDT'] - init_cdp.debt_amt + init_pool.liquidity['USDT']
    final_DOT = treasury_agent.holdings['DOT'] + cdp.collateral_amt + omnipool.liquidity['DOT']
    final_USDT = treasury_agent.holdings['USDT'] - cdp.debt_amt + omnipool.liquidity['USDT']
    if before_DOT != pytest.approx(final_DOT, rel=1e-20):  # check that total collateral asset amounts are correct
        raise
    if before_USDT != pytest.approx(final_USDT, rel=1e-20):  # check that total debt asset amounts are correct
        raise

    if not cdp.validate():
        raise
    if not mm.validate():
        raise

    if treasury_agent.holdings['DOT'] > penalty * (init_cdp.collateral_amt - cdp.collateral_amt):
        raise  # treasury should collect at most penalty
    for tkn in treasury_agent.holdings:
        if tkn != cdp.collateral_asset and treasury_agent.holdings[tkn] != 0:
            raise  # treasury_agent should accrue no other token


def test_omnipool_liquidate_cdp_delta_debt_too_large():
    collat_ratio = 5.0
    collateral_amt = 200
    omnipool = omnipool_setup_for_liquidation_testing()
    dot_price = omnipool.price(omnipool, "DOT", "USDT")

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP('USDT', 'DOT', mpf(debt_amt), mpf(collateral_amt))
    cdp = init_cdp.copy()

    agent = Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): dot_price},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)],
        min_ltv=0.6,
        liquidation_penalty=0.01
    )

    init_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    treasury_agent = init_agent.copy()
    with pytest.raises(Exception):  # liquidation should fail because delta_debt > cdp.debt_amt
        omnipool_liquidate_cdp(omnipool, mm, 0, treasury_agent, cdp.debt_amt * 1.01)


def test_omnipool_liquidate_cdp_not_profitable():
    collat_ratio = 5.0
    collateral_amt = 2000
    omnipool = omnipool_setup_for_liquidation_testing()
    for tkn in omnipool.asset_list:  # reduce TVL to increase slippage for the test
        omnipool.liquidity[tkn] /= 2000
    dot_price = omnipool.price(omnipool, "DOT", "USDT")

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP('USDT', 'DOT', mpf(debt_amt), mpf(collateral_amt))
    cdp = init_cdp.copy()

    agent = Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): dot_price},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp)],
        min_ltv=0.6,
        liquidation_penalty=0.01
    )

    init_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    treasury_agent = init_agent.copy()
    with pytest.raises(Exception):  # liquidation should fail because it is not profitable
        omnipool_liquidate_cdp(omnipool, mm, 0, treasury_agent, cdp.debt_amt)


#################################################
# tests for liquidate_against_omnipool          #
#################################################

# test_liquidate_against_omnipool_full_liquidation
# - tests full liquidation
# - tests liquidation of some CDPs but not others
# - test with CDP list with different assets

# test_liquidate_against_omnipool_partial_liquidation
# - test partial liquidation due to higher full liquidation threshold
# - test partial liquidation due to limited Omnipool trade profitability

# test_liquidate_against_omnipool_no_liquidation
# - test no liquidation due to overcollateralization
# - test no liquidation due to undercollateralization
# - test no liquidation due to unprofitability of liquidation against Omnipool

# fuzz test logic that determines how much is liquidated

# For fuzz test of omnipool_liquidate_cdp, we will want to see that
# - any liquidation results in treasury profit
# - if liquidation doesn't happen, buying delta_debt requires too much collateral


@given(st.floats(min_value=0.7, max_value=0.9), st.floats(min_value=0.3, max_value=0.5),
       st.floats(min_value=2.0, max_value=3.0))
def test_liquidate_against_omnipool_full_liquidation(ratio1: float, ratio2: float, ratio3: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    # CDP1 should be fully liquidated
    collateral_amt1 = 200
    debt_amt1 = ratio1 * collateral_amt1 * omnipool.price(omnipool, "DOT", "USDT")
    cdp1 = CDP('USDT', 'DOT', debt_amt1, collateral_amt1)

    # CDP2 should not be liquidated at all
    collateral_amt2 = 1000000
    debt_amt2 = ratio2 * collateral_amt2 * omnipool.price(omnipool, "HDX", "USDT")
    cdp2 = CDP('USDT', 'HDX', debt_amt2, collateral_amt2)

    # CDP3 should be fully liquidated, due to lower liquidation threshold for HDX
    collateral_amt3 = 1000
    debt_amt3 = ratio2 * collateral_amt3 * omnipool.price(omnipool, "USDT", "HDX")
    cdp3 = CDP('HDX', 'USDT', debt_amt3, collateral_amt3)

    # CDP4 should be fully liquidated, with debt left over
    collateral_amt4 = 10000
    debt_amt4 = ratio3 * collateral_amt4 * omnipool.price(omnipool, "HDX", "USDT")
    cdp4 = CDP('USDT', 'HDX', debt_amt4, collateral_amt4)

    liq_agent, agent1, agent2, agent3, agent4 = Agent(), Agent(), Agent(), Agent(), Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000, "HDX": 100000000},
        oracles={
            ("DOT", "USDT"): omnipool.price(omnipool, "DOT", "USDT"),
            ("HDX", "USDT"): omnipool.price(omnipool, "HDX", "USDT")
        },
        liquidation_threshold={"DOT": 0.7, "HDX": 0.7, "USDT": 0.3},
        cdps=[(agent1, cdp1), (agent2, cdp2), (agent3, cdp3), (agent4, cdp4)],
        min_ltv=0.6,
        liquidation_penalty=0.01
    )

    evolve_function = liquidate_against_omnipool("omnipool", "liq_agent")
    state = GlobalState(agents={"agent1": agent1, "agent2": agent2, "agent3": agent3, "agent4": agent4,
                                "liq_agent": liq_agent},
                        pools={"omnipool": omnipool}, money_market=mm, evolve_function=evolve_function)

    state.evolve()

    if cdp1.debt_amt != 0:
        raise ValueError("CDP1 should be fully liquidated")
    if cdp2.debt_amt != debt_amt2:
        raise ValueError("CDP2 should not be liquidated")
    if cdp3.debt_amt != 0:
        raise ValueError("CDP3 should be fully liquidated")
    if cdp4.collateral_amt != 0:
        raise ValueError("CDP4 should be fully liquidated")
    if cdp4.debt_amt == 0:
        raise ValueError("CDP4 should still have debt left over")


def test_liquidate_against_omnipool_partial_liquidation():
    omnipool = omnipool_setup_for_liquidation_testing()

    dot_full_liq_threshold = 0.8
    dot_liq_threshold = 0.6
    collateral_amt1 = 200
    collat_ratio1 = 0.7
    debt_amt1 = collat_ratio1 * collateral_amt1 * omnipool.price(omnipool, "DOT", "USDT")
    cdp1 = CDP('USDT', 'DOT', debt_amt1, collateral_amt1)

    hdx_full_liq_threshold = 0.7
    hdx_liq_threshold = 0.7
    collateral_amt2 = 10000000
    collat_ratio2 = 0.7
    debt_amt2 = collat_ratio2 * collateral_amt2 * omnipool.price(omnipool, "HDX", "USDT")
    cdp2 = CDP('USDT', 'HDX', debt_amt2, collateral_amt2)

    liq_agent = Agent(holdings={"HDX": mpf(0), "USDT": mpf(0), "DOT": mpf(0)})
    agent1, agent2 = Agent(), Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000, "HDX": 100000000},
        oracles={
            ("DOT", "USDT"): omnipool.price(omnipool, "DOT", "USDT"),
            ("HDX", "USDT"): omnipool.price(omnipool, "HDX", "USDT")
        },
        full_liquidation_threshold={"DOT": dot_full_liq_threshold, "HDX": hdx_full_liq_threshold, "USDT": 0.7},
        liquidation_threshold={"DOT": dot_liq_threshold, "HDX": hdx_liq_threshold, "USDT": 0.7},
        cdps=[(agent1, cdp1), (agent2, cdp2)],
        min_ltv=0.6,
        liquidation_penalty=0.01,
        partial_liquidation_pct=0.5
    )

    evolve_function = liquidate_against_omnipool("omnipool", "liq_agent", 100)
    state = GlobalState(agents={"agent1": agent1, "agent2": agent2, "liq_agent": liq_agent},
                        pools={"omnipool": omnipool}, money_market=mm, evolve_function=evolve_function)

    state.evolve()

    if cdp1.debt_amt != debt_amt1 * (1 - mm.partial_liquidation_pct):
        raise ValueError("CDP1 should be partially liquidated by partial_liquidation_pct")
    if cdp2.debt_amt == 0:
        raise ValueError("CDP2 should not be fully liquidated")
    if cdp2.debt_amt == debt_amt2:
        raise ValueError("CDP2 should be partially liquidated")
    hdx_liquidated = collateral_amt2 - cdp2.collateral_amt
    if liq_agent.holdings["HDX"] / hdx_liquidated > 1e-25:
        raise ValueError("If liquidation agent is profitable, they should have liquidated more")


@given(st.floats(min_value=0.0, max_value=0.7, exclude_min=True, exclude_max=True))
def test_liquidate_against_omnipool_no_liquidation(ratio1: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    collateral_amt1 = 200
    debt_amt1 = ratio1 * collateral_amt1 * omnipool.price(omnipool, "DOT", "USDT")
    cdp1 = CDP('USDT', 'DOT', debt_amt1, collateral_amt1)

    collateral_amt2 = 10
    price_mult = 10  # this offsets the oracle price from Omnipool price, making liquidation unprofitable
    debt_amt2 = 0.7 * collateral_amt2 * price_mult * omnipool.price(omnipool, "WETH", "USDT")
    cdp2 = CDP('USDT', 'WETH', debt_amt2, collateral_amt2)

    liq_agent, agent1, agent2, agent3 = Agent(), Agent(), Agent(), Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000, "HDX": 1000000, "WETH": 1000000},
        oracles={("DOT", "USDT"): omnipool.price(omnipool, "DOT", "USDT"),
                 ("HDX", "USDT"): omnipool.price(omnipool, "HDX", "USDT"),
                 ("WETH", "USDT"): price_mult * omnipool.price(omnipool, "WETH", "USDT")},
        liquidation_threshold=0.7,
        cdps=[(agent1, cdp1), (agent3, cdp2)],
        min_ltv=0.6,
        liquidation_penalty=0.01
    )

    evolve_function = liquidate_against_omnipool("omnipool", "liq_agent")
    state = GlobalState(agents={"agent1": agent1, "agent2": agent2, "agent3": agent3, "liq_agent": liq_agent},
                        pools={"omnipool": omnipool}, money_market=mm, evolve_function=evolve_function)

    state.evolve()

    if debt_amt1 != cdp1.debt_amt:
        raise ValueError("No liquidation should occur")
    if debt_amt2 != cdp2.debt_amt:
        raise ValueError("No liquidation should occur")


@settings(print_blob=True)
@given(
    st.floats(min_value=100, max_value=1000000),
    # st.floats(min_value=0.1, max_value=10.0),
    st.floats(min_value=0.8, max_value=1.0),
    st.floats(min_value=0.5, max_value=1.5)
)
def test_liquidate_against_omnipool_fuzz(collateral_amt1: float, ratio1: float, price_mult: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    # collateral_amt1 = 516949.1872315724
    # ratio1 = 0.8699232790986524
    # price_mult = 0.5238060155349247
    liq_threshold = mpf(0.7)
    full_liq_threshold = mpf(0.8)

    debt_amt1 = ratio1 * mpf(collateral_amt1) * price_mult * omnipool.price(omnipool, "DOT", "USDT")
    cdp1 = CDP('USDT', 'DOT', debt_amt1, mpf(collateral_amt1))

    liq_agent, agent1 = Agent(), Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): price_mult * omnipool.price(omnipool, "DOT", "USDT")},
        liquidation_threshold=liq_threshold,
        full_liquidation_threshold=full_liq_threshold,
        partial_liquidation_pct=0.5,
        cdps=[(agent1, cdp1)],
        liquidation_penalty=0.01
    )

    evolve_function = liquidate_against_omnipool("omnipool", "liq_agent", 100)
    state = GlobalState(agents={"agent1": agent1, "liq_agent": liq_agent},
                        pools={"omnipool": omnipool}, money_market=mm, evolve_function=evolve_function)

    state.evolve()

    if 0 == pytest.approx(cdp1.debt_amt, rel=1e-12):  # fully liquidated
        assert ratio1 >= full_liq_threshold
    elif cdp1.collateral_amt == 0:  # fully liquidated, bad debt remaining
        assert ratio1 > 1
    elif cdp1.debt_amt == debt_amt1:  # not liquidated
        if ratio1 < liq_threshold:  # 1. overcollateralized
            pass
        elif ratio1 > 1 - mm.liquidation_penalty['DOT']:  # 2. undercollateralized
            pass
        elif price_mult >= 1:  # 3. not profitable to liquidate
            pass
        else:
            raise ValueError("CDP should be liquidated")
    elif 0 < cdp1.debt_amt < debt_amt1:  # partially liquidated
        assert ratio1 >= liq_threshold
        if ratio1 - full_liq_threshold <= 1e-20 and cdp1.debt_amt / debt_amt1 == 1 - mm.partial_liquidation_pct:
            pass  # partially liquidated due to partial_liquidation_pct
        elif liq_agent.holdings["DOT"] / (collateral_amt1 - cdp1.collateral_amt) > 1e-25:
            raise ValueError("If liquidation agent is profitable, they should have liquidated more")
        elif liq_agent.holdings["DOT"] < 0:
            raise ValueError("Liquidation agent should not have negative holdings")
    else:
        raise ValueError("CDP debt amount should not go above initial debt_amt or below 0")


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
    initial_omnipool_tvl = mpf(20000000)
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
    cdp = CDP('USDT', 'DOT', debt_amt, collateral_amt)
    penalty = 0.01

    cdp_copy = cdp.copy()
    agent = Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): prices['DOT']},
        liquidation_threshold=0.7,
        cdps=[(agent, cdp_copy)],
        min_ltv=0.6,
        liquidation_penalty=penalty
    )

    liquidation_amount = find_partial_liquidation_amount(omnipool, mm, 0, 100)
    if liquidation_amount >= cdp.debt_amt:
        raise  # liquidation should be partial
    if liquidation_amount == 0:
        raise  # liquidation should happen

    omnipool_copy = omnipool.copy()
    treasury_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    omnipool_liquidate_cdp(omnipool_copy, mm, 0, treasury_agent, liquidation_amount)

    # collat_penalty = treasury_agent.holdings[cdp.collateral_asset]
    collat_sold = cdp.collateral_amt - cdp_copy.collateral_amt
    # if collat_penalty != pytest.approx(penalty * (collat_sold - collat_penalty), rel=1e-15):
    #     raise

    if treasury_agent.holdings[cdp.collateral_asset] < 0:
        raise
    if treasury_agent.holdings[cdp.collateral_asset] >= 1e10:
        raise  # partial liquidation means no profit left over for treasury

    # debt_liquidated = cdp.debt_amt - cdp_copy.debt_amt
    # exec_price = debt_liquidated / collat_sold
    #
    # if exec_price != pytest.approx(collat_ratio, rel=1e-12):
    #     raise  # execution price should be equal to collateral ratio


def test_liquidate_against_omnipool():
    prices = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}

    assets = {
        'DOT': {'usd price': prices['DOT'], 'weight': mpf(0.40)},
        'HDX': {'usd price': prices['HDX'], 'weight': mpf(0.10)},
        'USDT': {'usd price': prices['USDT'], 'weight': mpf(0.30)},
        'WETH': {'usd price': prices['WETH'], 'weight': mpf(0.10)},
        'iBTC': {'usd price': prices['iBTC'], 'weight': mpf(0.10)}
    }

    lrna_price_usd = 35
    initial_omnipool_tvl = 20000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    oracles = {('DOT', 'USDT'): prices['DOT']}

    init_pool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
        },
        preferred_stablecoin='USDT',
    )

    # should be fully liquidated
    cdp1 = CDP('USDT', 'DOT', 1000, 200)
    # should be partially liquidated
    cdp2 = CDP('USDT', 'DOT', 1000000, 1000000 / 6.5)
    # should be fully liquidated
    cdp3 = CDP('DOT', 'USDT', 100, 900)
    agents = {"treasury": Agent(holdings={"USDT": 0, "DOT": 0})}

    agent = Agent()
    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles=oracles,
        liquidation_threshold=0.7,
        cdps=[(agent, cdp1), (agent, cdp2), (agent, cdp3)],
        min_ltv=0.6,
        liquidation_penalty=0.02
    )

    init_state = GlobalState(
        agents=agents,
        pools={'omnipool': init_pool},
        money_market=mm
    )

    transform_fn = liquidate_against_omnipool("omnipool", "treasury")
    transform_fn(init_state)

    if cdp1.debt_amt != 0:
        raise  # should be fully liquidated
    if cdp2.debt_amt == 0 or cdp2.debt_amt == 1000000:
        raise  # should be partially liquidated
    if cdp3.debt_amt != 0:
        raise  # should be fully liquidated
    if len(init_state.money_market.cdps) != 3:
        raise


def test_liquidate_against_omnipool_and_settle_otc():
    prices = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}

    assets = {
        'DOT': {'usd price': prices['DOT'], 'weight': mpf(0.40)},
        'HDX': {'usd price': prices['HDX'], 'weight': mpf(0.10)},
        'USDT': {'usd price': prices['USDT'], 'weight': mpf(0.30)},
        'WETH': {'usd price': prices['WETH'], 'weight': mpf(0.10)},
        'iBTC': {'usd price': prices['iBTC'], 'weight': mpf(0.10)}
    }

    lrna_price_usd = 35
    initial_omnipool_tvl = 20000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    init_pool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
        },
        preferred_stablecoin='USDT',
    )

    # should be fully liquidated
    cdp1 = CDP('USDT', 'DOT', 1000, 200)
    # can't be liquidated
    cdp2 = CDP('USDT', 'DOT', 1000, 500)

    # should be executed and removed
    otc1 = OTC('DOT', 'USDT', 120, 12, partially_fillable=True)
    # should not be executed
    otc2 = OTC('DOT', 'USDT', 120, 5, partially_fillable=True)

    agent = Agent()
    cdps = [(agent, cdp1), (agent, cdp2)]
    otcs = [otc1, otc2]
    agents = {"treasury": Agent(holdings={"USDT": 0, "DOT": 0})}

    mm = money_market(
        liquidity={"USDT": 1000000, "DOT": 1000000},
        oracles={("DOT", "USDT"): prices['DOT']},
        liquidation_threshold=0.7,
        cdps=cdps,
        min_ltv=0.6,
        liquidation_penalty=0.02
    )

    init_state = GlobalState(
        agents=agents, pools={'omnipool': init_pool}, money_market=mm, otcs=otcs)

    transform_fn = liquidate_against_omnipool_and_settle_otc("omnipool", "treasury")
    transform_fn(init_state)

    if cdp1.debt_amt != 0:
        raise  # should be fully liquidated
    # if cdp2.debt_amt != 1000:
    #     raise  # shouldn't be liquidated at all
    if otc1.sell_amount != 0:
        raise  # should be completed
    if otc2.sell_amount != 120:
        raise  # shouldn't be traded
    if len(init_state.money_market.cdps) != 2:
        raise
    if len(init_state.otcs) != 1:
        raise
