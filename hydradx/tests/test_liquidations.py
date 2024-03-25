import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from mpmath import mp, mpf

from hydradx.model.amm.liquidations import CDP
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.global_state import find_partial_liquidation_amount, omnipool_liquidate_cdp

mp.dps = 50


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

    cdp.liquidate_cdp(agent, init_debt_amt * liquidate_pct, init_collat_amt * liquidate_pct)
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
        cdp.liquidate_cdp(agent, init_debt_amt * 1.2, init_collat_amt * 0.7)
    with pytest.raises(Exception):
        cdp.liquidate_cdp(agent, init_debt_amt * 0.7, init_collat_amt * 1.2)
    cdp.liquidate_cdp(agent, init_debt_amt * 0.7, init_collat_amt * 0.7)


@given(
    st.floats(min_value=4, max_value=10),
    st.floats(min_value=200, max_value=200000)
)
def test_omnipool_liquidate_cdp(collat_ratio: float, collateral_amt: float):
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
    omnipool = init_pool.copy()

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP('USDT', 'DOT', mpf(debt_amt), mpf(collateral_amt), True)
    cdp = init_cdp.copy()
    penalty = 0.01
    liquidation_amount = debt_amt
    init_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    treasury_agent = init_agent.copy()
    omnipool_liquidate_cdp(omnipool, cdp, treasury_agent, liquidation_amount, penalty)

    before_DOT = init_agent.holdings['DOT'] + init_cdp.collateral_amt + init_pool.liquidity['DOT']
    before_USDT = init_agent.holdings['USDT'] - init_cdp.debt_amt + init_pool.liquidity['USDT']
    final_DOT = treasury_agent.holdings['DOT'] + cdp.collateral_amt + omnipool.liquidity['DOT']
    final_USDT = treasury_agent.holdings['USDT'] - cdp.debt_amt + omnipool.liquidity['USDT']
    if before_DOT != pytest.approx(final_DOT, rel=1e-20):  # check that total collateral asset amounts are correct
        raise
    if before_USDT != pytest.approx(final_USDT, rel=1e-20):  # check that total debt asset amounts are correct
        raise

    if cdp.debt_amt < 0 or cdp.collateral_amt < 0:  # validity check of cdp state
        raise

    if treasury_agent.holdings['DOT'] != pytest.approx(
            penalty * (omnipool.liquidity['DOT'] - init_pool.liquidity['DOT']), rel=1e-15):
        raise  # check that penalty amount to treasury is correct
    for tkn in treasury_agent.holdings:
        if tkn != cdp.collateral_asset and treasury_agent.holdings[tkn] != 0:
            raise  # treasury_agent should accrue no other token


def test_omnipool_liquidate_cdp_delta_debt_too_large():
    collat_ratio = 5.0
    collateral_amt = 200
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
    omnipool = init_pool.copy()

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP('USDT', 'DOT', mpf(debt_amt), mpf(collateral_amt), True)
    cdp = init_cdp.copy()
    penalty = 0.01
    init_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    treasury_agent = init_agent.copy()
    omnipool_liquidate_cdp(omnipool, cdp, treasury_agent, cdp.debt_amt * 1.01, penalty)

    if cdp.debt_amt != init_cdp.debt_amt or cdp.collateral_amt != init_cdp.collateral_amt:
        raise  # liquidation should fail because delta_debt > cdp.debt_amt


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
    if liquidation_amount != debt_amt:
        raise  # liquidation should be full


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
    if liquidation_amount >= cdp.debt_amt:
        raise  # liquidation should be partial
    if liquidation_amount == 0:
        raise  # liquidation should happen

    omnipool_copy = omnipool.copy()
    cdp_copy = cdp.copy()
    treasury_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    omnipool_liquidate_cdp(omnipool_copy, cdp_copy, treasury_agent, liquidation_amount, penalty)

    collat_penalty = treasury_agent.holdings[cdp.collateral_asset]
    collat_sold = cdp.collateral_amt - cdp_copy.collateral_amt
    if collat_penalty != pytest.approx(penalty * (collat_sold - collat_penalty), rel=1e-15):
        raise

    debt_liquidated = cdp.debt_amt - cdp_copy.debt_amt
    exec_price = debt_liquidated / collat_sold

    if exec_price != pytest.approx(collat_ratio, rel=1e-12):
        raise  # execution price should be equal to collateral ratio


@given(
    st.floats(min_value=4, max_value=10),
    st.floats(min_value=200, max_value=200000),
    st.floats(min_value=0.0, max_value=1.0)
)
def test_find_partial_liquidation_amount(collat_ratio: float, collateral_amt: float, min_amt: float):
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

    omnipool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
        },
        preferred_stablecoin='USDT',
    )

    debt_amt = collat_ratio * collateral_amt
    cdp = CDP('USDT', 'DOT', debt_amt, collateral_amt, True)
    penalty = 0.01

    liquidation_amount = find_partial_liquidation_amount(omnipool, cdp, penalty, 100, min_amt)
    if liquidation_amount > cdp.debt_amt:
        raise
    if liquidation_amount < min_amt and liquidation_amount != 0:
        raise
    if omnipool.buy_spot(cdp.debt_asset, cdp.collateral_asset) > (cdp.collateral_amt * (1 + penalty)) / cdp.debt_amt:
        if liquidation_amount != 0:
            raise

    omnipool_copy = omnipool.copy()
    cdp_copy = cdp.copy()
    treasury_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    omnipool_liquidate_cdp(omnipool_copy, cdp_copy, treasury_agent, liquidation_amount, penalty)

    collat_penalty = treasury_agent.holdings[cdp.collateral_asset]
    collat_sold = cdp.collateral_amt - cdp_copy.collateral_amt
    if collat_penalty != pytest.approx(penalty * (collat_sold - collat_penalty), rel=1e-15):
        raise

    if liquidation_amount > 0:
        debt_liquidated = cdp.debt_amt - cdp_copy.debt_amt
        exec_price = debt_liquidated / collat_sold

        if cdp_copy.debt_amt > 0 and exec_price != pytest.approx(collat_ratio, rel=1e-12):
            raise  # if liquidation is partial, execution price should be equal to collateral ratio
