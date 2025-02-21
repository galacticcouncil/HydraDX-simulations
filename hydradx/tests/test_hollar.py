import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.hollar import LiquidityFacility
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

mp.dps = 50

def test_liquidity_facility_constructor():
    # liquidity facility should work with this params
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    buyback_speed = 1/10_000
    usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    usdc_pool = StableSwapPoolState(tokens={'USDC': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    max_buy_price = 0.999
    buy_fee = 0.0001
    native_stable = 'HOLLAR'
    LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # liquidity facility should allow differing params for different assets
    usdt_buyback_speed = 1/10_000
    usdc_buyback_speed = 1/11_000
    buyback_speed = [usdt_buyback_speed, usdc_buyback_speed]
    usdt_sel_price = 1.001
    usdc_sell_price = 1.002
    sell_price = [usdt_sel_price, usdc_sell_price]
    usdt_max_buy_price = 0.999
    usdc_max_buy_price = 0.998
    max_buy_price = [usdt_max_buy_price, usdc_max_buy_price]
    usdt_buy_fee = 0.0001
    usdc_buy_fee = 0.0002
    buy_fee = [usdt_buy_fee, usdc_buy_fee]
    LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # liquidity facility should fail if buyback_speed is not in [0, 1]
    bad_buyback_speed = 1.1
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, bad_buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)
    bad_buyback_speed = -0.1
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, bad_buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # liquidity facility should fail if sell_price is < 1
    bad_sell_price = 0.999
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, pools, bad_sell_price, max_buy_price, buy_fee, native_stable)

    # liquidity facility should fail if max_buy_price is not in (0, 1]
    bad_max_buy_price = 1.1
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, pools, sell_price, bad_max_buy_price, buy_fee, native_stable)
    bad_max_buy_price = 0
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, pools, sell_price, bad_max_buy_price, buy_fee, native_stable)

    # liquidity facility should fail if buy_fee is not in [0, 1]
    bad_buy_fee = 1.1
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, bad_buy_fee, native_stable)
    bad_buy_fee = -0.1
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, bad_buy_fee, native_stable)

    # liquidity facility should fail if native_stablecoin is in liquidity
    bad_native_stable = 'USDT'
    with pytest.raises(AssertionError):
        LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, bad_native_stable)

    # liquidity facility should fail if a pool does not have correct stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'aUSDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, usdc_pool]
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, bad_pools, sell_price, max_buy_price, buy_fee, native_stable)

    # liquidity facility should fail if a pool does not have native stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'USDC': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, usdc_pool]
    with pytest.raises(ValueError):
        LiquidityFacility(liquidity, buyback_speed, bad_pools, sell_price, max_buy_price, buy_fee, native_stable)


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    buy_fee = st.floats(min_value=0, max_value=0.01),
    sell_ratio = st.floats(min_value=1e-10, max_value=1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
)
@reproduce_failure('6.39.6', b'AXicY2AgGgAAACsAAQ==')
def test_facility_sell_hollar(ratios, buyback_speed, max_buy_price, buy_fee, sell_ratio, buy_tkn_i):
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-10)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-10)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    tkn_buy = list(liquidity.keys())[buy_tkn_i]
    tkn_sell = 'HOLLAR'
    facility = LiquidityFacility(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)
    max_sell_amt, buy_price = facility.get_buy_params(tkn_buy)
    assert max_sell_amt > 0  # this test case focuses on parameters in which facility can buy HOLLAR
    liq_diff = usdt_pool.liquidity['HOLLAR'] - usdt_pool.liquidity[tkn_buy]
    if liq_diff > 0:
        if max_sell_amt != pytest.approx(buyback_speed * liq_diff / 2, rel=1e-15):
            raise ValueError(f"Expected {buyback_speed * liq_diff / 2}, got {max_sell_amt}")
    else:
        if max_sell_amt != 0:
            raise ValueError("max_sell_amt should be 0")
    sell_amt = max_sell_amt * sell_ratio
    agent = Agent(holdings = {tkn_sell: sell_amt})
    facility.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
    if agent.holdings[tkn_sell] != 0:
        raise ValueError("Agent should have 0 holdings after selling")
    buy_amt = agent.holdings[tkn_buy]
    exec_price = buy_amt / sell_amt
    if exec_price > max_buy_price:
        raise ValueError("Execution price should be less than max buy price")
    # execution price should be slightly higher compared to pool price of closing arb
    # as long as we are away from the price boundary
    pool_sell_amt = usdt_pool.calculate_sell_from_buy(tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=sell_amt)
    if (pool_sell_amt - buy_amt)/pool_sell_amt > 1e-15:
        raise AssertionError("Liquidity facility not giving arbitragable prices")

