import copy

import pytest
from hypothesis import given, strategies as st, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.hollar import StabilityModule
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

mp.dps = 50

def test_stability_module_constructor():
    # stability module should work with this params
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    buyback_speed = 1/10_000
    usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    usdc_pool = StableSwapPoolState(tokens={'USDC': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    max_buy_price = 0.999
    buy_fee = 0.0001
    native_stable = 'HOLLAR'
    StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # stability module should allow differing params for different assets
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
    StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # stability module should fail if buyback_speed is not in [0, 1]
    bad_buyback_speed = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, bad_buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)
    bad_buyback_speed = -0.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, bad_buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    # stability module should fail if sell_price is < 1
    bad_sell_price = 0.999
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, bad_sell_price, max_buy_price, buy_fee, native_stable)

    # stability module should fail if max_buy_price is not in (0, 1]
    bad_max_buy_price = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, bad_max_buy_price, buy_fee, native_stable)
    bad_max_buy_price = 0
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, bad_max_buy_price, buy_fee, native_stable)

    # stability module should fail if buy_fee is not in [0, 1]
    bad_buy_fee = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, bad_buy_fee, native_stable)
    bad_buy_fee = -0.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, bad_buy_fee, native_stable)

    # stability module should fail if native_stablecoin is in liquidity
    bad_native_stable = 'USDT'
    with pytest.raises(AssertionError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, bad_native_stable)

    # stability module should fail if a pool does not have correct stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'aUSDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, usdc_pool]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, bad_pools, sell_price, max_buy_price, buy_fee, native_stable)

    # stability module should fail if a pool does not have native stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'USDC': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, usdc_pool]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, bad_pools, sell_price, max_buy_price, buy_fee, native_stable)


def test_overlapping_tokens():
    three_pool_tokens = {'A': 1_000_000, 'HOLLAR': 1_000_000, 'B': 1_000_000}
    three_pool = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pools = [three_pool, three_pool]

    liquidity = {'A': 1_000_000, 'B': 1_000_000}
    sell_price = 1.001
    max_buy_price = 0.999
    buy_fee = 0.0001
    native_stable = 'HOLLAR'
    usdt_buyback_speed = 1/10_000
    usdc_buyback_speed = 1/11_000
    buyback_speed = [usdt_buyback_speed, usdc_buyback_speed]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)

    pool1 = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pool2 = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pools = [pool1, pool2]

    StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee, native_stable)


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    buy_fee = st.floats(min_value=0, max_value=0.01),
    sell_ratio = st.floats(min_value=0.01, max_value=1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
)
def test_sell_hollar_to_stability_module(ratios, buyback_speed, max_buy_price, buy_fee, sell_ratio, buy_tkn_i):
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    pool_buy = pools[buy_tkn_i]
    sell_price = 1.001
    tkn_buy = list(liquidity.keys())[buy_tkn_i]
    tkn_sell = 'HOLLAR'
    init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    max_sell_amt, buy_price = init_hsm.get_buy_params(tkn_buy)
    assert max_sell_amt > 0  # this test case focuses on parameters in which HSM can buy HOLLAR
    liq_diff = pool_buy.liquidity['HOLLAR'] - pool_buy.liquidity[tkn_buy]
    if max_sell_amt != pytest.approx(buyback_speed * liq_diff / 2, rel=1e-15):
        raise ValueError(f"Expected {buyback_speed * liq_diff / 2}, got {max_sell_amt}")

    # test sell Hollar to HSM
    stability_module = copy.deepcopy(init_hsm)
    sell_amt = max_sell_amt * sell_ratio
    agent = Agent(holdings = {tkn_sell: sell_amt})
    stability_module.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
    if agent.validate_holdings(tkn_sell):
        raise ValueError("Agent should have 0 holdings after selling")
    buy_amt = agent.get_holdings(tkn_buy)
    exec_price = buy_amt / sell_amt
    if exec_price > max_buy_price:
        raise ValueError("Execution price should be less than max buy price")
    # execution price should be higher than pool price of closing arb
    pool_sell_amt = pool_buy.calculate_sell_from_buy(tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=sell_amt)
    if (pool_sell_amt - buy_amt)/pool_buy.liquidity[tkn_buy] > 1e-12:
        raise AssertionError("Stability module not giving arbitragable prices")
    if abs(buy_amt - (init_hsm.liquidity[tkn_buy] - stability_module.liquidity[tkn_buy])) / stability_module.liquidity[tkn_buy] >= 1e-15:
        raise ValueError("Stability module did not deduct correct amount of tokens")

    # test buy USDT from HSM
    buy_amt *= 0.99999  # make sure to stay under max buy amount
    sell_amt *= 0.99999
    buy_hsm = copy.deepcopy(init_hsm)
    buy_agent = Agent(enforce_holdings=False)
    buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_amt)
    if buy_agent.get_holdings(tkn_buy) != buy_amt:
        raise ValueError("Agent has incorrect amount of buy token")
    if abs(-buy_agent.get_holdings(tkn_sell) - sell_amt)/sell_amt >= 1e-15:
        raise ValueError("Agent sold incorrect amount")
    if abs(buy_amt - (init_hsm.liquidity[tkn_buy] - buy_hsm.liquidity[tkn_buy])) / buy_hsm.liquidity[tkn_buy] >= 1e-15:
        raise ValueError("Stability module did not deduct correct amount of tokens")


@given(
    ratios = st.lists(st.floats(min_value=0.999, max_value=1.001), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1, exclude_max=True),
    buy_fee = st.floats(min_value=0, max_value=0.01),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
)
def test_sell_hollar_fails_when_balanced(ratios, buyback_speed, max_buy_price, buy_fee, buy_tkn_i):
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    tkn_buy = list(liquidity.keys())[buy_tkn_i]
    tkn_sell = 'HOLLAR'
    init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    max_sell_amt, buy_price = init_hsm.get_buy_params(tkn_buy)
    if max_sell_amt != 0:
        raise ValueError("Should not be able to sell HOLLAR when pools are balanced")

    # test sell Hollar to HSM
    hsm = copy.deepcopy(init_hsm)
    sell_amt = 10
    agent = Agent(holdings = {tkn_sell: sell_amt})
    hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
    if not hsm.fail:
        raise ValueError("Should not be able to sell HOLLAR when pools are balanced")

    # test buy USDT from HSM
    buy_hsm = copy.deepcopy(init_hsm)
    buy_agent = Agent(enforce_holdings=False)
    buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=10)
    if not buy_hsm.fail:
        raise ValueError("Should not be able to buy USDT when pools are balanced")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/10_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    buy_fee = st.floats(min_value=0.00001, max_value=0.01),
    buy_tkn_i = st.integers(min_value=0, max_value=1)
)
def test_sell_hollar_to_stability_module_fee(ratios, buyback_speed, max_buy_price, buy_fee, buy_tkn_i):
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    pool_buy = pools[buy_tkn_i]
    sell_price = 1.001
    tkn_buy = list(liquidity.keys())[buy_tkn_i]
    tkn_sell = 'HOLLAR'
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)
    max_sell_amt, buy_price = hsm.get_buy_params(tkn_buy)
    assert max_sell_amt > 0  # this test case focuses on parameters in which stability module can buy HOLLAR
    agent = Agent(holdings = {tkn_sell: max_sell_amt})
    hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=max_sell_amt)
    buy_amt = agent.get_holdings(tkn_buy)
    # execution price should be higher than pool price of closing arb
    pool_sell_amt = pool_buy.calculate_sell_from_buy(tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=max_sell_amt)
    if (pool_sell_amt - buy_amt)/pool_sell_amt > 1e-15:
        raise AssertionError("Stability module not giving arbitragable prices")
    # execution price should only be higher than pool by fee, since we are trading max_sell_amt
    if (buy_amt * (1 - buy_fee) - pool_sell_amt)/pool_sell_amt > 1e-15:
        raise AssertionError("Stability module not giving correct buy fee")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    sell_price = st.floats(min_value=1, max_value=1.1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
)
def test_buy_hollar_from_stability_module(ratios, buyback_speed, sell_price, buy_tkn_i):
    # stability module should work with this params
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)

    pools = [usdt_pool, usdc_pool]
    pool_buy = pools[buy_tkn_i]
    tkn_sell = list(liquidity.keys())[buy_tkn_i]
    tkn_buy = 'HOLLAR'
    max_buy_price = 0.999
    buy_fee = 0.0001
    init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    # test sell USDT
    hsm = copy.deepcopy(init_hsm)
    sell_amt = 1_000
    agent = Agent(holdings = {tkn_sell: sell_amt})
    hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
    buy_amt = agent.get_holdings(tkn_buy)
    if agent.validate_holdings(tkn_sell):
        raise ValueError("Agent should have 0 holdings after selling")
    if agent.get_holdings(tkn_buy) != sell_amt / sell_price:
        raise ValueError("Agent has incorrect amount of HOLLAR")
    if abs(sell_amt - (hsm.liquidity[tkn_sell] - init_hsm.liquidity[tkn_sell])) / hsm.liquidity[tkn_sell] >= 1e-15:
        raise ValueError("Stability module did not deduct correct amount of tokens")

    # test buy HOLLAR
    buy_hsm = copy.deepcopy(init_hsm)
    buy_agent = Agent(enforce_holdings=False)
    buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_amt)
    if buy_agent.get_holdings(tkn_buy) != buy_amt:
        raise ValueError("Agent has incorrect amount of HOLLAR")
    if abs(-buy_agent.get_holdings(tkn_sell) - sell_amt) / pool_buy.liquidity[tkn_sell] >= 1e-15:
        raise ValueError("Agent sold incorrect amount")
    if abs(sell_amt - (hsm.liquidity[tkn_sell] - init_hsm.liquidity[tkn_sell])) / buy_hsm.liquidity[tkn_sell] >= 1e-15:
        raise ValueError("Stability module did not deduct correct amount of tokens")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    sell_extra = st.floats(min_value=1e-15, max_value=10),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
)
def test_large_trade_fails(ratios, buyback_speed, max_buy_price, sell_extra, buy_tkn_i):
    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    tkn_buy = list(liquidity.keys())[buy_tkn_i]
    tkn_sell = 'HOLLAR'
    buy_fee = 0.0001
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    max_sell_amt, buy_price = hsm.get_buy_params(tkn_buy)
    sell_amt = max_sell_amt * (1 + sell_extra)
    agent = Agent(holdings = {tkn_sell: sell_amt})
    hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
    if not hsm.fail:
        raise AssertionError("Swap should have failed")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
    liquidity = st.lists(st.floats(min_value=1, max_value=1_000_000), min_size=2, max_size=2),
)
def test_insufficient_liquidity(ratios, buyback_speed, max_buy_price, buy_tkn_i, liquidity):
    liquidity = {'USDT': liquidity[0], 'USDC': liquidity[1]}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    sell_price = 1.001
    tkn = list(liquidity.keys())[buy_tkn_i]
    buy_fee = 0.0001
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    max_sell_amt, buy_price = hsm.get_buy_params(tkn)
    if (max_sell_amt * buy_price - hsm.liquidity[tkn]) / hsm.liquidity[tkn] > 1e-15:
        raise ValueError("Stability module should not be able to sell more than it has")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/10_000, max_value=1),
    max_buy_price = st.floats(min_value=0.999, max_value=1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
    buy_fee = st.floats(min_value=0.0001, max_value=0.01),
)
def test_arb_loop_known_profitable(ratios, buyback_speed, max_buy_price, buy_tkn_i, buy_fee):
    sell_price = 1.001

    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    tkn = list(liquidity.keys())[buy_tkn_i]
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    # signature of arb function is arb(self, agent: Agent, tkn: str) -> None:
    agent = Agent()
    hsm.arb(agent, tkn)
    # arb function should result in agent's USDT holdings going up
    if not agent.validate_holdings(tkn):
        raise ValueError("Agent should have positive USDT holdings after arb")
    # agent should have no HOLLAR after arb
    if agent.validate_holdings('HOLLAR'):
        raise ValueError("Agent should have 0 HOLLAR holdings after arb")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price = st.floats(min_value=0.99, max_value=1),
    buy_tkn_i = st.integers(min_value=0, max_value=1),
    buy_fee = st.floats(min_value=0, max_value=0.01),
)
def test_arb_loop_known(ratios, buyback_speed, max_buy_price, buy_tkn_i, buy_fee):
    sell_price = 1.001

    liquidity = {'USDT': 1_000_000, 'USDC': 1_000_000}
    tkn = list(liquidity.keys())[buy_tkn_i]
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    usdc_pool = StableSwapPoolState(tokens={'USDC': ratios[1] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100, trade_fee=0.0001, precision=1e-8)
    pools = [usdt_pool, usdc_pool]
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_price, max_buy_price, buy_fee)

    # signature of arb function is arb(self, agent: Agent, tkn: str) -> None:
    agent = Agent()
    hsm.arb(agent, tkn)
    # agent should have no HOLLAR after arb
    if agent.validate_holdings('HOLLAR'):
        raise ValueError("Agent should have 0 HOLLAR holdings after arb")
