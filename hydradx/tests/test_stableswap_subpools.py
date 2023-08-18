import hydradx.model.amm.omnipool_amm as oamm
import hydradx.model.amm.stableswap_amm as ssamm
from hydradx.model.amm.agents import Agent
from hydradx.tests.strategies_omnipool import omnipool_config
from hypothesis import given, settings
import pytest
from hydradx.tests.test_stableswap import stable_swap_equation


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
@settings(deadline=500)
def test_buy_from_stable_swap(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    # deposit stable pool shares into omnipool
    stable_shares = stable_pool.unique_id

    # agent holds some of everything
    agent = Agent(holdings={tkn: 10000000000 for tkn in initial_state.asset_list + stable_pool.asset_list})
    # attempt buying an asset from the stableswap pool
    tkn_buy = stable_pool.asset_list[0]
    tkn_sell = initial_state.asset_list[2]
    buy_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    new_d = new_stable_pool.calculate_d()
    d = stable_pool.calculate_d()
    if not (stable_swap_equation(
            new_d,
            new_stable_pool.amplification,
            new_stable_pool.n_coins,
            new_stable_pool.liquidity.values()
    )):
        raise AssertionError("Stableswap equation didn't hold.")
    if not (
            d * new_stable_pool.shares <=
            new_d * stable_pool.shares
    ):
        raise AssertionError("Shares/invariant ratio changed in the wrong direction.")
    if (
            (new_stable_pool.shares - stable_pool.shares) * d * (1 - stable_pool.trade_fee) !=
            pytest.approx(stable_pool.shares * (new_d - d))
    ):
        raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (
            new_state.liquidity[stable_shares] + stable_pool.shares !=
            pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_shares])
    ):
        raise AssertionError("Shares before and after trade don't add up.")
    if new_state.lrna_imbalance > 0:
        raise AssertionError('LRNA imbalance should be negative.')
    execution_price = (
            (agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    _, lesser_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity - 1
    )
    lesser_execution_price = (
            (agent.holdings[tkn_sell] - lesser_trade_agent.holdings[tkn_sell]) /
            (lesser_trade_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    if not lesser_execution_price < execution_price:
        raise AssertionError(f"Execution price did not decrease with smaller trade")
    if new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy] != buy_quantity:
        raise AssertionError('Agent did not get exactly the amount they specified.')


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
def test_sell_stableswap_for_omnipool(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    stable_shares = stable_pool.unique_id
    # agent holds some of everything
    agent = Agent(holdings={tkn: 10000000000 for tkn in initial_state.asset_list + stable_pool.asset_list})
    # attempt buying an asset from the stableswap pool
    tkn_buy = initial_state.asset_list[2]
    tkn_sell = stable_pool.asset_list[0]
    sell_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    if not (stable_swap_equation(
            new_stable_pool.calculate_d(),
            new_stable_pool.amplification,
            new_stable_pool.n_coins,
            new_stable_pool.liquidity.values()
    )):
        raise AssertionError("Stableswap equation didn't hold.")
    if not (
            stable_pool.calculate_d() * new_stable_pool.shares ==
            pytest.approx(new_stable_pool.calculate_d() * stable_pool.shares)
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    if (
            (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d() !=
            pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
    ):
        raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (
            new_state.liquidity[stable_shares] + stable_pool.shares !=
            pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_shares])
    ):
        raise AssertionError("Shares before and after trade don't add up.")
    execution_price = (
            (agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    _, lesser_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity / 2
    )
    lesser_execution_price = (
            (agent.holdings[tkn_sell] - lesser_trade_agent.holdings[tkn_sell]) /
            (lesser_trade_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    if not min(execution_price - lesser_execution_price, 0) == pytest.approx(0):
        raise AssertionError(f"Execution price did not decrease with smaller trade")
    if agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell] != sell_quantity:
        raise AssertionError('Agent did not sell exactly the amount they specified.')


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
def test_buy_omnipool_with_stable_swap(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    stable_shares = stable_pool.unique_id

    # agent holds some of everything
    agent = Agent(holdings={tkn: 1000000000000000 for tkn in initial_state.asset_list + stable_pool.asset_list})
    # attempt buying an asset from the stableswap pool
    tkn_buy = initial_state.asset_list[2]
    tkn_sell = stable_pool.asset_list[0]
    buy_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    if not (
            stable_pool.calculate_d() * new_stable_pool.shares ==
            pytest.approx(new_stable_pool.calculate_d() * stable_pool.shares)
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    if (
            new_stable_pool.shares * stable_pool.calculate_d() >
            stable_pool.shares * new_stable_pool.calculate_d()
    ):
        raise AssertionError("New_shares * D changed in the wrong direction.")
    if (
            new_state.liquidity[stable_shares] + stable_pool.shares !=
            pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_shares])
    ):
        raise AssertionError("Shares before and after trade don't add up.")
    execution_price = (
            (agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    _, lesser_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity - 1
    )
    lesser_execution_price = (
            (agent.holdings[tkn_sell] - lesser_trade_agent.holdings[tkn_sell]) /
            (lesser_trade_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    if not lesser_execution_price < execution_price:
        raise AssertionError(f"Execution price did not decrease with smaller trade")
    if new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy] != buy_quantity:
        raise AssertionError('Agent did not get exactly the amount they specified.')


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
def test_sell_omnipool_for_stable_swap(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    stable_shares = stable_pool.unique_id

    # agent holds some of everything
    agent = Agent(holdings={tkn: 1000000000000000 for tkn in initial_state.asset_list + stable_pool.asset_list})
    # attempt buying an asset from the stableswap pool
    tkn_buy = stable_pool.asset_list[0]
    tkn_sell = initial_state.asset_list[2]
    sell_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    if (
            stable_pool.calculate_d() * new_stable_pool.shares >
            new_stable_pool.calculate_d() * stable_pool.shares
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    # commented out because withdrawal fees and trade fees are no longer (close to) equivalent
    # since the new execute_remove_liquidity function does not use the trade fee in the same way
    #
    # if (
    #         (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d()
    #         * (1 - stable_pool.trade_fee) !=
    #         pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
    # ):
    #     raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (
            new_state.liquidity[stable_shares] + stable_pool.shares !=
            pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_shares])
    ):
        raise AssertionError("Shares before and after trade don't add up.")
    execution_price = (
            (agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    _, lesser_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity - 1
    )
    lesser_execution_price = (
            (agent.holdings[tkn_sell] - lesser_trade_agent.holdings[tkn_sell]) /
            (lesser_trade_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    if not lesser_execution_price < execution_price:
        raise AssertionError(f"Execution price did not decrease with smaller trade")
    if agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell] != sell_quantity:
        raise AssertionError('Agent did not get exactly the amount they specified.')


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
def test_buy_stableswap_with_LRNA(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    # stable_shares = stable_pool.unique_id
    agent = Agent(holdings={tkn: 1000000000000000 for tkn in initial_state.asset_list + ['LRNA']})
    # attempt buying an asset from the stableswap pool
    tkn_buy = stable_pool.asset_list[0]
    agent.holdings.update({tkn_buy: 0})
    tkn_sell = 'LRNA'
    buy_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    if (
            round(stable_pool.calculate_d() * new_stable_pool.shares, 12) >
            round(new_stable_pool.calculate_d() * stable_pool.shares, 12)
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    if (
            (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d()
            * (1 - stable_pool.trade_fee) !=
            pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
    ):
        raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (new_state.liquidity[stable_pool.unique_id] + stable_pool.shares
            != new_stable_pool.shares + initial_state.liquidity[stable_pool.unique_id]):
        raise AssertionError("Shares before and after trade don't add up.")

    delta_qi = new_state.lrna[stable_pool.unique_id] - initial_state.lrna[stable_pool.unique_id]
    qi_arb = (
            initial_state.lrna[stable_pool.unique_id] + delta_qi
            * initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total
    )
    ri_arb = initial_state.liquidity[stable_pool.unique_id] * initial_state.lrna_total / new_state.lrna_total

    if (
            (initial_state.lrna[stable_pool.unique_id] + initial_state.lrna_imbalance
             * (initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)) * ri_arb
    ) != pytest.approx(
        (qi_arb + new_state.lrna_imbalance * (qi_arb / new_state.lrna_total))
        * initial_state.liquidity[stable_pool.unique_id]
    ):
        raise AssertionError("LRNA imbalance incorrect.")
    execution_price = (
            (agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    _, lesser_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity - 1
    )
    lesser_execution_price = (
            (agent.holdings[tkn_sell] - lesser_trade_agent.holdings[tkn_sell]) /
            (lesser_trade_agent.holdings[tkn_buy] - agent.holdings[tkn_buy])
    )
    if not lesser_execution_price < execution_price:
        raise AssertionError(f"Execution price did not decrease with smaller trade")
    if new_agent.holdings[tkn_buy] - agent.holdings[tkn_buy] != buy_quantity:
        raise AssertionError('Agent did not get exactly the amount they specified.')


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
def test_sell_LRNA_for_stableswap(initial_state: oamm.OmnipoolState):
    stable_pool: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    agent = Agent(holdings={tkn: 1000000000000000 for tkn in initial_state.asset_list + ['LRNA']})
    # attempt buying an asset from the stableswap pool
    tkn_buy = stable_pool.asset_list[0]
    tkn_sell = 'LRNA'
    sell_quantity = 10
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity
    )
    new_stable_pool: oamm.StableSwapPoolState = new_state.sub_pools['stableswap']
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return
    if not (
            stable_pool.calculate_d() * new_stable_pool.shares ==
            pytest.approx(new_stable_pool.calculate_d() * stable_pool.shares)
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    if (
            new_stable_pool.shares * stable_pool.calculate_d() !=
            pytest.approx(stable_pool.shares * new_stable_pool.calculate_d())
    ):
        raise AssertionError("New_shares * D did not yield expected result.")
    # commented out because withdrawal fees and trade fees are no longer (close to) equivalent
    # since the new execute_remove_liquidity function does not use the trade fee in the same way
    #
    # if (
    #         (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d()
    #         * (1 - stable_pool.trade_fee) !=
    #         pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
    # ):
    #     raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (
            new_state.liquidity[stable_pool.unique_id] + stable_pool.shares
            != pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_pool.unique_id])
    ):
        raise AssertionError("Shares in stable pool and omnipool do not add up.")
    if (
            new_state.liquidity[stable_pool.unique_id] *
            (initial_state.lrna_total + initial_state.lrna_imbalance *
             initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)
    ) != pytest.approx(
        new_state.liquidity[stable_pool.unique_id] *
        (initial_state.lrna_total + initial_state.lrna_imbalance *
         initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)
    ):
        raise AssertionError("LRNA imbalance incorrect.")


@given(omnipool_config(
    token_count=3,
    sub_pools={'stableswap1': {'trade_fee': 0}, 'stableswap2': {'trade_fee': 0}},
    lrna_fee=0,
    asset_fee=0
))
def test_buy_stableswap_for_stableswap(initial_state: oamm.OmnipoolState):
    pool_buy: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap1']
    pool_sell: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap2']
    # attempt buying an asset from the stableswap pool
    tkn_buy = pool_buy.asset_list[0]
    tkn_sell = pool_sell.asset_list[1]
    initial_agent = Agent(holdings={tkn_sell: 1000000, tkn_buy: 1000000})
    buy_quantity = 1
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity
    )
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return

    new_pool_buy: ssamm.StableSwapPoolState = new_state.sub_pools['stableswap1']
    new_pool_sell: ssamm.StableSwapPoolState = new_state.sub_pools['stableswap2']
    if not new_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy] == buy_quantity:
        raise AssertionError('Agent did not get what it paid for, but transaction passed!')
    if (
            round(new_state.lrna[new_pool_buy.unique_id] * new_state.liquidity[new_pool_buy.unique_id], 12)
            < round(initial_state.lrna[pool_buy.unique_id] * initial_state.liquidity[pool_buy.unique_id], 12)
    ):
        raise AssertionError('Pool_buy price moved in the wrong direction.')
    if (
            round(new_state.lrna[new_pool_sell.unique_id] * new_state.liquidity[new_pool_sell.unique_id], 12)
            < round(initial_state.lrna[pool_sell.unique_id] * initial_state.liquidity[pool_sell.unique_id], 12)
    ):
        raise AssertionError('Pool_sell price moved in the wrong direction.')

    if round(new_pool_buy.shares * pool_buy.d, 12) > round(new_pool_buy.d * pool_buy.shares, 12):
        raise AssertionError('Shares * invariant inconsistent in pool_buy.')
    if round(new_pool_sell.shares * pool_sell.d, 12) > round(new_pool_sell.d * pool_sell.shares, 12):
        raise AssertionError('Shares * invariant inconsistent in pool_sell.')

    if (
            new_state.liquidity[pool_buy.unique_id] + pool_buy.shares
            != pytest.approx(initial_state.liquidity[pool_buy.unique_id] + new_pool_buy.shares)
    ):
        raise AssertionError("Omnipool and subpool shares before and after don't add up in pool_buy.")
    if (
            new_state.liquidity[pool_sell.unique_id] + pool_sell.shares
            != pytest.approx(initial_state.liquidity[pool_sell.unique_id] + new_pool_sell.shares)
    ):
        raise AssertionError("Omnipool and subpool shares before and after don't add up in pool_sell.")
    sell_quantity = initial_agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]
    before_trade_state, before_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=buy_quantity / 1000
    )
    # print(f'sell quantity {sell_quantity}')
    after_trade_state, after_trade_agent = oamm.swap(
        old_state=new_state,
        old_agent=new_agent,
        tkn_buy=tkn_sell,
        tkn_sell=tkn_buy,
        buy_quantity=sell_quantity / 1000  # initial_agent.holdings[tkn_buy] - before_trade_agent.holdings[tkn_buy]
    )

    if before_trade_state.fail or after_trade_state.fail:
        return

    spot_price_before = (
            (initial_agent.holdings[tkn_sell] - before_trade_agent.holdings[tkn_sell]) /
            (before_trade_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy])
    )
    spot_price_after = (
            (after_trade_agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - after_trade_agent.holdings[tkn_buy])
    )
    execution_price = sell_quantity / (new_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy])
    if not (spot_price_after > execution_price > spot_price_before):
        raise AssertionError('Execution price out of bounds.')


@given(omnipool_config(
    token_count=3,
    sub_pools={'stableswap1': {'trade_fee': 0}, 'stableswap2': {'trade_fee': 0}},
    lrna_fee=0,
    asset_fee=0
))
def test_sell_stableswap_for_stableswap(initial_state: oamm.OmnipoolState):
    pool_buy: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap1']
    pool_sell: oamm.StableSwapPoolState = initial_state.sub_pools['stableswap2']
    # attempt selling an asset to the stableswap pool
    tkn_buy = pool_buy.asset_list[0]
    tkn_sell = pool_sell.asset_list[1]
    initial_agent = Agent(holdings={tkn_sell: 1000000, tkn_buy: 1000000})
    sell_quantity = 1
    new_state, new_agent = oamm.swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity
    )
    if new_state.fail:
        # transaction failed, doesn't mean there is anything wrong with the mechanism
        return

    new_pool_buy: ssamm.StableSwapPoolState = new_state.sub_pools['stableswap1']
    new_pool_sell: ssamm.StableSwapPoolState = new_state.sub_pools['stableswap2']
    if not initial_agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell] == sell_quantity:
        raise AssertionError('Agent did not get what it paid for, but transaction passed!')
    if (
            round(new_state.lrna[new_pool_buy.unique_id] * new_state.liquidity[new_pool_buy.unique_id], 12)
            < round(initial_state.lrna[pool_buy.unique_id] * initial_state.liquidity[pool_buy.unique_id], 12)
    ):
        raise AssertionError('Pool_buy rice moved in the wrong direction.')
    if (
            round(new_state.lrna[new_pool_sell.unique_id] * new_state.liquidity[new_pool_sell.unique_id], 12)
            < round(initial_state.lrna[pool_sell.unique_id] * initial_state.liquidity[pool_sell.unique_id], 12)
    ):
        raise AssertionError('Pool_sell price moved in the wrong direction.')
    if new_pool_buy.shares * pool_buy.d != pytest.approx(new_pool_buy.d * pool_buy.shares):
        raise AssertionError('Shares * invariant inconsistent in pool_buy.')
    if new_pool_sell.shares * pool_sell.d != pytest.approx(new_pool_sell.d * pool_sell.shares):
        raise AssertionError('Shares * invariant inconsistent in pool_sell.')

    if (
            new_state.liquidity[pool_buy.unique_id] + pool_buy.shares
            != pytest.approx(initial_state.liquidity[pool_buy.unique_id] + new_pool_buy.shares)
    ):
        raise AssertionError("Omnipool and subpool shares before and after don't add up in pool_buy.")
    if (
            new_state.liquidity[pool_sell.unique_id] + pool_sell.shares
            != pytest.approx(initial_state.liquidity[pool_sell.unique_id] + new_pool_sell.shares)
    ):
        raise AssertionError("Omnipool and subpool shares before and after don't add up in pool_sell.")
    buy_quantity = initial_agent.holdings[tkn_buy] - new_agent.holdings[tkn_buy]
    before_trade_state, before_trade_agent = oamm.swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=sell_quantity / 1000
    )
    # print(f'sell quantity {sell_quantity}')
    after_trade_state, after_trade_agent = oamm.swap(
        old_state=new_state,
        old_agent=new_agent,
        tkn_buy=tkn_sell,
        tkn_sell=tkn_buy,
        buy_quantity=buy_quantity / 1000  # initial_agent.holdings[tkn_buy] - before_trade_agent.holdings[tkn_buy]
    )

    if before_trade_state.fail or after_trade_state.fail:
        return

    spot_price_before = (
            (initial_agent.holdings[tkn_sell] - before_trade_agent.holdings[tkn_sell]) /
            (before_trade_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy])
    )
    spot_price_after = (
            (after_trade_agent.holdings[tkn_sell] - new_agent.holdings[tkn_sell]) /
            (new_agent.holdings[tkn_buy] - after_trade_agent.holdings[tkn_buy])
    )
    execution_price = sell_quantity / (new_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy])
    if not (spot_price_after > execution_price > spot_price_before):
        raise AssertionError('Execution price out of bounds.')




@given(omnipool_config(
    asset_dict={
        'USD': {'liquidity': 1000, 'LRNA': 1000},
        'HDX': {'liquidity': 1000, 'LRNA': 1000},
        'DAI': {'liquidity': 1000, 'LRNA': 1000}
    },
    sub_pools={'stableswap': {'token_count': 2}}
))
def test_migrate_asset(initial_state: oamm.OmnipoolState):
    s = 'stableswap'
    i = 'DAI'
    initial_agent = Agent(
        holdings={'DAI': 100}
    )
    new_state = oamm.migrate(initial_state, tkn_migrate='DAI', sub_pool_id='stableswap')
    if (
            pytest.approx(new_state.lrna[s] * new_state.protocol_shares[s] / new_state.shares[s])
            != initial_state.lrna[i] * initial_state.protocol_shares[i] / initial_state.shares[i]
            + initial_state.lrna[s] * initial_state.protocol_shares[s] / initial_state.shares[s]
    ):
        raise AssertionError("Protocol didn't get the right number of shares.")
    new_sub_pool: ssamm.StableSwapPoolState = new_state.sub_pools[s]
    if new_state.shares[s] != new_sub_pool.shares:
        raise AssertionError("Subpool and Omnipool shares aren't equal.")

    lp_state, lp = oamm.add_liquidity(
        old_state=initial_state,
        old_agent=initial_agent,
        quantity=100, tkn_add='DAI'
    )
    temp_state = oamm.execute_migrate_asset(lp_state.copy(), 'DAI', 'stableswap')
    migrated_state, migrated_lp = oamm.execute_migrate_lp(
        state=temp_state,
        agent=lp.copy(),
        sub_pool_id='stableswap',
        tkn_migrate='DAI'
    )
    migrated_sub_pool: ssamm.StableSwapPoolState = migrated_state.sub_pools[s]
    pui = migrated_sub_pool.conversion_metrics['DAI']['price']
    pa = lp.share_prices[(initial_state.unique_id, 'DAI')]
    pb = migrated_lp.share_prices['stableswap']
    if pui * pb != pa:
        raise AssertionError("Share prices didn't come out right.")
    sa = lp.holdings[(initial_state.unique_id, 'DAI')]
    sb = migrated_lp.holdings[s]
    d_si = migrated_sub_pool.conversion_metrics[i]['old_shares']
    d_ss = migrated_state.shares[s] - initial_state.shares[s]
    if sb / sa != pytest.approx(abs(d_ss / d_si), rel=1e-10):
        raise AssertionError("Share quantities didn't come out right.")


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0, withdrawal_fee=False))
def test_migration_scenarios_no_withdrawal_fee(initial_state: oamm.OmnipoolState):
    asset1 = initial_state.asset_list[2]
    asset2 = 'DAI'
    asset3 = 'USDC'
    initial_state.asset_list.append(asset2)
    initial_state.liquidity[asset2] = initial_state.liquidity[asset1] * 1.1
    initial_state.lrna[asset2] = initial_state.lrna[asset1] * 1.1
    initial_state.shares[asset2] = initial_state.shares[asset1] * 1.1
    initial_state.protocol_shares[asset2] = initial_state.protocol_shares[asset1] * 1.1
    initial_state.asset_list.append(asset3)
    initial_state.liquidity[asset3] = initial_state.liquidity[asset1] * 1.1
    initial_state.lrna[asset3] = initial_state.lrna[asset1] * 1.1
    initial_state.shares[asset3] = initial_state.shares[asset1] * 1.1
    initial_state.protocol_shares[asset3] = initial_state.protocol_shares[asset1] * 1.1
    initial_state.weight_cap[asset3] = 1
    initial_lp = Agent(
        holdings={
            asset1: initial_state.liquidity[asset2] - initial_state.liquidity[asset1],
            asset2: 0,
            asset3: 0,
            'LRNA': 0
        }
    )
    initial_state, initial_lp = oamm.add_liquidity(
        initial_state, initial_lp,
        quantity=initial_lp.holdings[asset1], tkn_add=asset1
    )
    # scenario 1: immediately remove liquidity
    s1_state, s1_lp = oamm.remove_liquidity(
        initial_state, initial_lp,
        quantity=initial_lp.holdings[(initial_state.unique_id, asset1)],
        tkn_remove=asset1
    )

    q1 = s1_lp.holdings['LRNA']
    r1 = s1_lp.holdings[asset1]

    # scenario 2: migrate assets to subpool, then withdraw an equal percentage of each
    migrate_state = oamm.execute_create_sub_pool(
        state=initial_state.copy(),
        tkns_migrate=[asset1, asset2, asset3],
        sub_pool_id='stableswap',
        amplification=10
    ).update()
    migrate_state, migrate_lp = oamm.execute_migrate_lp(
        state=migrate_state,
        agent=initial_lp.copy(),
        sub_pool_id='stableswap',
        tkn_migrate=asset1
    )
    s2_state = migrate_state.copy()
    s2_lp = migrate_lp.copy()
    s2_sub_pool: ssamm.StableSwapPoolState = s2_state.sub_pools['stableswap']
    withdraw_fraction = migrate_lp.holdings['stableswap'] / s2_sub_pool.shares
    # withdraw an equal fraction of each asset from the subpool
    for tkn in s2_sub_pool.asset_list:
        delta_tkn = s2_sub_pool.liquidity[tkn] * withdraw_fraction
        s2_sub_pool.liquidity[tkn] -= delta_tkn
        s2_lp.holdings[tkn] += delta_tkn
    s2_sub_pool.shares *= 1 - withdraw_fraction
    s2_lp.holdings['stableswap'] = 0

    q2 = s2_lp.holdings['LRNA']
    r2 = s2_lp.holdings[asset1] + s2_lp.holdings[asset2] + s2_lp.holdings[asset3]

    # scenario 3: sell all stableswap assets withdrawn in scenario 2 except asset1, buy asset 1
    s3_state = s2_state.copy()
    s3_lp = s2_lp.copy()
    s3_sub_pool = s3_state.sub_pools['stableswap']
    for tkn in [asset2, asset3]:
        ssamm.execute_swap(
            state=s3_sub_pool,
            agent=s3_lp,
            tkn_sell=tkn,
            tkn_buy=asset1,
            sell_quantity=s3_lp.holdings[tkn]
        )

    r3 = s3_lp.holdings[asset1]

    # scenario 4: withdraw only asset1
    s4_state, s4_lp = oamm.remove_liquidity(
        migrate_state, migrate_lp,
        quantity=migrate_lp.holdings['stableswap'],
        tkn_remove=asset1
    )

    q4 = s4_lp.holdings['LRNA']
    r4 = s4_lp.holdings[asset1]

    if q1 != pytest.approx(q2) or r1 != pytest.approx(r2) or r4 != pytest.approx(r3) or q4 != pytest.approx(q2):
        raise AssertionError("Equivalent transactions didn't come out the same.")

    # test all the same scenarios again, but with an already-existing pool and 4 assets
    initial_state = migrate_state.copy()
    initial_sub_pool = initial_state.sub_pools['stableswap']
    asset4 = "superstableUSDcoin"

    initial_state.add_token(
        tkn=asset4,
        liquidity=initial_sub_pool.liquidity[asset1],
        lrna=initial_state.lrna['stableswap'] / 3,
        shares=initial_state.lrna['stableswap'] / 3,
        protocol_shares=initial_state.lrna['stableswap'] / 3
    )
    initial_state.update()

    initial_lp = Agent(
        holdings={
            asset1: 0,
            asset2: 0,
            asset3: 0,
            asset4: initial_sub_pool.liquidity[asset1] * 0.1,
            'LRNA': 0
        }
    )

    lp_state, invested_lp = oamm.add_liquidity(
        initial_state, initial_lp,
        quantity=initial_lp.holdings[asset4], tkn_add=asset4
    )
    s1_state, s1_lp = oamm.remove_liquidity(
        lp_state, invested_lp,
        quantity=invested_lp.holdings[(initial_state.unique_id, asset4)],
        tkn_remove=asset4
    )

    q1 = s1_lp.holdings['LRNA']
    r1 = s1_lp.holdings[asset4]

    migrate_state = oamm.execute_migrate_asset(
        state=lp_state.copy(),
        tkn_migrate=asset4,
        sub_pool_id='stableswap'
    ).update()
    migrate_state, migrate_lp = oamm.execute_migrate_lp(
        state=migrate_state,
        agent=invested_lp.copy(),
        sub_pool_id='stableswap',
        tkn_migrate=asset4
    )
    s2_state = migrate_state.copy()
    s2_lp = migrate_lp.copy()
    s2_sub_pool: ssamm.StableSwapPoolState = s2_state.sub_pools['stableswap']
    withdraw_fraction = migrate_lp.holdings['stableswap'] / s2_sub_pool.shares
    # withdraw an equal fraction of each asset from the subpool
    for tkn in s2_sub_pool.asset_list:
        delta_tkn = s2_sub_pool.liquidity[tkn] * withdraw_fraction
        s2_sub_pool.liquidity[tkn] -= delta_tkn
        s2_lp.holdings[tkn] += delta_tkn
    s2_sub_pool.shares *= 1 - withdraw_fraction
    s2_lp.holdings['stableswap'] = 0

    q2 = s2_lp.holdings['LRNA']
    r2 = sum([s2_lp.holdings[tkn] for tkn in s2_sub_pool.asset_list])

    s3_state = s2_state.copy()
    s3_lp = s2_lp.copy()
    s3_sub_pool = s3_state.sub_pools['stableswap']
    for tkn in [asset1, asset2, asset3]:
        ssamm.execute_swap(
            state=s3_sub_pool,
            agent=s3_lp,
            tkn_sell=tkn,
            tkn_buy=asset4,
            sell_quantity=s3_lp.holdings[tkn]
        )

    r3 = s3_lp.holdings[asset4]

    s3_state, s3_lp = oamm.remove_liquidity(
        migrate_state, migrate_lp,
        quantity=migrate_lp.holdings['stableswap'],
        tkn_remove=asset4
    )

    q4 = s3_lp.holdings['LRNA']
    r4 = s3_lp.holdings[asset4]

    if q1 != pytest.approx(q2) or r1 != pytest.approx(r2) or r4 != pytest.approx(r3) or q4 != pytest.approx(q2):
        raise AssertionError("Equivalent transactions didn't come out the same.")


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0, sub_pools={'stableswap': {}}))
@settings(deadline=500)
def test_add_stableswap_liquidity(initial_state: oamm.OmnipoolState):
    stable_pool: ssamm.StableSwapPoolState = initial_state.sub_pools['stableswap']
    agent = Agent(
        holdings={stable_pool.asset_list[0]: 1000}
    )
    new_state, new_agent = oamm.add_liquidity(
        initial_state, agent,
        quantity=1000, tkn_add=stable_pool.asset_list[0]
    )

    if (initial_state.unique_id, stable_pool.unique_id) not in new_agent.holdings:
        raise ValueError("Agent did not receive shares.")
    if not (new_agent.holdings[(initial_state.unique_id, stable_pool.unique_id)] > 0):
        raise AssertionError("Sanity check failed.")
