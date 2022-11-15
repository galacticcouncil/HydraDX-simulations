import copy
import pytest
import random
from hypothesis import given, strategies as st, assume
from mpmath import mpf, mp

from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.tests.test_stableswap import stableswap_config, stable_swap_equation, StableSwapPoolState

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)

mp.dps = 50


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA_price': 1 / usd_price_lrna
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        } for _ in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        sub_pools: dict = None
) -> oamm.OmnipoolState:
    asset_dict = asset_dict or draw(assets_config(token_count))
    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
        sub_pools={
            name: draw(stableswap_config(
                asset_dict=pool['asset_dict'] if 'asset_dict' in pool else None,
                token_count=pool['token_count'] if 'token_count' in pool else None,
                trade_fee=pool['trade_fee'] if 'trade_fee' in pool else None,
                amplification=pool['amplification'] if 'amplification' in pool else None,
                unique_id=name
            ))
            for name, pool in sub_pools.items()
        } if sub_pools else None
    )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy)
    return test_state


@given(omnipool_config(asset_fee=0, lrna_fee=0, token_count=3), asset_quantity_strategy)
def test_swap_lrna_delta_Qi_respects_invariant(d: oamm.OmnipoolState, delta_ri: float):
    i = d.asset_list[-1]
    assume(i in d.asset_list)
    assume(d.liquidity[i] > delta_ri > -d.liquidity[i])
    d2 = copy.deepcopy(d)
    delta_Qi = oamm.swap_lrna_delta_Qi(d, delta_ri, i)
    d2.liquidity[i] += delta_ri
    d2.lrna[i] += delta_Qi

    # Test basics
    for j in d2.asset_list:
        assert d2.liquidity[j] > 0
        assert d2.lrna[j] > 0
    assert not (delta_ri > 0 and delta_Qi > 0)
    assert not (delta_ri < 0 and delta_Qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


@given(omnipool_config(asset_fee=0, lrna_fee=0, token_count=3), asset_quantity_strategy)
def test_swap_lrna_delta_Ri_respects_invariant(d: oamm.OmnipoolState, delta_qi: float):
    i = d.asset_list[-1]
    assume(i in d.asset_list)
    assume(d.lrna[i] > delta_qi > -d.lrna[i])
    d2 = copy.deepcopy(d)
    delta_Ri = oamm.swap_lrna_delta_Ri(d, delta_qi, i)
    d2.lrna[i] += delta_qi
    d2.liquidity[i] += delta_Ri

    # Test basics
    for j in d.asset_list:
        assert d2.liquidity[j] > 0
        assert d2.lrna[j] > 0
    assert not (delta_Ri > 0 and delta_qi > 0)
    assert not (delta_Ri < 0 and delta_qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


@given(omnipool_config(asset_fee=0, lrna_fee=0))
def test_weights(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    for i in old_state.asset_list:
        assert oamm.weight_i(old_state, i) >= 0
    assert sum([oamm.weight_i(old_state, i) for i in old_state.asset_list]) == pytest.approx(1.0)


@given(omnipool_config())
def test_prices(market_state: oamm.OmnipoolState):
    for i in market_state.asset_list:
        assert oamm.lrna_price(market_state, i) > 0


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    old_agent = Agent(
        holdings={i: 1000 for i in old_state.asset_list}
    )
    i = old_state.asset_list[-1]
    delta_R = 1000

    new_state, new_agents = oamm.add_liquidity(old_state, old_agent, delta_R, i)
    for j in initial_state.asset_list:
        assert oamm.lrna_price(old_state, j) == pytest.approx(oamm.lrna_price(new_state, j))
    if old_state.liquidity[i] / old_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError(f'Price change in {i}'
                             f'({old_state.liquidity[i] / old_state.shares[i]}) -->'
                             f'({pytest.approx(new_state.liquidity[i] / new_state.shares[i])})'
                             )

    if old_state.lrna_imbalance / old_state.lrna_total != \
            pytest.approx(new_state.lrna_imbalance / new_state.lrna_total):
        raise AssertionError(f'LRNA imbalance did not remain constant.')

    # check enforcement of weight caps
    # first assign some weight caps
    for i in initial_state.asset_list:
        initial_state.weight_cap[i] = mpf(min(initial_state.lrna[i] / initial_state.lrna_total * 1.1, 1))

    # calculate what should be the maximum allowable liquidity provision
    max_amount = ((old_state.weight_cap[i] / (1 - old_state.weight_cap[i])
                   * old_state.lrna_total - old_state.lrna[i] / (1 - old_state.weight_cap[i]))
                  / old_state.lrna_price(i))

    if max_amount < 0:
        raise AssertionError('This calculation makes no sense.')  # but actually, it works :)

    # make sure agent has enough funds
    old_agent.holdings[i] = max_amount * 2
    # eliminate general tvl cap, so we can test just the weight cap
    old_state.tvl_cap = float('inf')

    # try one just above and just below the maximum allowable amount
    illegal_state, illegal_agents = oamm.add_liquidity(old_state, old_agent, max_amount * 1.0000001, i)
    if not illegal_state.fail:
        raise AssertionError(f'illegal transaction passed against weight limit in {i}')
    legal_state, legal_agents = oamm.add_liquidity(old_state, old_agent, max_amount * 0.9999999, i)
    if legal_state.fail:
        raise AssertionError(f'legal transaction failed against weight limit in {i} ({new_state.fail})')


@given(omnipool_config(token_count=3))
def test_remove_liquidity(initial_state: oamm.OmnipoolState):
    i = initial_state.asset_list[2]
    initial_agent = Agent(
        holdings={token: mpf(1000) for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.add_liquidity(initial_state, initial_agent, 1000, i)
    p_init = old_state.lrna_price(i)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if old_state.price(j) != pytest.approx(new_state.price(j)):
            raise AssertionError(f'Price change in asset {j}')
    if old_state.liquidity[i] / old_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError('')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = oamm.lrna_price(old_state, i)
    val_withdrawn = piq * delta_r + delta_q
    if (-2 * piq / (piq + p_init) * delta_S / old_state.shares[i] * piq
            * old_state.liquidity[i] != pytest.approx(val_withdrawn)
            and not new_state.fail):
        raise AssertionError('something is wrong')

    if old_state.lrna_imbalance / old_state.lrna_total != \
            pytest.approx(new_state.lrna_imbalance / new_state.lrna_total):
        raise AssertionError(f'LRNA imbalance did not remain constant.')


@given(omnipool_config(token_count=3))
def test_swap_lrna(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']}
    )
    delta_Ra = 1000
    delta_Qa = -1000
    i = old_state.asset_list[2]

    # Test with trader selling asset i
    feeless_state = initial_state.copy()
    feeless_state.lrna_fee = 0
    feeless_state.asset_fee = 0
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, delta_Ra, 0, i)
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise

    # Test with trader selling LRNA
    new_state, new_agent = oamm.swap_lrna(old_state, old_agent, 0, delta_Qa, i)
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, 0, delta_Qa, i)
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise
    for j in old_state.asset_list:
        if min(new_state.liquidity[j] - feeless_swap_state.liquidity[j], 0) != pytest.approx(0):
            raise
    if min(oamm.asset_invariant(new_state, i) / oamm.asset_invariant(old_state, i), 1) != pytest.approx(1):
        raise

    if (new_state.liquidity[i] * old_state.lrna[i] *
        new_state.lrna_total * (old_state.lrna_total + old_state.lrna_imbalance)) != pytest.approx(
        old_state.liquidity[i] * new_state.lrna[i] *
        old_state.lrna_total * (new_state.lrna_total + new_state.lrna_imbalance)
    ):
        raise AssertionError(f'LRNA imbalance is wrong.')

    if (new_state.liquidity[i] + new_agent.holdings[i] != old_state.liquidity[i] + old_agent.holdings[i]
        or new_state.lrna[i] + new_agent.holdings['LRNA'] != old_state.lrna[i] + old_agent.holdings['LRNA']):
        raise AssertionError('System-wide asset total is wrong.')
    # try swapping into LRNA and back to see if that's equivalent


@given(omnipool_config(token_count=3), st.integers(min_value=1, max_value=2))
def test_swap_assets(initial_state: oamm.OmnipoolState, i):
    i_buy = initial_state.asset_list[i]
    old_state = initial_state

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )
    sellable_tokens = len(old_state.asset_list) - 1
    i_sell = old_state.asset_list[i % sellable_tokens + 1]
    delta_R = min(1000, old_state.liquidity[i_sell] / 2, old_state.liquidity[i_buy] / 2)

    # Test with trader selling asset i, no LRNA fee... price should match feeless
    new_state, new_agent = \
        oamm.swap(old_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)

    # create copies of the old state with fees removed
    asset_fee_only_state = old_state.copy()
    asset_fee_only_state.lrna_fee = 0
    feeless_state = asset_fee_only_state.copy()
    feeless_state.asset_fee = 0

    asset_fee_only_state, asset_fee_only_agent = \
        oamm.swap(asset_fee_only_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)
    feeless_state, feeless_agent = \
        oamm.swap(feeless_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)

    for j in old_state.asset_list:
        # assets in pools only go up compared to asset_fee_only_state
        if min(asset_fee_only_state.liquidity[j] - feeless_state.liquidity[j], 0) != pytest.approx(0):
            raise AssertionError("asset in pool {j} is lesser when compared with no-fee case")
        # asset in pool goes up from asset_fee_only_state -> new_state (i.e. introduction of LRNA fee)
        if min(new_state.liquidity[j] - asset_fee_only_state.liquidity[j], 0) != pytest.approx(0):
            raise AssertionError("asset in pool {j} is lesser when LRNA fee is added vs only asset fee")
        # invariant does not decrease
        if min(oamm.asset_invariant(new_state, j) / oamm.asset_invariant(old_state, j), 1) != pytest.approx(1):
            raise AssertionError("invariant ratio less than zero")
        # total quantity of R_i remains unchanged
        if (old_state.liquidity[j] + old_agent.holdings[j]
                != pytest.approx(new_state.liquidity[j] + new_agent.holdings[j])):
            raise AssertionError("total quantity of R[{j}] changed")

    # test that no LRNA is lost
    delta_Qi = new_state.lrna[i_sell] - old_state.lrna[i_sell]
    delta_Qj = new_state.lrna[i_buy] - old_state.lrna[i_buy]
    delta_Qh = new_state.lrna['HDX'] - old_state.lrna['HDX']
    delta_L = new_state.lrna_imbalance - old_state.lrna_imbalance
    if delta_L + delta_Qj + delta_Qi + delta_Qh != pytest.approx(0, abs=1e10):
        raise AssertionError('Some LRNA was lost along the way.')

    delta_out_new = feeless_agent.holdings[i_buy] - old_agent.holdings[i_buy]

    # Test with trader buying asset i, no LRNA fee... price should match feeless
    buy_state = old_state.copy()
    buy_state.lrna_fee = 0
    buy_state.asset_fee = 0
    buy_state, buy_agent = oamm.swap(
        buy_state, old_agent, i_buy, i_sell, buy_quantity=delta_out_new
    )

    for j in old_state.asset_list:
        if not buy_state.liquidity[j] == pytest.approx(feeless_state.liquidity[j]):
            raise AssertionError(f'Liquidity mismatch in {j}')
        if not buy_state.lrna[j] == pytest.approx(feeless_state.lrna[j]):
            raise AssertionError(f'LRNA mismatch in {j}')
        if not (
            old_state.liquidity[j] + old_agent.holdings[j] ==
            pytest.approx(buy_state.liquidity[j] + buy_agent.holdings[j])
        ):
            raise AssertionError('Change in the total quantity of {j}.')
        # assert buy_agent.holdings[j] == pytest.approx(feeless_agent.holdings[j])
        # assert buy_agent.holdings['LRNA'] == pytest.approx(feeless_agent.holdings['LRNA'])


@given(omnipool_config(token_count=3, sub_pools={'stableswap': {}}))
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
    if not(stable_swap_equation(
            new_stable_pool.calculate_d(),
            new_stable_pool.amplification,
            new_stable_pool.n_coins,
            new_stable_pool.liquidity.values()
    )):
        raise AssertionError("Stableswap equation didn't hold.")
    if not (
        stable_pool.calculate_d() * new_stable_pool.shares <=
        new_stable_pool.calculate_d() * stable_pool.shares
    ):
        raise AssertionError("Shares/invariant ratio changed in the wrong direction.")
    if (
        (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d() * (1 - stable_pool.trade_fee) !=
        pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
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
    agent = Agent(holdings={tkn: mpf(10000000000) for tkn in initial_state.asset_list + stable_pool.asset_list})
    # attempt buying an asset from the stableswap pool
    tkn_buy = initial_state.asset_list[2]
    tkn_sell = stable_pool.asset_list[0]
    sell_quantity = mpf(10)
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
    if not(stable_swap_equation(
            new_stable_pool.calculate_d(),
            new_stable_pool.amplification,
            new_stable_pool.n_coins,
            new_stable_pool.liquidity.values()
    )):
        raise AssertionError("Stableswap equation didn't hold.")
    if not(
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
    if not(
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
    if (
        (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d()
        * (1 - stable_pool.trade_fee) !=
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
        stable_pool.calculate_d() * new_stable_pool.shares >
        new_stable_pool.calculate_d() * stable_pool.shares
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
    if (
        (new_state.lrna[stable_pool.unique_id] + new_state.lrna_imbalance
         * new_state.lrna[stable_pool.unique_id] / new_state.lrna_total)
        * initial_state.liquidity[stable_pool.unique_id]
        != pytest.approx(
            (initial_state.lrna[stable_pool.unique_id] + initial_state.lrna_imbalance
             * initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)
            * new_state.liquidity[stable_pool.unique_id]
        )
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
    stable_shares = stable_pool.unique_id
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
    if not(
            stable_pool.calculate_d() * new_stable_pool.shares ==
            pytest.approx(new_stable_pool.calculate_d() * stable_pool.shares)
    ):
        raise AssertionError("Shares/invariant ratio incorrect.")
    if (
            new_stable_pool.shares * stable_pool.calculate_d() !=
            pytest.approx(stable_pool.shares * new_stable_pool.calculate_d())
    ):
        raise AssertionError("New_shares * D did not yield expected result.")
    if (
        (new_stable_pool.shares - stable_pool.shares) * stable_pool.calculate_d()
        * (1 - stable_pool.trade_fee) !=
        pytest.approx(stable_pool.shares * (new_stable_pool.calculate_d() - stable_pool.calculate_d()))
    ):
        raise AssertionError("Delta_shares * D * (1 - fee) did not yield expected result.")
    if (
        new_state.liquidity[stable_pool.unique_id] + stable_pool.shares
        != pytest.approx(new_stable_pool.shares + initial_state.liquidity[stable_pool.unique_id])
    ):
        raise AssertionError("Shares in stable pool and omnipool do not add up.")
    if (new_state.liquidity[stable_pool.unique_id] * (initial_state.lrna_total + initial_state.lrna_imbalance *
        initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)) != pytest.approx(
        new_state.liquidity[stable_pool.unique_id] * (initial_state.lrna_total + initial_state.lrna_imbalance *
        initial_state.lrna[stable_pool.unique_id] / initial_state.lrna_total)
    ):
        raise AssertionError("LRNA imbalance incorrect.")


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
    sub_pool: StableSwapPoolState = initial_state.sub_pools[s]
    new_state = oamm.migrate(initial_state, tkn_migrate='DAI', sub_pool_id='stableswap')
    if (
        pytest.approx(new_state.lrna[s] * new_state.protocol_shares[s] / new_state.shares[s])
        != initial_state.lrna[i] * initial_state.protocol_shares[i] / initial_state.shares[i]
        + initial_state.lrna[s] * initial_state.protocol_shares[s] / initial_state.shares[s]
    ):
        raise AssertionError("Protocol didn't get the right number of shares.")
    new_sub_pool: StableSwapPoolState = new_state.sub_pools[s]
    if new_state.shares[s] != new_sub_pool.shares:
        raise AssertionError("Subpool and Omnipool shares aren't equal.")


@given(omnipool_config(token_count=4), st.floats(min_value=0.1, max_value=1), st.floats(min_value=0.1, max_value=1))
def test_slip_fees(initial_state: oamm.OmnipoolState, lrna_slip_rate: float, asset_slip_rate: float):
    initial_state.lrna_fee = oamm.OmnipoolState.slip_fee(lrna_slip_rate, minimum_fee=0.0001)
    initial_state.asset_fee = oamm.OmnipoolState.slip_fee(asset_slip_rate, minimum_fee=0.0001)
    initial_agent = Agent(holdings={tkn: 1000 for tkn in initial_state.asset_list})
    tkn_buy = initial_state.asset_list[2]
    tkn_sell = initial_state.asset_list[3]
    sell_quantity = 10
    sell_state, sell_agent = oamm.swap(initial_state, initial_agent, tkn_buy, tkn_sell, sell_quantity=sell_quantity)
    split_sell_state, split_sell_agent = initial_state.copy(), initial_agent.copy()
    next_state, next_agent = {}, {}
    for i in range(2):
        next_state[i], next_agent[i] = oamm.swap(
            old_state=split_sell_state,
            old_agent=split_sell_agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity / 2
        )
        split_sell_state, split_sell_agent = next_state[i], next_agent[i]
    if split_sell_agent.holdings[tkn_buy] < sell_agent.holdings[tkn_buy]:
        raise AssertionError('Agent failed to save money by splitting the sell order.')

    buy_quantity = 10
    buy_state, buy_agent = oamm.swap(initial_state, initial_agent, tkn_buy, tkn_sell, buy_quantity=buy_quantity)
    split_buy_state, split_buy_agent = initial_state.copy(), initial_agent.copy()
    next_state, next_agent = {}, {}
    for i in range(2):
        next_state[i], next_agent[i] = oamm.swap(
            old_state=split_buy_state,
            old_agent=split_buy_agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity / 2
        )
        split_buy_state, split_buy_agent = next_state[i], next_agent[i]
    if split_buy_agent.holdings[tkn_sell] < buy_agent.holdings[tkn_sell]:
        raise AssertionError('Agent failed to save money by splitting the buy order.')

    if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
         + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
            != pytest.approx(buy_agent.holdings[tkn_sell] + buy_agent.holdings[tkn_buy]
                             + buy_state.liquidity[tkn_sell] + buy_state.liquidity[tkn_buy])):
        raise AssertionError('Asset quantity is not constant after trade (one-part)')

    if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
         + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
            != pytest.approx(split_buy_agent.holdings[tkn_sell] + split_buy_agent.holdings[tkn_buy]
                             + split_buy_state.liquidity[tkn_sell] + split_buy_state.liquidity[tkn_buy])):
        raise AssertionError('Asset quantity is not constant after trade (two-part)')
