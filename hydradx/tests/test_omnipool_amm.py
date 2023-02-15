import copy
import random

import pytest
from hypothesis import given, strategies as st, assume
from mpmath import mp, mpf

from hydradx.model import run
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm import stableswap_amm as stableswap
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_amm import price
from hydradx.model.amm.trade_strategies import steady_swaps, constant_swaps
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config
from hydradx.tests.test_stableswap import stableswap_config, stable_swap_equation, StableSwapPoolState

mp.dps = 50

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': mpf(draw(asset_quantity_strategy)),
            'LRNA': mpf(draw(asset_quantity_strategy))
        },
        'USD': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA_price': usd_price_lrna
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
def assets_reasonable_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_bounded_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA_price': usd_price_lrna
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
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
    asset_dict: dict = asset_dict or draw(assets_config(token_count))

    sub_pool_instances: dict['str', StableSwapPoolState] = {}
    if sub_pools:
        for i, (name, pool) in enumerate(sub_pools.items()):
            base_token = list(asset_dict.keys())[i + 1]
            sub_pool_instance = draw(stableswap_config(
                asset_dict=pool['asset_dict'] if 'asset_dict' in pool else None,
                token_count=pool['token_count'] if 'token_count' in pool else None,
                amplification=pool['amplification'] if 'amplification' in pool else None,
                trade_fee=pool['trade_fee'] if 'trade_fee' in pool else None,
                base_token=base_token
            ))
            asset_dict.update({tkn: {
                'liquidity': sub_pool_instance.liquidity[tkn],
                'LRNA': (
                    asset_dict[base_token]['LRNA'] * sub_pool_instance.liquidity[tkn]
                    / asset_dict[base_token]['liquidity']
                    if 'LRNA' in asset_dict[base_token] else
                    asset_dict[base_token]['LRNA_price'] * sub_pool_instance.liquidity[tkn]
                )
            } for tkn in sub_pool_instance.asset_list})
            sub_pool_instances[name] = sub_pool_instance

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
    )

    for name, pool in sub_pool_instances.items():
        oamm.execute_create_sub_pool(
            state=test_state,
            tkns_migrate=pool.asset_list,
            sub_pool_id=name,
            amplification=pool.amplification,
            trade_fee=pool.trade_fee
        )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy)
    test_state.update()
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
        initial_state.weight_cap[i] = min(initial_state.lrna[i] / initial_state.lrna_total * 1.1, 1)

    # calculate what should be the maximum allowable liquidity provision
    max_amount = ((old_state.weight_cap[i] / (1 - old_state.weight_cap[i])
                   * old_state.lrna_total - old_state.lrna[i] / (1 - old_state.weight_cap[i]))
                  / oamm.lrna_price(old_state, i))

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
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.add_liquidity(initial_state, initial_agent, 1000, i)
    p_init = oamm.lrna_price(old_state, i)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if oamm.price(old_state, j) != pytest.approx(oamm.price(new_state, j)):
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
    delta_ra = 1000
    delta_qa = -1000
    i = old_state.asset_list[2]

    # Test with trader buying asset i
    feeless_state = initial_state.copy()
    feeless_state.lrna_fee = 0
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_lrna_fee[asset] = 0
        feeless_state.last_fee[asset] = 0
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, delta_ra, 0, i)
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise

    # Test with trader selling LRNA
    new_state, new_agent = oamm.swap_lrna(old_state, old_agent, 0, delta_qa, i)
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, 0, delta_qa, i)
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise
    for j in old_state.asset_list:
        if min(new_state.liquidity[j] - feeless_swap_state.liquidity[j], 0) != pytest.approx(0):
            raise
    if min(oamm.asset_invariant(new_state, i) / oamm.asset_invariant(old_state, i), 1) != pytest.approx(1):
        raise

    delta_qi = new_state.lrna[i] - old_state.lrna[i]
    qi_arb = old_state.lrna[i] + delta_qi * old_state.lrna[i] / old_state.lrna_total
    ri_arb = old_state.liquidity[i] * old_state.lrna_total / new_state.lrna_total

    if ((old_state.lrna[i] + old_state.lrna_imbalance * (old_state.lrna[i] / old_state.lrna_total)) * ri_arb
    ) != pytest.approx(
        (qi_arb + new_state.lrna_imbalance * (qi_arb / new_state.lrna_total)) * old_state.liquidity[i]
    ):
        raise AssertionError(f'LRNA imbalance is wrong.')

    if (new_state.liquidity[i] + new_agent.holdings[i] != pytest.approx(old_state.liquidity[i] + old_agent.holdings[i])
            or new_state.lrna[i] + new_agent.holdings['LRNA']
            != pytest.approx(old_state.lrna[i] + old_agent.holdings['LRNA'])):
        raise AssertionError('System-wide asset total is wrong.')

    # try swapping into LRNA and back to see if that's equivalent
    reverse_state, reverse_agent = oamm.swap_lrna(
        old_state=feeless_swap_state,
        old_agent=feeless_swap_agent,
        delta_qa=-delta_qa,
        tkn=i
    )

    # We do not currently expect imbalance to be symmetric
    # if reverse_state.lrna_imbalance != pytest.approx(old_state.lrna_imbalance):
    #     raise AssertionError('LRNA imbalance is wrong.')

    if reverse_agent.holdings[i] != pytest.approx(old_agent.holdings[i]):
        print(reverse_agent.holdings[i])
        print(old_agent.holdings[i])
        raise AssertionError('Agent holdings are wrong.')


@given(omnipool_reasonable_config(token_count=3, lrna_fee=0.0005, asset_fee=0.0025, imbalance=-1000))
def test_lrna_buy_nonzero_fee_nonzero_imbalance(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list + ['LRNA']}
    )
    delta_qa = 10
    i = old_state.asset_list[2]

    # Test with trader selling asset i
    new_state, new_agent = oamm.swap_lrna(old_state, old_agent, 0, delta_qa, i, modify_imbalance=False)

    expected_delta_qi = -delta_qa / (1 - 0.0005)
    expected_fee = -(delta_qa + expected_delta_qi)

    if old_state.lrna_total - new_state.lrna_total != pytest.approx(
            new_agent.holdings['LRNA'] - old_agent.holdings['LRNA'] + expected_fee):
        raise AssertionError('LRNA total is wrong.')


@given(omnipool_reasonable_config(token_count=3, lrna_fee=0.0005, asset_fee=0.0025, imbalance=0))
def test_lrna_buy_nonzero_fee_zero_imbalance(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list + ['LRNA']}
    )
    delta_qa = 10
    i = old_state.asset_list[2]

    # Test with trader selling asset i
    new_state, new_agent = oamm.swap_lrna(old_state, old_agent, 0, delta_qa, i, modify_imbalance=False)

    expected_delta_qi = -delta_qa / (1 - 0.0005)
    expected_fee = -(delta_qa + expected_delta_qi)

    if expected_fee != pytest.approx(new_state.lrna['HDX'] - old_state.lrna['HDX']):
        raise AssertionError('Fee to HDX pool is wrong.')

    if old_state.lrna[i] - new_state.lrna[i] != pytest.approx(
            new_agent.holdings['LRNA'] - old_agent.holdings['LRNA'] + expected_fee):
        raise AssertionError('Delta Qi is wrong.')

    if old_state.lrna_total - new_state.lrna_total != pytest.approx(
            new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']):
        raise AssertionError('Some LRNA is being incorrectly burned or minted.')


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
    for asset in feeless_state.asset_list:
        feeless_state.last_lrna_fee[asset] = 0
        feeless_state.last_fee[asset] = 0

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
    for asset in buy_state.asset_list:
        buy_state.last_lrna_fee[asset] = 0
        buy_state.last_fee[asset] = 0
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
    if not (stable_swap_equation(
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

    new_pool_buy: StableSwapPoolState = new_state.sub_pools['stableswap1']
    new_pool_sell: StableSwapPoolState = new_state.sub_pools['stableswap2']
    if not new_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy] == buy_quantity:
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

    new_pool_buy: StableSwapPoolState = new_state.sub_pools['stableswap1']
    new_pool_sell: StableSwapPoolState = new_state.sub_pools['stableswap2']
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


@given(omnipool_config(token_count=4), st.floats(min_value=0.1, max_value=1), st.floats(min_value=0.1, max_value=1))
def test_slip_fees(initial_state: oamm.OmnipoolState, lrna_slip_rate: float, asset_slip_rate: float):
    initial_state.lrna_fee = oamm.slip_fee(lrna_slip_rate, minimum_fee=0.0001)
    initial_state.asset_fee = oamm.slip_fee(asset_slip_rate, minimum_fee=0.0001)
    initial_agent = Agent(holdings={tkn: 1000000 for tkn in initial_state.asset_list})
    tkn_buy = initial_state.asset_list[2]
    tkn_sell = initial_state.asset_list[3]
    sell_quantity = 1
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

    buy_quantity = 1
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
    new_sub_pool: StableSwapPoolState = new_state.sub_pools[s]
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
    migrated_sub_pool: StableSwapPoolState = migrated_state.sub_pools[s]
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


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_migration_scenarios(initial_state: oamm.OmnipoolState):
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
    s2_sub_pool: StableSwapPoolState = s2_state.sub_pools['stableswap']
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
        stableswap.execute_swap(
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
    s2_sub_pool: StableSwapPoolState = s2_state.sub_pools['stableswap']
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
        stableswap.execute_swap(
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
def test_add_stableswap_liquidity(initial_state: oamm.OmnipoolState):
    stable_pool: StableSwapPoolState = initial_state.sub_pools['stableswap']
    agent = Agent(
        holdings={stable_pool.asset_list[0]: 1000}
    )
    new_state, new_agent = oamm.add_liquidity(
        initial_state, agent,
        quantity=1000, tkn_add=stable_pool.asset_list[0]
    )
    if new_state.fail:
        # this could be due to liquidity overload or whatever
        return
    if (initial_state.unique_id, stable_pool.unique_id) not in new_agent.holdings:
        raise ValueError("Agent did not receive shares.")
    if not (new_agent.holdings[(initial_state.unique_id, stable_pool.unique_id)] > 0):
        raise AssertionError("Sanity check failed.")


def test_arbitrage():
    import sys
    sys.path.append('../..')

    from hydradx.model import run
    from hydradx.model.amm.omnipool_amm import OmnipoolState
    from hydradx.model.amm.agents import Agent
    from hydradx.model.amm.trade_strategies import omnipool_arbitrage
    from hydradx.model.amm.global_state import GlobalState, fluctuate_prices

    assets = {
        'HDX': {'usd price': 0.05, 'weight': 0.10},
        'USD': {'usd price': 1, 'weight': 0.20},
        'AUSD': {'usd price': 1, 'weight': 0.10},
        'ETH': {'usd price': 2500, 'weight': 0.40},
        'DOT': {'usd price': 5.37, 'weight': 0.20}
    }

    lrna_price_usd = 0.07
    initial_omnipool_tvl = 10000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    initial_state = GlobalState(
        pools={
            'Omnipool': OmnipoolState(
                tokens={
                    tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
                },
                lrna_fee=0,
                asset_fee=0,
                preferred_stablecoin='USD'
            )
        },
        agents={
            # 'Attacker': Agent(
            #     holdings={'USD': 0, 'AUSD': 1000000000},
            #     trade_strategy=toxic_asset_attack(
            #         pool_id='omnipool',
            #         asset_name='AUSD',
            #         trade_size=10000
            #     )
            # ),
            'Arbitrageur': Agent(
                holdings={tkn: float('inf') for tkn in list(assets.keys()) + ['LRNA']},
                trade_strategy=omnipool_arbitrage('Omnipool')
            )
        },
        evolve_function=fluctuate_prices(volatility={tkn: 0.1 for tkn in assets}),
        external_market={tkn: assets[tkn]['usd price'] for tkn in assets}
    )
    # print(initial_state)
    time_steps = 1000  # len(price_list) - 1
    events = run.run(initial_state, time_steps=time_steps)


def test_trade_limit():
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000},
            'R1': {'liquidity': 1000000, 'LRNA': 1000000},
        },
        trade_limit_per_block=0.25
    )
    agent = Agent(
        holdings={'HDX': 1000000, 'USD': 1000000, 'R1': 1000000, 'LRNA': 1000000}
    )
    new_state = initial_state.copy()
    trades_allowed = 0
    while not new_state.fail:
        new_state, new_agent = oamm.swap(
            new_state, agent, 'USD', 'R1', sell_quantity=100000
        )
        if new_state.fail:
            break
        trades_allowed += 1

    assert trades_allowed == 2

    new_state = initial_state.copy()
    for i in range(26):
        if new_state.fail:
            raise AssertionError('Not enough trades allowed')
        new_state, new_agent = oamm.swap(
            new_state, agent, 'R1', 'USD', buy_quantity=1000
        )
        new_state, new_agent = oamm.swap(
            new_state, agent, 'USD', 'R1', sell_quantity=11000
        )

    if not new_state.fail:
        raise AssertionError('Too many trades allowed')


@given(
    st.floats(min_value=1e-5, max_value=1e5),
)
def test_dynamic_fees(hdx_price: float):
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 100000 / hdx_price, 'LRNA': 100000},
            'USD': {'liquidity': 100000, 'LRNA': 100000},
            'R1': {'liquidity': 100000, 'LRNA': 100000},
        },
        oracles={
            'mid': 100
        },
        asset_fee=oamm.dynamicadd_asset_fee(
            minimum=0.0025,
            amplification=10,
            raise_oracle_name='mid',
            decay=0.0005,
            fee_max=0.40
        ),
        lrna_fee=oamm.dynamicadd_lrna_fee(
            minimum=0.0005,
            amplification=10,
            raise_oracle_name='mid',
            decay=0.0001,
            fee_max=0.10
        ), last_asset_fee={'R1': 0.1}, last_lrna_fee={'R1': 0.1}
    )
    initial_hdx_fee = initial_state.asset_fee['HDX'].compute('HDX', 10000)
    initial_usd_fee = initial_state.asset_fee['USD'].compute('USD', 10000)
    initial_usd_lrna_fee = initial_state.lrna_fee['USD'].compute('USD', 10000)
    initial_hdx_lrna_fee = initial_state.lrna_fee['HDX'].compute('HDX', 10000)
    initial_R1_fee = initial_state.asset_fee['R1'].compute('R1', 10000)
    initial_R1_lrna_fee = initial_state.lrna_fee['R1'].compute('R1', 10000)
    test_agent = Agent(
        holdings={tkn: initial_state.liquidity[tkn] / 100 for tkn in initial_state.asset_list}
    )
    test_state = initial_state.copy()
    oamm.execute_swap(
        state=test_state,
        agent=test_agent,
        tkn_sell='USD',
        tkn_buy='HDX',
        sell_quantity=test_agent.holdings['USD']
    )
    test_state.update()
    if test_state.last_fee['R1'] >= initial_R1_fee:
        raise AssertionError('R1 fee should be decreasing due to decay.')
    if test_state.last_lrna_fee['R1'] >= initial_R1_lrna_fee:
        raise AssertionError('R1 LRNA fee should be decreasing due to decay.')
    intermediate_hdx_fee = test_state.asset_fee['HDX'].compute('HDX', 10000)
    intermediate_usd_fee = test_state.asset_fee['USD'].compute('USD', 10000)
    intermediate_usd_lrna_fee = test_state.lrna_fee['USD'].compute('USD', 10000)
    intermediate_hdx_lrna_fee = test_state.lrna_fee['HDX'].compute('HDX', 10000)
    if not intermediate_hdx_fee > initial_hdx_fee:
        raise AssertionError('Fee should increase when price increases.')
    if not intermediate_usd_lrna_fee > initial_usd_lrna_fee:
        raise AssertionError('LRNA fee should increase when price decreases.')
    if not intermediate_usd_fee == initial_usd_fee:
        raise AssertionError('Asset fee should not change.')
    if not intermediate_hdx_lrna_fee == initial_hdx_lrna_fee:
        raise AssertionError('LRNA fee should not change.')

    oamm.execute_swap(
        state=test_state,
        agent=test_agent,
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=test_agent.holdings['HDX']
    )
    test_state.update()
    final_hdx_fee = test_state.asset_fee['HDX'].compute('HDX', 10000)
    final_usd_fee = test_state.asset_fee['USD'].compute('USD', 10000)
    final_usd_lrna_fee = test_state.lrna_fee['USD'].compute('USD', 10000)
    final_hdx_lrna_fee = test_state.lrna_fee['HDX'].compute('HDX', 10000)
    if not final_usd_fee > intermediate_usd_fee:
        raise AssertionError('Fee should increase when price increases.')
    if not final_hdx_lrna_fee > intermediate_hdx_lrna_fee:
        raise AssertionError('LRNA fee should increase when price decreases.')
    if not final_hdx_fee < intermediate_hdx_fee:
        raise AssertionError('Asset fee should decrease with time.')
    if not final_usd_lrna_fee < intermediate_usd_lrna_fee:
        raise AssertionError('LRNA fee should decrease with time.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_oracle_one_empty_block(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                oracle_volume_in: list[float], oracle_volume_out: list[float],
                                oracle_prices: list[float], n):
    alpha = 2 / (n + 1)

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    state = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'oracle': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'oracle': copy.deepcopy(init_oracle)
        }
    )

    initial_state = GlobalState(
        pools={'omnipool': state},
        agents={}
    )

    events = run.run(initial_state=initial_state, time_steps=1, silent=True)
    omnipool_oracle = events[0].pools['omnipool'].oracles['oracle']
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle.liquidity[tkn] != expected_liquidity:
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_in[tkn] != expected_vol_in:
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_out[tkn] != expected_vol_out:
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle.price[tkn] != expected_price:
            raise AssertionError('Price is not correct.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.lists(st.floats(min_value=10, max_value=1000), min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_oracle_one_block_with_swaps(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                     oracle_volume_in: list[float], oracle_volume_out: list[float],
                                     oracle_prices: list[float], trade_sizes: list[float], n):
    alpha = 2 / (n + 1)

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    state = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'oracle': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'oracle': copy.deepcopy(init_oracle)
        }
    )

    trader1_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}
    trader2_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}

    initial_state = GlobalState(
        pools={'omnipool': state},
        agents={
            'Trader1': Agent(
                holdings=trader1_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_sizes[0],
                    sell_asset='LRNA',
                    buy_asset='DOT'
                )
            ),
            'Trader2': Agent(
                holdings=trader2_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_sizes[1],
                    sell_asset='DOT',
                    buy_asset='LRNA'
                )
            ),
        }
    )

    events = run.run(initial_state=initial_state, time_steps=2, silent=True)
    omnipool_oracle_0 = events[0].pools['omnipool'].oracles['oracle']

    vol_in = {
        'HDX': 0,
        'USD': 0,
        'DOT': trader2_holdings['DOT'] - events[0].agents['Trader2'].holdings['DOT'],
    }

    vol_out = {
        'HDX': 0,
        'USD': 0,
        'DOT': events[0].agents['Trader1'].holdings['DOT'] - trader1_holdings['DOT'],
    }

    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle_0.liquidity[tkn] != expected_liquidity:
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_in[tkn] != expected_vol_in:
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_out[tkn] != expected_vol_out:
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle_0.price[tkn] != expected_price:
            raise AssertionError('Price is not correct.')

    omnipool_oracle_1 = events[1].pools['omnipool'].oracles['oracle']
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = omnipool_oracle_0.liquidity[tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle_1.liquidity[tkn] != pytest.approx(expected_liquidity, 1e-10):
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = omnipool_oracle_0.volume_in[tkn] * (1 - alpha) + alpha * vol_in[tkn]
        if omnipool_oracle_1.volume_in[tkn] != pytest.approx(expected_vol_in, 1e-10):
            raise AssertionError('Volume is not correct.')

        expected_vol_out = omnipool_oracle_0.volume_out[tkn] * (1 - alpha) + alpha * vol_out[tkn]
        if omnipool_oracle_1.volume_out[tkn] != pytest.approx(expected_vol_out, 1e-9):
            raise AssertionError('Volume is not correct.')

        price_1 = price(events[0].pools['omnipool'], tkn)
        expected_price = omnipool_oracle_0.price[tkn] * (1 - alpha) + alpha * price_1
        if omnipool_oracle_1.price[tkn] != pytest.approx(expected_price, 1e-10):
            raise AssertionError('Price is not correct.')