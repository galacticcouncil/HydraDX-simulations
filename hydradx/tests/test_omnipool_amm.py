import copy
import math

import pytest
from hypothesis import given, strategies as st, assume

from hydradx.model.amm import omnipool_amm as oamm

# Token counts
tkn_ct_strat = st.floats(min_value=10000, max_value=10000000, allow_nan=False, allow_infinity=False)


def get_tkn_ct_strat(n):
    args = [tkn_ct_strat] * n
    return st.tuples(*args)


# States with R, Q lists
QiRi_strat = get_tkn_ct_strat(2)


def get_state_from_strat(x, key_list):
    """x is a tuple, key_list is a list of keys each of which follow strategy x"""
    d = {}
    if isinstance(key_list, list):
        assert len(x[0]) == len(key_list)
        for j in range(len(key_list)):
            k = key_list[j]
            d[k] = [None] * len(x)
            for i in range(len(x)):
                d[k][i] = x[i][j]
    else:
        assert not (isinstance(x, tuple))
        k = key_list
        d[k] = copy.deepcopy(x)
    return d


QR_strat = st.lists(QiRi_strat, min_size=2, max_size=5).map(lambda x: get_state_from_strat(x, ['Q', 'R']))

# Delta_TKN variables
delta_tkn_strat = st.floats(allow_nan=False, allow_infinity=False)

# TKN quantity list
Q_strat = st.lists(tkn_ct_strat, min_size=1, max_size=5).map(lambda x: get_state_from_strat(x, 'Q'))

# Indexes
i_strat = st.integers(min_value=0)

RQBSHD_strat = get_tkn_ct_strat(6).map(lambda x: get_state_from_strat(x, ['R', 'Q', 'B', 'S', 'H', 'D']))


# Tests over input space of Q, R, delta_TKN, i

def test_swap_hdx_delta_Qi_respects_invariant(d, delta_Ri, i):
    assume(i < len(d['R']))
    assume(d['R'][i] > delta_Ri > -d['R'][i])
    d2 = copy.deepcopy(d)
    delta_Qi = oamm.swap_hdx_delta_Qi(d, delta_Ri, i)
    d2['R'][i] += delta_Ri
    d2['Q'][i] += delta_Qi

    # Test basics
    for j in range(len(d2['R'])):
        assert d2['R'][j] > 0
        assert d2['Q'][j] > 0
    assert not (delta_Ri > 0 and delta_Qi > 0)
    assert not (delta_Ri < 0 and delta_Qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


def test_swap_hdx_delta_Ri_respects_invariant(d, delta_Qi, i):
    assume(i < len(d['Q']))
    assume(d['Q'][i] > delta_Qi > -d['Q'][i])
    d2 = copy.deepcopy(d)
    delta_Ri = oamm.swap_hdx_delta_Ri(d, delta_Qi, i)
    d2['Q'][i] += delta_Qi
    d2['R'][i] += delta_Ri

    # Test basics
    for j in range(len(d2['R'])):
        assert d2['R'][j] > 0
        assert d2['Q'][j] > 0
    assert not (delta_Ri > 0 and delta_Qi > 0)
    assert not (delta_Ri < 0 and delta_Qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


# Combining these two tests because the valid input space is the same
@given(QR_strat, delta_tkn_strat, i_strat)
def test_swap_hdx_delta_TKN_respects_invariant(d, delta_TKN, i):
    test_swap_hdx_delta_Qi_respects_invariant(d, delta_TKN, i)
    test_swap_hdx_delta_Ri_respects_invariant(d, delta_TKN, i)


# Tests over input space of a list of token quantities

@given(Q_strat)
def test_weights(d):
    for i in range(len(d['Q'])):
        assert oamm.weight_i(d, i) >= 0
    assert sum([oamm.weight_i(d, i) for i in range(len(d['Q']))]) == pytest.approx(1.0)


def test_prices(d):
    for i in range(len(d['Q'])):
        assert oamm.price_i(d, i) > 0


def test_initialize_pool_state(d):
    pass  # TODO


@given(QR_strat)
def test_QR_strat(d):
    test_prices(d)
    test_initialize_pool_state(d)


@given(QR_strat)
def test_add_risk_liquidity(old_state):
    n = len(old_state['R'])
    old_state['S'] = [1500000] * 2
    old_state['P'] = [oamm.price_i(old_state, j) for j in range(n)]

    LP_id = 'LP'
    old_agents = {
        LP_id: {
            'r': [1000, 0],
            's': [0, 0],
            'p': [None, None]
        }
    }
    delta_R = 1000
    i = 0

    new_state, new_agents = oamm.add_risk_liquidity(old_state, old_agents, LP_id, delta_R, i)
    for j in range(len(['R'])):
        assert oamm.price_i(old_state, j) == pytest.approx(oamm.price_i(new_state, j))
    assert old_state['R'][i] / old_state['S'][i] == pytest.approx(new_state['R'][i] / new_state['S'][i])


@given(QR_strat)
def test_remove_risk_liquidity(old_state):
    n = len(old_state['R'])
    old_state['S'] = [1500000] * n
    old_state['P'] = [oamm.price_i(old_state, j) for j in range(n)]
    B_init = 0
    old_state['B'] = [B_init] * n

    LP_id = 'LP'
    p_init = 1
    old_agents = {
        LP_id: {
            'r': [0] * n,
            's': [1000] * n,
            'p': [p_init] * n,
            'q': 0
        }
    }
    delta_S = -1000
    i = 0

    new_state, new_agents = oamm.remove_risk_liquidity(old_state, old_agents, LP_id, delta_S, i)
    for j in range(len(['R'])):
        assert oamm.price_i(old_state, j) == pytest.approx(oamm.price_i(new_state, j))
    assert old_state['R'][i] / old_state['S'][i] == pytest.approx(new_state['R'][i] / new_state['S'][i])
    delta_r = new_agents[LP_id]['r'][i] - old_agents[LP_id]['r'][i]
    delta_q = new_agents[LP_id]['q'] - old_agents[LP_id]['q']
    assert delta_q >= 0 or delta_q == pytest.approx(0)
    assert delta_r >= 0 or delta_r == pytest.approx(0)

    piq = oamm.price_i(old_state, i)
    val_withdrawn = piq * delta_r + delta_q
    assert -2 * piq / (piq + p_init) * delta_S / old_state['S'][i] * piq * old_state['R'][
        i] == pytest.approx(val_withdrawn)


@given(QR_strat)
def test_swap_hdx(old_state):
    n = len(old_state['R'])
    old_state['S'] = [1000] * n
    old_state['A'] = [0] * n
    old_state['B'] = [0] * n
    old_state['D'] = 0
    trader_id = 'trader'
    old_agents = {
        trader_id: {
            'r': [1000] * n,
            'q': 1000,
            's': [0] * n
        }
    }
    delta_R = 1000
    delta_Q = 1000
    i = 0

    # Test with trader selling asset i
    new_state, new_agents = oamm.swap_lhdx(old_state, old_agents, trader_id, delta_R, 0, i)
    assert oamm.asset_invariant(old_state, i) == pytest.approx(oamm.asset_invariant(new_state, i))

    # Test with trader selling HDX
    new_state, new_agents = oamm.swap_lhdx(old_state, old_agents, trader_id, 0, delta_Q, i)
    assert oamm.asset_invariant(old_state, i) == pytest.approx(oamm.asset_invariant(new_state, i))


fee_strat = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)
@given(QR_strat, fee_strat)
def test_swap_hdx_fee(old_state, fee):
    n = len(old_state['R'])
    old_state['S'] = [1000] * n
    old_state['A'] = [0] * n
    old_state['B'] = [100] * n
    old_state['D'] = 0
    trader_id = 'trader'
    LP_id = 'lp'
    old_agents = {
        trader_id: {
            'r': [1000] * n,
            'q': 1000,
            's': [0] * n
        },
        LP_id: {
            'r': [0] * n,
            'q': 0,
            's': [900] * n
        }
    }
    delta_R = 1000
    delta_Q = 1000
    i = 0

    # Test with trader selling asset i
    new_state, new_agents = oamm.swap_lhdx_fee(old_state, old_agents, trader_id, delta_R, 0, i, fee, fee)
    assert oamm.asset_invariant(old_state, i) == pytest.approx(oamm.asset_invariant(new_state, i))
    assert sum(old_state['Q']) + old_agents[trader_id]['q'] == pytest.approx(sum(new_state['Q']) + new_state['D'] + new_agents[trader_id]['q'])

    # Test with trader selling HDX
    new_state, new_agents = oamm.swap_lhdx_fee(old_state, old_agents, trader_id, 0, delta_Q, i, fee, fee)
    feeless_state, feeless_agents = oamm.swap_lhdx_fee(old_state, old_agents, trader_id, 0, delta_Q, i, 0, 0)
    for j in range(len(old_state['R'])):
        assert oamm.price_i(feeless_state, j) == pytest.approx(oamm.price_i(new_state, j))
        assert min(new_state['R'][j] - feeless_state['R'][j], 0) == pytest.approx(0)
    assert min(oamm.asset_invariant(new_state, i) / oamm.asset_invariant(old_state, i), 1) == pytest.approx(1)

fee_strat = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)
@given(QR_strat, fee_strat, fee_strat)
def test_swap_assets(old_state, fee_LHDX, fee_assets):

    n = len(old_state['R'])
    old_state['S'] = [1000] * n
    old_state['A'] = [0] * n
    old_state['B'] = [100] * n
    old_state['D'] = 0
    trader_id = 'trader'
    LP_id = 'lp'

    old_agents = {
        trader_id: {
            'r': [10000] * n,
            'q': 10000,
            's': [0] * n
        },
        LP_id: {
            'r': [0] * n,
            'q': 0,
            's': [900] * n
        }
    }
    delta_R = 1000
    i_buy = 0
    i_sell = 1


    # Test with trader selling asset i, no HDX fee... price should match feeless
    new_state, new_agents = oamm.swap_assets(old_state, old_agents, trader_id, 'sell', delta_R, i_buy, i_sell, fee_assets, fee_LHDX)
    asset_only_state, asset_only_agents = oamm.swap_assets(old_state, old_agents, trader_id, 'sell', delta_R, i_buy, i_sell, fee_assets, 0)
    feeless_state, feeless_agents = oamm.swap_assets(old_state, old_agents, trader_id, 'sell', delta_R, i_buy, i_sell, 0, 0)
    for j in range(len(old_state['R'])):
        # price tracks feeless price
        assert oamm.price_i(feeless_state, j) == pytest.approx(oamm.price_i(asset_only_state, j))
        # assets in pools only go up compared to asset_only_state
        assert min(asset_only_state['R'][j] - feeless_state['R'][j], 0) == pytest.approx(0)
        # asset in pool goes up from asset_only_state -> new_state (i.e. introduction of LHDX fee)
        assert min(new_state['R'][j] - asset_only_state['R'][j], 0) == pytest.approx(0)
        # invariant does not decrease
        assert min(oamm.asset_invariant(new_state, j) / oamm.asset_invariant(old_state, j), 1) == pytest.approx(1)
        assert old_state['R'][j] + old_agents[trader_id]['r'][j] == pytest.approx(new_state['R'][j] + new_agents[trader_id]['r'][j])

    delta_out_new = new_agents[trader_id]['r'][i_buy] - old_agents[trader_id]['r'][i_buy]

    # Test with trader buying asset i, no HDX fee... price should match feeless
    buy_state, buy_agents = oamm.swap_assets(old_state, old_agents, trader_id, 'buy', -delta_out_new, i_buy, i_sell,  fee_assets, fee_LHDX)

    for j in range(len(old_state['R'])):
        assert buy_state['R'][j] == pytest.approx(new_state['R'][j])
        assert buy_state['Q'][j] == pytest.approx(new_state['Q'][j])
        assert old_state['R'][j] + old_agents[trader_id]['r'][j] == pytest.approx(buy_state['R'][j] + buy_agents[trader_id]['r'][j])
        assert buy_agents[trader_id]['r'][j] == pytest.approx(new_agents[trader_id]['r'][j])
        assert buy_agents[trader_id]['q'] == pytest.approx(new_agents[trader_id]['q'])


price_strat = st.floats(min_value=1e-5, max_value=1e5, allow_nan=False, allow_infinity=False)


@given(QR_strat, price_strat)
def test_add_asset(old_state, price):
    old_state['S'] = [1000000, 1000000]
    old_state['B'] = [0, 0]
    old_state['A'] = [0, 0]
    old_state['D'] = 0

    n = len(old_state['R'])
    init_R = 100000

    new_state = oamm.add_asset(old_state, init_R, price)
    assert oamm.price_i(new_state, n) == pytest.approx(price)


# Want to make sure this does not change pij, only changes piq proportionally
# Also should make sure things stay reasonably bounded
# Requires state with H, T, Q, burn_rate
rate_strat = st.floats(min_value=1e-4, max_value=.99, allow_nan=False, allow_infinity=False)
@given(QR_strat, rate_strat)
def test_adjust_supply(old_state, r):
    old_state['H'] = 20000000000
    old_state['T'] = 15000000000
    blocks_per_day = 5 * 60 * 24  # 12-second blocks
    old_state['burn_rate'] = (1 + r)**(1/blocks_per_day) - 1

    new_state = oamm.adjust_supply(old_state)

    assert new_state['H'] > 0
    assert new_state['H'] >= new_state['T']
    for i in range(len(old_state['Q'])):
        assert new_state['Q'][i] > 0
        for j in range(i):
            piq_old = oamm.price_i(old_state, i)
            piq_new = oamm.price_i(new_state, i)
            pjq_old = oamm.price_i(old_state, j)
            pjq_new = oamm.price_i(new_state, j)
            assert piq_old/pjq_old == pytest.approx(piq_new/pjq_new)


if __name__ == '__main__':
    test_swap_hdx_delta_TKN_respects_invariant()
    test_swap_hdx()
    test_swap_hdx_fee()
    test_weights()
    test_QR_strat()
    test_add_risk_liquidity()
    test_remove_risk_liquidity()
    #test_add_asset()
    test_adjust_supply()
    test_swap_assets()
