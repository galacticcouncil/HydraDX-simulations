import copy
import pytest
import random
from hypothesis import given, strategies as st, assume
from mpmath import mpf, mp

from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)

mp.dps = 50


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    lrna_price_usd = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA_price': 1 / lrna_price_usd
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
        tvl_cap_usd=0
) -> oamm.OmnipoolState:
    asset_dict = asset_dict or draw(assets_config(token_count))
    return oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee
    )


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
        assert oamm.price_i(market_state, i) > 0


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity(initial_state: oamm.OmnipoolState):
    initial_state.tvl_cap = initial_state.tvl_total * 2
    for i in initial_state.asset_list:
        initial_state.weight_cap[i] = mpf(min(initial_state.lrna[i] / initial_state.lrna_total * 1.001, 1))

    old_state = initial_state
    lp_id = 'LP'
    i = old_state.asset_list[-1]

    old_agent = Agent(
        holdings={i: 1000},
        shares={i: 0},
        share_prices={i: old_state.price(i)}
    )
    delta_R = 1000

    new_state, new_agents = oamm.add_liquidity(old_state, old_agent, delta_R, i)
    for j in initial_state.asset_list:
        assert oamm.price_i(old_state, j) == pytest.approx(oamm.price_i(new_state, j))
    if old_state.liquidity[i] / old_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError(f'Price change in {i}'
                             f'({old_state.liquidity[i] / old_state.shares[i]}) -->'
                             f'({pytest.approx(new_state.liquidity[i] / new_state.shares[i])})'
                             )

    if old_state.lrna_imbalance / old_state.lrna_total != \
            pytest.approx(new_state.lrna_imbalance / new_state.lrna_total):
        raise AssertionError(f'LRNA imbalance did not remain constant.')

    # check enforcement of overall TVL cap
    # calculate what should be the maximum allowable liquidity provision
    max_amount = ((old_state.weight_cap[i] / (1 - old_state.weight_cap[i])
                   * old_state.lrna_total - old_state.lrna[i] / (1 - old_state.weight_cap[i]))
                  / old_state.price(i))

    if max_amount < 0:
        raise AssertionError('This calculation makes no sense.')

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
    p_init = 1
    old_state = initial_state
    old_agent = Agent(
        holdings={token: mpf(1000) for token in initial_state.asset_list + ['LRNA']},
        share_prices={token: mpf(p_init) for token in initial_state.asset_list},
        shares={token: mpf(1000) for token in initial_state.asset_list}
    )
    # add LP shares to the pool
    for tkn in old_agent.shares:
        shares = old_agent.shares[tkn]
        old_state.liquidity[tkn] += shares * old_state.liquidity[tkn] / old_state.shares[tkn]
        old_state.lrna[tkn] += shares * old_state.lrna[tkn] / old_state.shares[tkn]
        old_state.shares[tkn] += shares

    delta_S = -old_agent.shares[i]

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

    piq = oamm.price_i(old_state, i)
    val_withdrawn = piq * delta_r + delta_q
    if (-2 * piq / (piq + p_init) * delta_S / old_state.shares[i] * piq
            * old_state.liquidity[i] != pytest.approx(val_withdrawn)) \
            and not new_state.fail:
        raise AssertionError('something is wrong')


@given(omnipool_config(token_count=3), fee_strategy)
def test_swap_lrna(initial_state: oamm.OmnipoolState, fee):
    old_state = initial_state
    trader_id = 'trader'
    LP_id = 'lp'
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
    new_state, new_agents = oamm.swap_lrna(old_state, old_agent, 0, delta_Qa, i)
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, 0, delta_Qa, i)
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise
    for j in old_state.asset_list:
        if min(new_state.liquidity[j] - feeless_swap_state.liquidity[j], 0) != pytest.approx(0):
            raise
    if min(oamm.asset_invariant(new_state, i) / oamm.asset_invariant(old_state, i), 1) != pytest.approx(1):
        raise

    if old_state.lrna[i] / old_state.liquidity[i] != pytest.approx(
                (feeless_swap_state.lrna[i] + feeless_swap_state.lrna_imbalance) / feeless_swap_state.liquidity[i]
            ):
        raise AssertionError(
            f'Impermanent loss calculation incorrect. '
            f'{old_state.lrna[i] / old_state.liquidity[i]} != '
            f'{(feeless_swap_state.lrna[i] + feeless_swap_state.lrna_imbalance) / feeless_swap_state.liquidity[i]}'
        )

    # try swapping into LRNA and back to see if that's equivalent


@given(omnipool_config(token_count=3), st.integers(min_value=1, max_value=2))
def test_swap_assets(initial_state: oamm.OmnipoolState, i):
    i_buy = initial_state.asset_list[i]
    old_state = initial_state

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']},
        shares={token: 10000 for token in initial_state.asset_list}
    )
    delta_R = 1000
    sellable_tokens = len(old_state.asset_list) - 1
    i_sell = old_state.asset_list[i % sellable_tokens + 1]

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

    delta_out_new = new_agent.holdings[i_buy] - old_agent.holdings[i_buy]

    # Test with trader buying asset i, no LRNA fee... price should match feeless
    buy_state, buy_agent = oamm.swap(
        old_state, old_agent, i_buy, i_sell, buy_quantity=-delta_out_new
    )

    for j in old_state.asset_list:
        assert buy_state.liquidity[j] == pytest.approx(new_state.liquidity[j])
        assert buy_state.lrna[j] == pytest.approx(new_state.lrna[j])
        assert old_state.liquidity[j] + old_agent.holdings[j] == \
               pytest.approx(buy_state.liquidity[j] + buy_agent.holdings[j])
        assert buy_agent.holdings[j] == pytest.approx(new_agent.holdings[j])
        assert buy_agent.holdings['LRNA'] == pytest.approx(new_agent.holdings['LRNA'])


if __name__ == '__main__':
    test_swap_lrna_delta_Ri_respects_invariant()
    test_swap_lrna_delta_Qi_respects_invariant()
    test_swap_lrna()
    test_weights()
    test_prices()
    test_add_liquidity()
    test_remove_liquidity()
    test_swap_assets()
