import copy
import math

import pytest
from hypothesis import given, strategies as st, assume, settings

from hydradx.model import run
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_amm import price, dynamicadd_asset_fee, dynamicadd_lrna_fee
from hydradx.model.amm.trade_strategies import constant_swaps, omnipool_arbitrage
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, omnipool_config, assets_config

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


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


@given(omnipool_config(token_count=3, withdrawal_fee=False))
def test_remove_liquidity_no_fee(initial_state: oamm.OmnipoolState):
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
        raise AssertionError('Ratio of liquidity to shares changed')
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
def test_remove_liquidity_min_fee(initial_state: oamm.OmnipoolState):
    min_fee = 0.0001
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
    if old_state.liquidity[i] / old_state.shares[i] >= new_state.liquidity[i] / new_state.shares[i]:
        raise AssertionError('Ratio of liquidity to shares decreased')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = oamm.lrna_price(old_state, i)
    val_withdrawn = piq * delta_r + delta_q
    if (-2 * piq / (piq + p_init) * delta_S / old_state.shares[i] * piq
            * old_state.liquidity[i] * (1 - min_fee) != pytest.approx(val_withdrawn)
            and not new_state.fail):
        raise AssertionError('something is wrong')

    if old_state.lrna_imbalance / old_state.lrna_total != \
            pytest.approx(new_state.lrna_imbalance / new_state.lrna_total):
        raise AssertionError(f'LRNA imbalance did not remain constant.')


@given(
    st.floats(min_value=0.0002, max_value=0.05),
    assets_config(token_count=3)
)
def test_remove_liquidity_dynamic_fee(price_diff: float, asset_dict: dict):
    i = list(asset_dict.keys())[2]

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        asset_fee=0.0025,
        lrna_fee=0.0005,
        withdrawal_fee=True,
    )

    test_state.oracles['price'].price[i] /= (1 + price_diff)

    min_fee = 0.0001

    initial_agent = Agent(
        holdings={token: 1000 for token in test_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.add_liquidity(test_state, initial_agent, 1000, i)
    p_init = oamm.lrna_price(old_state, i)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if oamm.price(old_state, j) != pytest.approx(oamm.price(new_state, j)):
            raise AssertionError(f'Price change in asset {j}')
    if old_state.liquidity[i] / old_state.shares[i] >= new_state.liquidity[i] / new_state.shares[i]:
        raise AssertionError('Ratio of liquidity to shares decreased')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = oamm.lrna_price(old_state, i)
    val_withdrawn = piq * delta_r + delta_q

    x = -2 * piq / (piq + p_init)
    share_ratio = delta_S / old_state.shares[i]
    feeless_val = x * share_ratio * piq * old_state.liquidity[i]
    theoretical_val = feeless_val * (1 - price_diff)
    if theoretical_val != pytest.approx(val_withdrawn) and not new_state.fail:
        raise AssertionError('something is wrong')

    if old_state.lrna_imbalance / old_state.lrna_total != \
            pytest.approx(new_state.lrna_imbalance / new_state.lrna_total):
        raise AssertionError(f'LRNA imbalance did not remain constant.')


@given(omnipool_config(token_count=3, withdrawal_fee=False),
       st.floats(min_value=0.001, max_value=0.2))
def test_remove_liquidity_no_fee_different_price(initial_state: oamm.OmnipoolState, trade_size_ratio: float):
    i = initial_state.asset_list[2]
    initial_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    init_contrib = 1000
    old_state, old_agent = oamm.add_liquidity(initial_state, initial_agent, init_contrib, i)
    p_init = oamm.lrna_price(old_state, i)

    trader_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    tkn2 = initial_state.asset_list[1]
    trade_state, _ = oamm.swap(old_state, trader_agent, tkn_buy=tkn2, tkn_sell=i,
                               sell_quantity=initial_state.liquidity[i] * trade_size_ratio)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.remove_liquidity(trade_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if oamm.price(trade_state, j) != pytest.approx(oamm.price(new_state, j)):
            raise AssertionError(f'Price change in asset {j}')
    if trade_state.liquidity[i] / trade_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError('Ratio of liquidity to shares changed')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = oamm.lrna_price(trade_state, i)
    val_withdrawn = piq * delta_r + delta_q
    value_percent = 2 * piq / (piq + p_init) * math.sqrt(piq / p_init)
    theoretical_val = value_percent * p_init * init_contrib
    if theoretical_val != pytest.approx(val_withdrawn) and not new_state.fail:
        raise AssertionError('something is wrong')

    if trade_state.lrna_imbalance / trade_state.lrna_total != \
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

    if new_state.liquidity[i] + new_agent.holdings[i] != pytest.approx(old_state.liquidity[i] + old_agent.holdings[i]):
        raise AssertionError('System-wide asset total is wrong.')
    if new_state.lrna[i] + new_agent.holdings['LRNA'] < old_state.lrna[i] + old_agent.holdings['LRNA']:
        raise AssertionError('System-wide LRNA decreased.')

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


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01))
def test_lrna_swap_buy_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'

    delta_ra = 1000
    delta_ra_feeless = delta_ra / (1 - asset_fee)

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.swap_lrna(initial_state, old_agent, delta_ra, 0, i, lrna_mint_pct=1.0)
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, delta_ra_feeless, 0, i,
                                                            lrna_mint_pct=1.0)
    feeless_spot_price = feeless_swap_state.price(feeless_swap_state, i)
    spot_price = swap_state.price(swap_state, i)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01))
def test_lrna_swap_sell_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'

    delta_qa = -1000

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.swap_lrna(initial_state, old_agent, 0, delta_qa, i, lrna_mint_pct=1.0)
    feeless_swap_state, feeless_swap_agent = oamm.swap_lrna(feeless_state, old_agent, 0, delta_qa, i, lrna_mint_pct=1.0)
    feeless_spot_price = feeless_swap_state.price(feeless_swap_state, i)
    spot_price = swap_state.price(swap_state, i)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_sell_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float,
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_ri = 1000

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.swap(initial_state, old_agent, j, i, 0, delta_ri, lrna_mint_pct=1.0)
    feeless_swap_state, feeless_swap_agent = oamm.swap(feeless_state, old_agent, j, i, 0, delta_ri, lrna_mint_pct=1.0)
    feeless_spot_price = feeless_swap_state.price(feeless_swap_state, j)
    spot_price = swap_state.price(swap_state, j)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_buy_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=0.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_rj = 1000
    delta_rj_feeless = delta_rj / (1 - asset_fee)

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.swap(initial_state, old_agent, j, i, delta_rj, 0, lrna_mint_pct=1.0)
    feeless_swap_state, feeless_swap_agent = oamm.swap(feeless_state, old_agent, j, i, delta_rj_feeless, 0,
                                                       lrna_mint_pct=1.0)
    feeless_spot_price = feeless_swap_state.price(feeless_swap_state, j)
    spot_price = swap_state.price(swap_state, j)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_sell_with_partial_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float,
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_ri = 1000

    # Test with trader buying asset i
    swap_state_10, swap_agent_10 = oamm.swap(initial_state, old_agent, j, i, 0, delta_ri, lrna_mint_pct=1.0)
    swap_state_5, swap_agent_5 = oamm.swap(initial_state, old_agent, j, i, 0, delta_ri, lrna_mint_pct=0.5)
    swap_state_0, swap_agent_0 = oamm.swap(initial_state, old_agent, j, i, 0, delta_ri, lrna_mint_pct=0.0)

    spot_price_10 = swap_state_10.price(swap_state_10, j)
    spot_price_5 = swap_state_5.price(swap_state_5, j)
    spot_price_0 = swap_state_0.price(swap_state_0, j)

    if swap_state_10.fail == '' and swap_state_5.fail == '' and swap_state_0.fail == '':
        if spot_price_10 <= spot_price_5:
            raise AssertionError('Spot price is wrong.')
        if spot_price_5 <= spot_price_0:
            raise AssertionError('Spot price is wrong.')


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


# @given(omnipool_config(token_count=4), st.floats(min_value=0.1, max_value=1), st.floats(min_value=0.1, max_value=1))
# def test_slip_fees(initial_state: oamm.OmnipoolState, lrna_slip_rate: float, asset_slip_rate: float):
#     initial_state.lrna_fee = oamm.slip_fee(lrna_slip_rate, minimum_fee=0.0001)
#     initial_state.asset_fee = oamm.slip_fee(asset_slip_rate, minimum_fee=0.0001)
#     initial_state.withdrawal_fee = False
#     initial_agent = Agent(holdings={tkn: 10000000 for tkn in initial_state.asset_list})
#     tkn_buy = initial_state.asset_list[2]
#     tkn_sell = initial_state.asset_list[3]
#     sell_quantity = 1
#     sell_state, sell_agent = oamm.swap(initial_state, initial_agent, tkn_buy, tkn_sell, sell_quantity=sell_quantity)
#     split_sell_state, split_sell_agent = initial_state.copy(), initial_agent.copy()
#     next_state, next_agent = {}, {}
#     for i in range(2):
#         next_state[i], next_agent[i] = oamm.swap(
#             old_state=split_sell_state,
#             old_agent=split_sell_agent,
#             tkn_sell=tkn_sell,
#             tkn_buy=tkn_buy,
#             sell_quantity=sell_quantity / 2
#         )
#         split_sell_state, split_sell_agent = next_state[i], next_agent[i]
#     if split_sell_agent.holdings[tkn_buy] < sell_agent.holdings[tkn_buy]:
#         raise AssertionError('Agent failed to save money by splitting the sell order.')
#
#     buy_quantity = 1
#     buy_state, buy_agent = oamm.swap(initial_state, initial_agent, tkn_buy, tkn_sell, buy_quantity=buy_quantity)
#     split_buy_state, split_buy_agent = initial_state.copy(), initial_agent.copy()
#     next_state, next_agent = {}, {}
#     for i in range(2):
#         next_state[i], next_agent[i] = oamm.swap(
#             old_state=split_buy_state,
#             old_agent=split_buy_agent,
#             tkn_sell=tkn_sell,
#             tkn_buy=tkn_buy,
#             buy_quantity=buy_quantity / 2
#         )
#         split_buy_state, split_buy_agent = next_state[i], next_agent[i]
#     if split_buy_agent.holdings[tkn_sell] < buy_agent.holdings[tkn_sell]:
#         raise AssertionError('Agent failed to save money by splitting the buy order.')
#
#     if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
#          + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
#             != pytest.approx(buy_agent.holdings[tkn_sell] + buy_agent.holdings[tkn_buy]
#                              + buy_state.liquidity[tkn_sell] + buy_state.liquidity[tkn_buy])):
#         raise AssertionError('Asset quantity is not constant after trade (one-part)')
#
#     if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
#          + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
#             != pytest.approx(split_buy_agent.holdings[tkn_sell] + split_buy_agent.holdings[tkn_buy]
#                              + split_buy_state.liquidity[tkn_sell] + split_buy_state.liquidity[tkn_buy])):
#         raise AssertionError('Asset quantity is not constant after trade (two-part)')
#

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
    events = run.run(initial_state, time_steps=time_steps, silent=True)


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

    initial_omnipool = oamm.OmnipoolState(
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
        pools={'omnipool': initial_omnipool},
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

    initial_omnipool = oamm.OmnipoolState(
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
        pools={'omnipool': initial_omnipool},
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


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_dynamic_fees_empty_block(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                  oracle_volume_in: list[float], oracle_volume_out: list[float],
                                  oracle_prices: list[float], n):
    lrna_fees = [0.0005, 0.0010, 0.0050]
    asset_fees = [0.01, 0.0025, 0.0040]

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

    init_lrna_fees = {
        'HDX': lrna_fees[0],
        'USD': lrna_fees[1],
        'DOT': lrna_fees[2],
    }

    init_asset_fees = {
        'HDX': asset_fees[0],
        'USD': asset_fees[1],
        'DOT': asset_fees[2],
    }

    asset_fee_params = {
        'minimum': 0.0025,
        'amplification': 0.2,
        'raise_oracle_name': 'oracle',
        'decay': 0.00005,
        'fee_max': 0.4,
    }

    lrna_fee_params = {
        'minimum': 0.0005,
        'amplification': 0.04,
        'raise_oracle_name': 'oracle',
        'decay': 0.00001,
        'fee_max': 0.1,
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'oracle': n
        },
        asset_fee={
            'HDX': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
            'USD': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
            'DOT': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
        },
        lrna_fee={
            'HDX': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
            'USD': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
            'DOT': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
        },
        last_oracle_values={
            'oracle': copy.deepcopy(init_oracle)
        },
        last_lrna_fee=copy.deepcopy(init_lrna_fees),
        last_asset_fee=copy.deepcopy(init_asset_fees),
    )

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={}
    )

    events = run.run(initial_state=initial_state, time_steps=1, silent=True)
    omnipool = events[0].pools['omnipool']
    omnipool_oracle = omnipool.oracles['oracle']
    for tkn in ['HDX', 'USD', 'DOT']:
        x = (omnipool_oracle.volume_out[tkn] - omnipool_oracle.volume_in[tkn]) / omnipool_oracle.liquidity[tkn]

        df = -lrna_fee_params['amplification'] * x - lrna_fee_params['decay']
        expected_lrna_fee = min(max(init_lrna_fees[tkn] + df, lrna_fee_params['minimum']), lrna_fee_params['fee_max'])
        if omnipool.last_lrna_fee[tkn] != pytest.approx(expected_lrna_fee, rel=1e-15):
            raise AssertionError('LRNA fee is not correct.')

        df = asset_fee_params['amplification'] * x - asset_fee_params['decay']
        expected_asset_fee = min(max(init_asset_fees[tkn] + df, asset_fee_params['minimum']),
                                 asset_fee_params['fee_max'])
        if omnipool.last_fee[tkn] != pytest.approx(expected_asset_fee, rel=1e-15):
            raise AssertionError('Asset fee is not correct.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
    st.floats(min_value=-1000, max_value=1000),
    st.lists(st.floats(min_value=0.0005, max_value=0.10), min_size=3, max_size=3),
    st.lists(st.floats(min_value=0.0025, max_value=0.40), min_size=3, max_size=3),
    st.lists(st.floats(min_value=0.001, max_value=100), min_size=2, max_size=2),
    st.lists(st.floats(min_value=0.000001, max_value=0.0001), min_size=2, max_size=2),
)
def test_dynamic_fees_with_trade(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                 oracle_volume_in: list[float], oracle_volume_out: list[float],
                                 oracle_prices: list[float], n, trade_size: float, lrna_fees: list[float],
                                 asset_fees: list[float], amp: list[float], decay: list[float]):
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

    init_lrna_fees = {
        'HDX': lrna_fees[0],
        'USD': lrna_fees[1],
        'DOT': lrna_fees[2],
    }

    init_asset_fees = {
        'HDX': asset_fees[0],
        'USD': asset_fees[1],
        'DOT': asset_fees[2],
    }

    asset_fee_params = {
        'minimum': 0.0025,
        'amplification': amp[0],
        'raise_oracle_name': 'oracle',
        'decay': decay[0],
        'fee_max': 0.4,
    }

    lrna_fee_params = {
        'minimum': 0.0005,
        'amplification': amp[1],
        'raise_oracle_name': 'oracle',
        'decay': decay[1],
        'fee_max': 0.1,
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'oracle': n
        },
        asset_fee={
            'HDX': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
            'USD': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
            'DOT': dynamicadd_asset_fee(
                minimum=asset_fee_params['minimum'],
                amplification=asset_fee_params['amplification'],
                raise_oracle_name=asset_fee_params['raise_oracle_name'],
                decay=asset_fee_params['decay'],
                fee_max=asset_fee_params['fee_max'],
            ),
        },
        lrna_fee={
            'HDX': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
            'USD': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
            'DOT': dynamicadd_lrna_fee(
                minimum=lrna_fee_params['minimum'],
                amplification=lrna_fee_params['amplification'],
                raise_oracle_name=lrna_fee_params['raise_oracle_name'],
                decay=lrna_fee_params['decay'],
                fee_max=lrna_fee_params['fee_max'],
            ),
        },
        last_oracle_values={
            'oracle': copy.deepcopy(init_oracle)
        },
        last_lrna_fee=copy.deepcopy(init_lrna_fees),
        last_asset_fee=copy.deepcopy(init_asset_fees),
    )

    trader_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={
            'trader': Agent(
                holdings=trader_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_size,
                    sell_asset='USD',
                    buy_asset='DOT'
                )
            ),
        }
    )

    events = run.run(initial_state=initial_state, time_steps=2, silent=True)

    # test non-empty block fee dynamics

    omnipool = events[1].pools['omnipool']
    prev_lrna_fees = events[0].pools['omnipool'].last_lrna_fee
    prev_asset_fees = events[0].pools['omnipool'].last_fee
    omnipool_oracle = omnipool.oracles['oracle']
    for tkn in ['HDX', 'USD', 'DOT']:
        x = (omnipool_oracle.volume_out[tkn] - omnipool_oracle.volume_in[tkn]) / omnipool_oracle.liquidity[tkn]

        df = -lrna_fee_params['amplification'] * x - lrna_fee_params['decay']
        expected_lrna_fee = min(max(prev_lrna_fees[tkn] + df, lrna_fee_params['minimum']), lrna_fee_params['fee_max'])
        if omnipool.last_lrna_fee[tkn] != pytest.approx(expected_lrna_fee, rel=1e-15):
            raise AssertionError('LRNA fee is not correct.')

        df = asset_fee_params['amplification'] * x - asset_fee_params['decay']
        expected_asset_fee = min(max(prev_asset_fees[tkn] + df, asset_fee_params['minimum']),
                                 asset_fee_params['fee_max'])
        if omnipool.last_fee[tkn] != pytest.approx(expected_asset_fee, rel=1e-15):
            raise AssertionError('Asset fee is not correct.')


@given(asset_quantity_strategy, omnipool_config())
def test_LP_delta_r(lp_amount, omnipool: oamm.OmnipoolState):
    agent = Agent(
        holdings={
            tkn: 100000000 for tkn in omnipool.asset_list
        }
    )
    initial_asset_holdings = copy.deepcopy(agent.holdings)
    oamm.execute_add_liquidity(
        state=omnipool,
        agent=agent,
        quantity=lp_amount,
        tkn_add='HDX'
    )
    if agent.holdings['HDX'] != initial_asset_holdings['HDX'] - agent.delta_r[('omnipool', 'HDX')]:
        raise AssertionError('Delta_r is not correct.')


@given(omnipool_reasonable_config(remove_liquidity_volatility_threshold=0.01))
def test_volatility_limit(omnipool: oamm.OmnipoolState):
    agent = Agent(holdings={'HDX': 1000000000})
    oamm.execute_add_liquidity(omnipool, agent, quantity=1000, tkn_add='HDX')
    oamm.execute_swap(omnipool, agent, tkn_sell='HDX', tkn_buy='LRNA', sell_quantity=omnipool.liquidity['HDX'] / 200)
    oamm.execute_remove_liquidity(omnipool, agent, quantity=1000, tkn_remove='HDX')

    if not omnipool.fail:
        raise ValueError("Volatility limit should be exceeded")

    # go forward one block, which should be enough for the volatility to decay
    updated_pool = omnipool.copy().update()

    oamm.execute_remove_liquidity(updated_pool, agent, agent.holdings[('omnipool', 'HDX')], tkn_remove='HDX')
    if updated_pool.fail:
        raise ValueError("Volatility limit should not be exceeded")


@given(omnipool_reasonable_config(), st.floats(min_value=0.01, max_value=0.1), st.floats(min_value=0.01, max_value=0.1))
def test_LP_limits(omnipool: oamm.OmnipoolState, max_withdrawal_per_block, max_lp_per_block):
    omnipool.max_withdrawal_per_block = max_withdrawal_per_block
    omnipool.max_lp_per_block = max_lp_per_block
    agent = Agent(holdings={'HDX': 10000000000})
    oamm.execute_add_liquidity(
        state=omnipool,
        agent=agent,
        tkn_add='HDX',
        quantity=omnipool.liquidity['HDX'] * max_lp_per_block
    )
    if omnipool.fail:
        raise AssertionError('Valid LP operation failed.')
    omnipool.update()
    oamm.execute_add_liquidity(
        state=omnipool,
        agent=agent,
        tkn_add='HDX',
        quantity=omnipool.liquidity['HDX'] * max_lp_per_block + 1
    )
    if not omnipool.fail:
        raise AssertionError('Invalid LP operation succeeded.')
    omnipool.update()
    # add liquidity again to test remove liquidity
    oamm.execute_add_liquidity(
        state=omnipool,
        agent=agent,
        tkn_add='HDX',
        quantity=omnipool.liquidity['HDX'] * max_lp_per_block
    )
    if omnipool.fail:
        raise AssertionError('Second LP operation failed.')
    withdraw_quantity = agent.holdings[('omnipool', 'HDX')]
    total_shares = omnipool.shares['HDX']
    oamm.execute_remove_liquidity(
        state=omnipool,
        agent=agent,
        tkn_remove='HDX',
        quantity=withdraw_quantity  # agent.holdings[('omnipool', 'HDX')]
    )
    if withdraw_quantity / total_shares > max_withdrawal_per_block and not omnipool.fail:
        raise AssertionError('Agent was able to remove too much liquidity.')
    omnipool.update()
    oamm.execute_remove_liquidity(
        state=omnipool,
        agent=agent,
        tkn_remove='HDX',
        quantity=omnipool.shares['HDX'] * max_withdrawal_per_block
    )
    if omnipool.fail:
        raise AssertionError('Agent was not able to remove liquidity.')


@given(
    st.floats(min_value=0.50, max_value=1.5)
)
def test_liquidity_operations_and_spot_prices(oracle_mult):
    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 44000000, 'LRNA': 275143},
            'WETH': {'liquidity': 1400, 'LRNA': 2276599},
            'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
            'DOT': {'liquidity': 88000, 'LRNA': 546461},
            'WBTC': {'liquidity': 47, 'LRNA': 1145210},
        },
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
    )

    agent = Agent(holdings={'DOT': 10000})
    add_state, add_agent = oamm.execute_add_liquidity(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_add='DOT',
        quantity=agent.holdings['DOT']
    )

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=add_state.copy(),
        agent=add_agent.copy(),
        tkn_remove='DOT',
        quantity=add_agent.holdings[('omnipool', 'DOT')]
    )

    if add_agent.holdings[('omnipool', 'DOT')] == 0:
        raise

    for tkn in omnipool.asset_list:
        initial_price = price(omnipool, tkn)
        add_price = price(add_state, tkn)
        remove_price = price(remove_state, tkn)
        if initial_price != pytest.approx(add_price, rel=1e-15):
            raise AssertionError('Price is not correct after add liquidity.')

        if initial_price != pytest.approx(remove_price, rel=1e-15):
            raise AssertionError('Price is not correct after remove liquidity.')


# @settings(max_examples=10000)
@given(
    st.floats(min_value=0, max_value=0.1, exclude_min=True),
    st.floats(min_value=-0.02, max_value=0, exclude_max=True),
    st.floats(min_value=0.50, max_value=1.5)
)
def test_lowering_price(lp_multiplier, price_movement, oracle_mult):
    # def test_lowering_price(lp_multiplier, price_movement):
    # lp_multiplier = 0.1
    # price_movement = -0.1
    # mult = 0.99

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    trade_size = tokens['DOT']['liquidity'] * 1 / math.sqrt(1 + price_movement) - tokens['DOT']['liquidity']

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    swap_state, swap_agent = oamm.execute_swap(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_sell='DOT',
        tkn_buy='DAI',
        sell_quantity=trade_size
    )

    add_state, add_agent = oamm.execute_add_liquidity(
        state=swap_state.copy(),
        agent=swap_agent.copy(),
        tkn_add='DOT',
        quantity=swap_state.liquidity['DOT'] * lp_multiplier
    )

    global_state = GlobalState(
        pools={'omnipool': add_state},
        agents={'attacker': add_agent},
        external_market=market_prices
    )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=arbed_pool.copy(),
        agent=arbed_agent.copy(),
        tkn_remove='DOT',
        quantity=arbed_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = oamm.cash_out_omnipool(omnipool, agent, market_prices)
    final_value = oamm.cash_out_omnipool(remove_state, remove_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


def test_add_and_remove_liquidity():
    lp_multiplier = 0.1
    oracle_mult = 0.99

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    add_state, add_agent = oamm.execute_add_liquidity(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_add='DOT',
        quantity=omnipool.liquidity['DOT'] * lp_multiplier
    )

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=add_state.copy(),
        agent=add_agent.copy(),
        tkn_remove='DOT',
        quantity=add_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = oamm.cash_out_omnipool(omnipool, agent, market_prices)
    final_value = oamm.cash_out_omnipool(remove_state, remove_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


# @settings(max_examples=1)
@given(
    st.floats(min_value=0, max_value=0.10, exclude_min=True),
    st.floats(min_value=0, max_value=0.01, exclude_min=True),
    # st.floats(min_value=0.90, max_value=1.1)
)
def test_add_liquidity_exploit(lp_multiplier, trade_mult):
    oracle_mult = 1.0
    # lp_multiplier = 0.5
    # trade_mult = 0.5

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}
    trade_size = tokens['DOT']['liquidity'] * trade_mult

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    swap_state, swap_agent = oamm.execute_swap(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_sell='DAI',
        tkn_buy='DOT',
        buy_quantity=trade_size
    )

    add_state, add_agent = oamm.execute_add_liquidity(
        state=swap_state.copy(),
        agent=swap_agent.copy(),
        tkn_add='DOT',
        quantity=swap_state.liquidity['DOT'] * lp_multiplier
    )

    global_state = GlobalState(
        pools={'omnipool': add_state},
        agents={'attacker': add_agent},
        external_market=market_prices
    )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=arbed_pool.copy(),
        agent=arbed_agent.copy(),
        tkn_remove='DOT',
        quantity=arbed_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = oamm.cash_out_omnipool(omnipool, agent, market_prices)
    final_value = oamm.cash_out_omnipool(remove_state, remove_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


@given(
    st.floats(min_value=0, max_value=0.10, exclude_min=True),
    st.floats(min_value=0, max_value=0.01, exclude_min=True),
    # st.floats(min_value=0.90, max_value=1.1)
)
def test_add_liquidity_exploit_sell(lp_multiplier, trade_mult):
    oracle_mult = 1.0
    # lp_multiplier = 0.5
    # trade_mult = 0.5

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}
    trade_size = tokens['DOT']['liquidity'] * trade_mult

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    swap_state, swap_agent = oamm.execute_swap(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_sell='DOT',
        tkn_buy='DAI',
        sell_quantity=trade_size
    )

    add_state, add_agent = oamm.execute_add_liquidity(
        state=swap_state.copy(),
        agent=swap_agent.copy(),
        tkn_add='DOT',
        quantity=swap_state.liquidity['DOT'] * lp_multiplier
    )

    global_state = GlobalState(
        pools={'omnipool': add_state},
        agents={'attacker': add_agent},
        external_market=market_prices
    )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=arbed_pool.copy(),
        agent=arbed_agent.copy(),
        tkn_remove='DOT',
        quantity=arbed_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = oamm.cash_out_omnipool(omnipool, agent, market_prices)
    final_value = oamm.cash_out_omnipool(remove_state, remove_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


def test_withdraw_exploit():
    oracle_mult = 1.0
    lp_multiplier = 0.1
    trade_mult = 0.01

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}
    trade_size = tokens['DOT']['liquidity'] * trade_mult

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    add_state, add_agent = oamm.execute_add_liquidity(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_add='DOT',
        quantity=omnipool.liquidity['DOT'] * lp_multiplier
    )

    swap_state, swap_agent = oamm.execute_swap(
        state=add_state.copy(),
        agent=add_agent.copy(),
        tkn_sell='DAI',
        tkn_buy='DOT',
        buy_quantity=trade_size
    )

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=swap_state.copy(),
        agent=swap_agent.copy(),
        tkn_remove='DOT',
        quantity=swap_agent.holdings[('omnipool', 'DOT')]
    )

    global_state = GlobalState(
        pools={'omnipool': remove_state},
        agents={'attacker': remove_agent},
        external_market=market_prices
    )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    initial_value = oamm.cash_out_omnipool(omnipool, agent, market_prices)
    final_value = oamm.cash_out_omnipool(arbed_pool, arbed_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


@settings(max_examples=1)
@given(
    st.floats(min_value=0, max_value=0.05, exclude_min=True),
    st.floats(min_value=0, max_value=0.1, exclude_min=True),
    st.floats(min_value=0.50, max_value=1.5)
)
def test_swap_exploit(lp_multiplier, trade_mult, oracle_mult):
    lp_multiplier = 0.2
    trade_mult = 0.01
    oracle_mult = 0.99

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    trade_size = tokens['DOT']['liquidity'] * trade_mult

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: oamm.usd_price(omnipool, tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    add_state, add_agent = oamm.execute_add_liquidity(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_add='DOT',
        quantity=omnipool.liquidity['DOT'] * lp_multiplier
    )

    swap_state, swap_agent = oamm.execute_swap(
        state=add_state.copy(),
        agent=add_agent.copy(),
        tkn_sell='DOT',
        tkn_buy='DAI',
        sell_quantity=trade_size
    )

    remove_state, remove_agent = oamm.execute_remove_liquidity(
        state=swap_state.copy(),
        agent=swap_agent.copy(),
        tkn_remove='DOT',
        quantity=swap_agent.holdings[('omnipool', 'DOT')]
    )

    swap_alone_state, swap_alone_agent = oamm.execute_swap(
        state=omnipool.copy(),
        agent=agent.copy(),
        tkn_sell='DOT',
        tkn_buy='DAI',
        sell_quantity=trade_size
    )

    swap_alone_dai = oamm.cash_out_omnipool(swap_alone_state, swap_alone_agent, market_prices)
    manipulated_dai = oamm.cash_out_omnipool(remove_state, remove_agent, market_prices)
    profit = manipulated_dai - swap_alone_dai
    if profit > 0:
        raise


@given(
    omnipool_reasonable_config(),
    st.floats(min_value=1e-8, max_value=0.02),
    st.booleans(),
    st.floats(min_value=1e-8, max_value=0.1),
    st.floats(min_value=0.1, max_value=10.0),
)
def test_withdraw_manipulation(
        initial_state: oamm.OmnipoolState,
        price_move: float,
        price_move_is_up: bool,
        lp_percent: float,
        price_ratio: float
):
    # uncommenting this will cause the test to fail, demonstrating that oracle length > 1 helps solve the problem
    # initial_state.oracles = {'price': Oracle(
    #     first_block=initial_state.current_block,
    #     sma_equivalent_length=1
    # )}

    agent_holdings = {
        tkn: 10000000 / oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list
    }

    initial_agent = Agent(
        holdings=agent_holdings
    )

    asset_index = 1
    options = copy.copy(initial_state.asset_list)
    lp_token = options[asset_index % len(options)]
    options.remove(lp_token)
    trade_token = options[asset_index % len(options)]
    lp_quantity = int(initial_state.liquidity[lp_token] * lp_percent)

    initial_agent.holdings[('omnipool', lp_token)] = lp_quantity
    initial_agent.share_prices[('omnipool', lp_token)] = price_ratio

    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    # trade to manipulate the price
    signed_price_move = price_move if price_move_is_up else -price_move
    first_trade = initial_state.liquidity[lp_token] * (1 - 1 / math.sqrt(1 + signed_price_move))
    trade_state, trade_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell=trade_token,
        tkn_buy=lp_token,
        buy_quantity=first_trade
    )

    withdraw_state, withdraw_agent = oamm.execute_remove_liquidity(
        state=trade_state.copy(),
        agent=trade_agent.copy(),
        quantity=trade_agent.holdings[('omnipool', lp_token)],
        tkn_remove=lp_token
    )

    glob = omnipool_arbitrage(pool_id='omnipool').execute(
        state=GlobalState(
            pools={
                'omnipool': withdraw_state.copy()
            },
            agents={
                'agent': withdraw_agent.copy()
            },
            external_market=market_prices
        ),
        agent_id='agent'
    )

    final_state, final_agent = glob.pools['omnipool'], glob.agents['agent']

    profit = (
            oamm.cash_out_omnipool(final_state, final_agent, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    if profit > 0:
        raise AssertionError(f'profit with manipulation {profit} > 0')


@given(
    omnipool_config(imbalance=0, asset_fee=0, lrna_fee=0),
    st.floats(min_value=0, max_value=0.02),
    st.floats(min_value=0.001, max_value=0.10)
)
def test_add_manipulation(
        initial_state: oamm.OmnipoolState,
        price_move: float,
        lp_percent: float
):
    initial_state.remove_liquidity_volatility_threshold = 0.01
    initial_state.trade_limit_per_block = 0.05
    initial_state.max_withdrawal_per_block = 0.05
    initial_state.max_lp_per_block = 0.05
    # uncommenting this will cause the test to fail, demonstrating that oracle length > 1 helps solve the problem
    # initial_state.oracles = {'price': Oracle(
    #     first_block=initial_state.current_block,
    #     sma_equivalent_length=1
    # )}

    agent_holdings = {
        tkn: 1000000 / oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list
    }

    initial_agent = Agent(
        holdings=agent_holdings
    )

    asset_index = 1
    options = copy.copy(initial_state.asset_list)
    asset1 = options[asset_index % len(options)]
    options.remove(asset1)
    asset2 = options[asset_index % len(options)]
    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    # trade to manipulate the price
    first_trade = initial_state.liquidity[asset1] * (1 - 1 / math.sqrt(1 + price_move))
    trade_state, trade_agent = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell=asset2,
        tkn_buy=asset1,
        buy_quantity=first_trade
    )

    # add liquidity
    lp_quantity = lp_percent * initial_agent.holdings[asset1]
    add_state, add_agent = oamm.execute_add_liquidity(
        state=trade_state.copy(),
        agent=trade_agent.copy(),
        tkn_add=asset1,
        quantity=min(lp_quantity, trade_state.liquidity[asset1] * trade_state.max_lp_per_block)
    )

    lp_quantity = lp_percent * initial_agent.holdings[asset2]
    add_state, add_agent = oamm.execute_add_liquidity(
        state=add_state,
        agent=add_agent,
        tkn_add=asset2,
        quantity=lp_quantity
    )

    glob = omnipool_arbitrage(pool_id='omnipool').execute(
        state=GlobalState(
            pools={
                'omnipool': add_state.copy()
            },
            agents={
                'agent': add_agent.copy()
            },
            external_market=market_prices
        ),
        agent_id='agent'
    )

    sell_state, sell_agent = glob.pools['omnipool'], glob.agents['agent']

    profit = (
            oamm.cash_out_omnipool(sell_state, sell_agent, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    if profit > 0:
        raise AssertionError(f'profit with manipulation {profit} > 0')


@given(
    omnipool_config(imbalance=0),
    st.integers(min_value=1, max_value=7),
    st.floats(min_value=0.1, max_value=1.0),
    st.floats(min_value=1000, max_value=1000000),
)
def test_trade_manipulation(
        initial_state: oamm.OmnipoolState,
        asset_index: int,
        lp_percent: float,
        sell_quantity: float,
):
    initial_state.remove_liquidity_volatility_threshold = 0.01
    initial_state.trade_limit_per_block = 0.05
    initial_state.max_withdrawal_per_block = 0.05
    initial_state.max_lp_per_block = 0.05

    initial_agent = Agent(
        holdings=copy.copy(initial_state.liquidity)
    )

    options = copy.copy(initial_state.asset_list)
    asset1 = options[asset_index % len(options)]
    options.remove(asset1)
    asset2 = options[asset_index % len(options)]
    market_prices = {tkn: oamm.usd_price(initial_state, tkn) for tkn in initial_state.asset_list}

    lp1_state, lp1_agent = oamm.execute_add_liquidity(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_add=asset1,
        quantity=min(
            lp_percent * initial_state.liquidity[asset1],
            initial_state.liquidity[asset1] * initial_state.max_lp_per_block
        )
    )

    lp2_state, lp2_agent = oamm.execute_add_liquidity(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_add=asset2,
        quantity=min(
            lp_percent * initial_state.liquidity[asset2],
            initial_state.liquidity[asset2] * initial_state.max_lp_per_block
        )
    )

    trade_state_1, trade_agent_1 = oamm.execute_remove_liquidity(
        *oamm.execute_swap(
            state=lp1_state.copy(),
            agent=lp1_agent.copy(),
            tkn_sell=asset1,
            tkn_buy=asset2,
            sell_quantity=sell_quantity
        ),
        tkn_remove=asset1,
        quantity=lp1_agent.holdings[('omnipool', asset1)]
    )

    trade_state_2, trade_agent_2 = oamm.execute_remove_liquidity(
        *oamm.execute_swap(
            state=lp2_state.copy(),
            agent=lp2_agent.copy(),
            tkn_sell=asset1,
            tkn_buy=asset2,
            sell_quantity=sell_quantity
        ),
        tkn_remove=asset2,
        quantity=lp2_agent.holdings[('omnipool', asset2)]
    )

    trade_state_3, trade_agent_3 = oamm.execute_swap(
        state=initial_state.copy(),
        agent=initial_agent.copy(),
        tkn_sell=asset1,
        tkn_buy=asset2,
        sell_quantity=sell_quantity
    )

    lp1_profit = (
            oamm.cash_out_omnipool(trade_state_1, trade_agent_1, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    lp2_profit = (
            oamm.cash_out_omnipool(trade_state_2, trade_agent_2, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    no_lp_profit = (
            oamm.cash_out_omnipool(trade_state_3, trade_agent_3, market_prices)
            - oamm.cash_out_omnipool(initial_state, initial_agent, market_prices)
    )

    if lp1_profit > no_lp_profit and trade_state_1.fail == '' and trade_state_3.fail == '':
        raise AssertionError(f'profit with LP asset1 ({asset1}) = {lp1_profit} > without {no_lp_profit}')

    if lp2_profit > no_lp_profit and trade_state_2.fail == '' and trade_state_3.fail == '':
        raise AssertionError(f'profit with LP asset2 ({asset2}) = {lp2_profit} > without {no_lp_profit}')
