import copy
from pprint import pprint
import random

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity, Phase, reproduce_failure

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState, simulate_swap
from mpmath import mp, mpf
import highspy
import numpy as np

from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution, \
    _find_solution_unrounded, add_buy_deltas, round_solution, find_solution_outer_approx, _solve_inclusion_problem, \
    ICEProblem, _get_leftover_bounds, AmmIndexObject
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.xyk_amm import XykState


##################################
# Functional tests #
##################################


def get_token_list(omnipool: OmnipoolState, amm_list: list[StableSwapPoolState]) -> list[str]:
    token_list = ['LRNA']
    for amm in [omnipool] + amm_list:
        token_list.extend([tkn for tkn in amm.asset_list])
    return token_list


def get_markets_minimal(asset_fee = 0.0025, lrna_fee = 0.0005, ss_fee = 0.0005):
    prices = {'HDX': 0.013, 'DOT': 9, 'vDOT': 13, '2-Pool': 1.01, '4-Pool': 1.01, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.08, 'DOT': 0.5, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.02}
    total_lrna = 65000000

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(asset_fee),
        lrna_fee=mpf(lrna_fee)
    )

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="4-Pool"
    )

    amm_list = [stablepool, stablepool4]
    return initial_state, amm_list


def get_markets_even(
        asset_fee: float = 0.0025,
        lrna_fee: float = 0.0005,
        ss_fee: float = 0.0005,
        include_2_pool: bool = True,
        include_4_pool: bool = True
) -> tuple[OmnipoolState, list[StableSwapPoolState]]:
    """Get markets with similar orders of magnitude across tokens, to avoid rounding errors."""
    prices = {'HDX': 0.1, 'DOT': 5, 'vDOT': 6, '2-Pool': 1.01, '4-Pool': 1.01, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.15, 'DOT': 0.25, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.2}
    total_lrna = 65000000

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(asset_fee),
        lrna_fee=mpf(lrna_fee)
    )

    amm_list = []
    if include_2_pool:
        sp_tokens = {
            "USDT": 7800000,
            "USDC": 5200000
        }
        stablepool = StableSwapPoolState(
            tokens=sp_tokens,
            amplification=1000,
            trade_fee=ss_fee,
            unique_id="2-Pool"
        )
        amm_list.append(stablepool)

    if include_4_pool:
        sp4_tokens = {
            "USDC": 52000000,
            "USDT": 2600000,
            "DAI": 2600000,
            "USDT2": 2600000
        }
        stablepool4 = StableSwapPoolState(
            tokens=sp4_tokens,
            amplification=1000,
            trade_fee=ss_fee,
            unique_id="4-Pool"
        )
        amm_list.append(stablepool4)

    return initial_state, amm_list


@given(st.integers(min_value=0, max_value=10), st.integers(min_value=2, max_value=5))
def test_AmmIndexObject_stableswap(offset: int, tkn_ct: int):
    from hydradx.model.amm.omnix_solver_simple import AmmIndexObject
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    # order of variables should be: X_0, X_1, ..., X_4, L_0, L_1, ..., L_4, a_0, a_1, ..., a_4
    n_amm = len(stablepool4.liquidity) + 1
    amm_index = AmmIndexObject(stablepool4, offset)
    assert amm_index.shares_net == offset  # X_0
    assert amm_index.asset_net == [offset + i for i in range(1, n_amm)]  # X_1, ... X_4
    assert amm_index.shares_out == offset + n_amm  # L_0
    assert amm_index.asset_out == [offset + n_amm + i for i in range(1, n_amm)]  # L_1, ... L_4
    assert amm_index.aux == [offset + 2 * n_amm + i for i in range(n_amm)]  # a_0, ... a_4
    assert amm_index.net_is == slice(offset, offset + n_amm)
    assert amm_index.out_is == slice(offset + n_amm, offset + 2 * n_amm)
    assert amm_index.aux_is == slice(offset + 2 * n_amm, offset + 3 * n_amm)
    assert amm_index.num_vars == 3 * n_amm

    # amm_index = AmmIndexObject(stablepool4, offset, False)
    # assert amm_index.shares_net == offset  # X_0
    # assert amm_index.asset_net == [offset + i for i in range(1, n_amm)]  # X_1, ... X_4
    # assert amm_index.shares_out == offset + n_amm  # L_0
    # assert amm_index.asset_out == [offset + n_amm + i for i in range(1, n_amm)]  # L_1, ... L_4
    # assert amm_index.aux == []  # a_0, ... a_4
    # assert amm_index.net_is == slice(offset, offset + n_amm)
    # assert amm_index.out_is == slice(offset + n_amm, offset + 2 * n_amm)
    # assert len(np.arange(amm_index.aux_is.start, amm_index.aux_is.stop, amm_index.aux_is.step or 1)) == 0
    # assert amm_index.num_vars == 2 * n_amm


@given(st.integers(min_value=0, max_value=10))
def test_AmmIndexObject_xyk(offset: int):
    from hydradx.model.amm.omnix_solver_simple import AmmIndexObject
    liquidity = {"A": 1_000_000, "B": 1_000_000}
    xyk = XykState(tokens=liquidity)
    # order of variables should be: X_0, X_1, ..., X_4, L_0, L_1, ..., L_4, a_0, a_1, ..., a_4
    n_amm = 3
    amm_index = AmmIndexObject(xyk, offset)
    assert amm_index.shares_net == offset  # X_0
    assert amm_index.asset_net == [offset + i for i in range(1, n_amm)]  # X_1, ... X_4
    assert amm_index.shares_out == offset + n_amm  # L_0
    assert amm_index.asset_out == [offset + n_amm + i for i in range(1, n_amm)]  # L_1, ... L_4
    assert amm_index.aux == []  # a_0, ... a_4
    assert amm_index.net_is == slice(offset, offset + n_amm)
    assert amm_index.out_is == slice(offset + n_amm, offset + 2 * n_amm)
    assert amm_index.aux_is.start == amm_index.aux_is.stop
    assert amm_index.num_vars == 2 * n_amm


def check_all_cone_feasibility(s, cones, cone_sizes, tol=2e-5):
    from hydradx.model.amm.omnix_solver_simple import check_cone_feasibility
    res = check_cone_feasibility(s, cones, cone_sizes, tol)
    for _, _, cone_feas in res:
        if not cone_feas:
            return False
    return True


def test_get_amm_limits_A():
    from hydradx.model.amm.omnix_solver_simple import _get_amm_limits_A, AmmIndexObject
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = XykState(tokens=xyk_liquidity)
    amm_directions = []
    last_amm_deltas = []
    offset = 5
    k = 30

    for amm, trading_pairs in [[stablepool4, [list(range(ss_tkn_ct + 1)), [0, 2], [1, 3]]], [xyk, [[1, 2]]]]:
        amm_i = AmmIndexObject(amm, offset)
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        for trading_is in trading_pairs:
            trading_tkns = [all_trading_tkns[i] for i in trading_is]

            A_limits, b_limits, cones, cones_sizes = _get_amm_limits_A(amm, amm_i, amm_directions, last_amm_deltas,
                                                                       k, trading_tkns)
            assert A_limits.shape[1] == k
            # assert A_limits.shape[0] == 2 * len(trading_tkns) + 2 * (tkn_ct + 1 - len(trading_tkns))
            # in this case, A_limits should be enforcing that Li >= 0 for the restricted trading tokens
            # it should be enforcing that Xi == 0 and Li == 0 for the other tokens
            x = np.zeros(k)
            for i in trading_is:
                x[amm_i.shares_out + i] = 1  # set correct Lis to 1
            s = b_limits - A_limits @ x
            if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                raise AssertionError("Cone feasibility check failed for Li >= 0")
            # check if appropriate Xis != 0
            for val in [1, -1]:
                x_copy = copy.deepcopy(x)
                for i in trading_is:
                    x_copy[amm_i.shares_net + i] = val  # set correct Xis to 1
                s = b_limits - A_limits @ x_copy
                if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                    raise AssertionError("Cone feasibility check failed for Xis != 0")
            # next we check what happens if we make one of the restricted Lis negative
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_out + i] = -1  # make one of the Ls negative
                s = b_limits - A_limits @ x_copy
                if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                    raise AssertionError("Cone feasibility check should fail for negative Li")
            # next we check what happens if we make one of the other Xs or Ls non-zero
            non_trading_is = [i for i in range(len(amm.asset_list) + 1) if i not in trading_is]
            for i in non_trading_is:
                for val in [-1, 1]:
                    for offset in [amm_i.shares_net, amm_i.shares_out]:
                        x_copy = copy.deepcopy(x)
                        x_copy[offset + i] = val  # make variable nonzero
                        s = b_limits - A_limits @ x_copy
                        if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                            raise AssertionError("Cone feasibility check should fail for non-trading variables")
            # next we check what happens if we make Xj + Lj < 0
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_net + i] = -2  # Xj + Lj = -1
                s = b_limits - A_limits @ x_copy
                assert not check_all_cone_feasibility(s, cones, cones_sizes, tol=0)

        with pytest.raises(ValueError):  # if k is too small, we should raise an error
            _get_amm_limits_A(amm, amm_i, amm_directions, last_amm_deltas, amm_i.shares_net + amm_i.num_vars - 1,
                              trading_tkns)


@given(st.lists(st.booleans(), min_size=5, max_size=5),
       st.booleans(),
       st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=5, unique=True))
def test_get_amm_limits_A_directions(ss_dirs, xyk_dir, trading_indices):
    from hydradx.model.amm.omnix_solver_simple import _get_amm_limits_A, AmmIndexObject
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = XykState(tokens=xyk_liquidity)
    xyk_directions = ['none', 'buy' if xyk_dir else 'sell', 'sell' if xyk_dir else 'buy']
    ss_directions = ['buy' if dir else 'sell' for dir in ss_dirs]
    last_amm_deltas = []
    offset = 5
    k = 30
    for amm, directions, trading_pairs in [[stablepool4, ss_directions, [list(range(ss_tkn_ct + 1)), trading_indices]],
                                           [xyk, xyk_directions, [[1, 2]]]]:
        amm_i = AmmIndexObject(amm, offset)
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        for trading_is in trading_pairs:
            trading_tkns = [all_trading_tkns[i] for i in trading_is]
            A_limits, b_limits, cones, cones_sizes = _get_amm_limits_A(amm, amm_i, directions,
                                                                       last_amm_deltas,
                                                                       k, trading_tkns)
            assert A_limits.shape[1] == k
            # in this case, A_limits should be enforcing that Li >= 0 for the restricted trading tokens
            # it should be enforcing that Xi == 0 and Li == 0 for the other tokens
            x = np.zeros(k)
            for i in trading_is:
                x[amm_i.shares_out + i] = 1  # set correct Lis to 1
            s = b_limits - A_limits @ x
            assert check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
            # check if appropriate Xis != 0
            for i in trading_is:
                x[amm_i.shares_net + i] = 1 if directions[i] == 'buy' else -1  # set correct Xis to 1
            s = b_limits - A_limits @ x
            assert check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
            # try flipping an Xi in the wrong direction
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_net + i] *= -1
                s = b_limits - A_limits @ x_copy
                assert not check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
            # next we check what happens if we make one of the restricted Lis negative
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_out + i] = -1  # make one of the Ls negative
                s = b_limits - A_limits @ x_copy
                assert not check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
            # next we check what happens if we make one of the other Xs or Ls non-zero
            non_trading_is = [i for i in range(len(amm.asset_list) + 1) if i not in trading_is]
            for i in non_trading_is:
                for val in [-1, 1]:
                    for offset in [amm_i.shares_net, amm_i.shares_out]:
                        x_copy = copy.deepcopy(x)
                        x_copy[offset + i] = val  # make variable nonzero
                        s = b_limits - A_limits @ x_copy
                        if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                            raise AssertionError("Cone feasibility check should fail for non-trading variables")
            # next we check what happens if we make Xj + Lj < 0
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_net + i] = -2  # Xj + Lj = -1
                s = b_limits - A_limits @ x_copy
                assert not check_all_cone_feasibility(s, cones, cones_sizes, tol=0)


def test_get_xyk_bounds():
    from hydradx.model.amm.omnix_solver_simple import _get_xyk_bounds
    amm = XykState(tokens={"A": 1_000_000, "B": 2_000_000})  # spot price is 2 B = 1 A
    scaling = {tkn: 1 for tkn in (amm.asset_list + [amm.unique_id])}
    offset = 7
    amm_i = AmmIndexObject(amm, 7)
    k = 40
    A, b, cones, cones_sizes = _get_xyk_bounds(amm, amm_i, "None", k, scaling)
    x = np.zeros(k)
    # selling 5 B for 1 A should work
    b_sell_amt, a_buy_amt = 5, 1
    x[amm_i.asset_net[0]] = -a_buy_amt
    x[amm_i.asset_net[1]] = b_sell_amt
    x[amm_i.asset_out[0]] = a_buy_amt
    s = b - A @ x
    if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
        raise AssertionError("Cone feasibility check failed for valid XYK bounds")
    # selling 1 A for 1 5 should not work
    a_sell_amt, b_buy_amt = 1, 5
    x[amm_i.asset_net[1]] = -b_buy_amt
    x[amm_i.asset_net[0]] = a_sell_amt
    x[amm_i.asset_out[1]] = b_buy_amt
    s = b - A @ x
    if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
        raise AssertionError("Cone feasibility check should fail")
    # selling 1 A for 1 B should work
    a_sell_amt, b_buy_amt = 1, 1
    x[amm_i.asset_net[1]] = -b_buy_amt
    x[amm_i.asset_net[0]] = a_sell_amt
    x[amm_i.asset_out[1]] = b_buy_amt
    s = b - A @ x
    if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
        raise AssertionError("Cone feasibility check should succeed")
    # selling 1 B for 1 A should not work
    b_sell_amt, a_buy_amt = 1, 1
    x[amm_i.asset_net[0]] = -a_buy_amt
    x[amm_i.asset_net[1]] = b_sell_amt
    x[amm_i.asset_out[0]] = a_buy_amt
    s = b - A @ x
    if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
        raise AssertionError("Cone feasibility check should fail")

def test_no_intent_arbitrage():

    # test where stablepool shares have different values
    prices = {'HDX': 0.013, 'DOT': 9, 'vDOT': 13, '2-Pool': 1.01, '4-Pool': 0.99, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.08, 'DOT': 0.5, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.02}
    total_lrna = 65000000

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna},
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    ss_fee = 0.0005

    sp_tokens = {"USDT": 7600000,"USDC": 9200000}
    # sp_tokens = {"USDT": 16800000/2,"USDC": 16800000/2}
    stablepool = StableSwapPoolState(
        tokens=sp_tokens, amplification=1000, trade_fee=ss_fee, unique_id="2-Pool"
    )

    sp4_tokens = {"USDC": 600000, "USDT": 340000, "DAI": 365000, "USDT2": 330000}
    # sp4_tokens = {"USDC": 940000/2, "USDT": 940000/2, "DAI": 365000, "USDT2": 330000}
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens, amplification=1000, trade_fee=ss_fee, unique_id="4-Pool"
    )

    amm_list = [stablepool, stablepool4]
    intents = []
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    assert x['omnipool_deltas']['2-Pool'] >= 9000
    assert x['omnipool_deltas']['4-Pool'] <= -9000
    assert x['omnipool_deltas']['HDX'] <= -6000
    assert x['profit'] >= 6000
    final_amm_list = [amm.copy() for amm in amm_list]
    final_state = initial_state.copy()
    assert validate_and_execute_solution(final_state, final_amm_list, copy.deepcopy(intents),
                                         copy.deepcopy(x['deltas']), copy.deepcopy(x['omnipool_deltas']),
                                         copy.deepcopy(x['amm_deltas']),"HDX")

    # test where stablecoin prices are different in different stablepools

    prices = {'HDX': 0.013, 'DOT': 9, 'vDOT': 13, '2-Pool': 1.01, '4-Pool': 1.01, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.08, 'DOT': 0.5, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.02}
    total_lrna = 65000000

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna},
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    sp_tokens = {"USDT": 7600000 * 10, "USDC": 9200000}
    stablepool = StableSwapPoolState(
        tokens=sp_tokens, amplification=1000, trade_fee=ss_fee, unique_id="2-Pool"
    )

    sp4_tokens = {"USDC": 600000, "USDT": 340000, "DAI": 365000, "USDT2": 330000}
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens, amplification=1000, trade_fee=ss_fee, unique_id="4-Pool"
    )

    amm_list = [stablepool, stablepool4]
    intents = []
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    assert x['omnipool_deltas']['2-Pool'] >= 3000
    assert x['omnipool_deltas']['HDX'] <= -250000
    assert x['profit'] >= 250000
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents),
                                         copy.deepcopy(x['deltas']), copy.deepcopy(x['omnipool_deltas']),
                                         copy.deepcopy(x['amm_deltas']),"HDX")


def test_no_intent_arbitrage_xyk():
    # test where xyk price is different from Omnipool
    prices = {'HDX': 0.013, 'DOT': 9, 'vDOT': 13, '2-Pool': 1.01, '4-Pool': 0.99, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.08, 'DOT': 0.5, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.02}
    total_lrna = 65000000

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna},
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    xyk_fee = 0.003

    xyk = XykState(
        tokens={
            'HDX': liquidity['HDX'] * 0.5,
            'DOT': liquidity['DOT'] * 0.5
        },
        trade_fee=xyk_fee
    )

    amm_list = [xyk]
    intents = []
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    agent = Agent(enforce_holdings=False)
    xyk.swap(agent, "DOT", "HDX", sell_quantity = x['amm_deltas'][0][1])
    initial_state.swap(agent, "HDX", "DOT", sell_quantity = agent.get_holdings("DOT"))
    print(x)


def test_single_trade_settles():

    prices = {'HDX': 0.013, 'DOT': 9, 'vDOT': 13, '2-Pool': 1.01, '4-Pool': 1.01, 'LRNA': 1, 'USDT': 1, 'USDC': 1,
              'USDT2': 1, 'DAI': 1}
    weights = {'HDX': 0.08, 'DOT': 0.5, 'vDOT': 0.2, '2-Pool': 0.2, '4-Pool': 0.02}
    total_lrna = 65000000

    # first we test with just Omnipool
    lrna = {tkn: weights[tkn] * total_lrna for tkn in ['HDX', 'DOT', 'vDOT']}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    buy_pct = 1e-4
    price_premium = 0.1

    trade_pairs = [["DOT", "vDOT"], ["vDOT", "DOT"], ["LRNA", "DOT"]]
    # for tkn_sell, tkn_buy in trade_pairs:
    #     buy_amount_init = buy_pct * initial_state.liquidity[tkn_buy]
    #     buy_amount = buy_amount_init * (1 - price_premium)
    #     sell_amount = buy_amount_init * prices[tkn_buy] / prices[tkn_sell]
    #     if tkn_sell != "LRNA" and sell_amount > buy_pct * initial_state.liquidity[tkn_sell]:
    #         sell_amount = buy_pct * initial_state.liquidity[tkn_sell]
    #         buy_amount = sell_amount * prices[tkn_sell] / prices[tkn_buy] * (1 - price_premium)
    #     for partial in [True, False]:
    #         agent = Agent(holdings={tkn_sell: sell_amount})
    #         intents = [{'sell_quantity': sell_amount, 'buy_quantity': buy_amount, 'tkn_sell': tkn_sell,
    #                     'tkn_buy': tkn_buy, 'agent': agent, 'partial': partial}]
    #         x = find_solution_outer_approx(initial_state, intents)
    #         assert validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents),
    #                                              copy.deepcopy(x['deltas']), copy.deepcopy(x['omnipool_deltas']),
    #                                              copy.deepcopy(x['amm_deltas']), "HDX")
    #         if x['deltas'][0][0] != pytest.approx(-intents[0]['sell_quantity'], rel=1e-10):
    #             raise
    #         if x['deltas'][0][1] != pytest.approx(intents[0]['buy_quantity'], rel=1e-10):
    #             raise

    # next we add 2-pool and 4-pool

    lrna = {tkn: weights[tkn] * total_lrna for tkn in weights}
    liquidity = {tkn: lrna[tkn] / prices[tkn] for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    ss_fee = 0.0005

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="4-Pool"
    )

    amm_list = [stablepool, stablepool4]

    # After adding 2-pool and 4-pool, there are several types of tokens:
    # LRNA, DOT/vDOT, 2-pool/4-pool, USDT/USDC, USDT2/USDC2
    # we want to test within these pairs as well as between each pairwise
    # within pairs: DOT<>vDOT, 2-pool<>4-pool, USDT<>USDC, USDT2<>USDC2
    trade_pairs = [["DOT", "vDOT"], ["2-Pool", "4-Pool"], ["USDT", "USDC"], ["USDT2", "DAI"],
                   ["vDOT", "DOT"], ["4-Pool", "2-Pool"], ["USDC", "USDT"], ["DAI", "USDT2"]]
    # LRNA: LRNA<>DOT, LRNA<>2-pool, LRNA<>USDT, LRNA<>USDT2
    trade_pairs.extend([["LRNA", tkn] for tkn in ["DOT", "2-Pool", "USDT", "USDT2"]])
    # Remaining: DOT<>2-pool, DOT<>USDT, DOT<>USDT2, 2-pool<>USDT, 2-pool<>USDT2, USDT<>USDT2
    trade_pairs.extend([["DOT", "2-Pool"], ["DOT", "USDT"], ["DOT", "USDT2"], ["2-Pool", "USDT"],
                        ["2-Pool", "USDT2"], ["USDT", "USDT2"], ["2-Pool", "DOT"], ["USDT", "DOT"],
                        ["USDT2", "DOT"], ["USDT", "2-Pool"], ["USDT2", "2-Pool"], ["USDT2", "USDT"]])

    total_liquidity = {tkn: initial_state.liquidity[tkn] for tkn in initial_state.liquidity}
    for amm in amm_list:
        total_liquidity[amm.unique_id] = amm.shares
        for tkn in amm.liquidity:
            if tkn not in total_liquidity:
                total_liquidity[tkn] = 0
            total_liquidity[tkn] += amm.liquidity[tkn]

    for tkn_sell, tkn_buy in trade_pairs:
        buy_amount_init = buy_pct * total_liquidity[tkn_buy]
        buy_amount = buy_amount_init * (1 - price_premium)
        sell_amount = buy_amount_init * prices[tkn_buy] / prices[tkn_sell]
        if tkn_sell != "LRNA" and sell_amount > buy_pct * total_liquidity[tkn_sell]:
            sell_amount = buy_pct * total_liquidity[tkn_sell]
            buy_amount = sell_amount * prices[tkn_sell] / prices[tkn_buy] * (1 - price_premium)
        for partial in [True, False]:
            agent = Agent(holdings={tkn_sell: sell_amount})
            intents = [{'sell_quantity': sell_amount, 'buy_quantity': buy_amount, 'tkn_sell': tkn_sell,
                        'tkn_buy': tkn_buy, 'agent': agent, 'partial': partial}]
            x = find_solution_outer_approx(initial_state, intents, amm_list)
            assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list),
                                                 copy.deepcopy(intents), copy.deepcopy(x['deltas']),
                                                 copy.deepcopy(x['omnipool_deltas']), copy.deepcopy(x['amm_deltas']),
                                                 "HDX")
            if x['deltas'][0][0] != pytest.approx(-intents[0]['sell_quantity'], rel=1e-10):
                raise
            if x['deltas'][0][1] != pytest.approx(intents[0]['buy_quantity'], rel=1e-10):
                raise


def test_single_trade_does_not_settle():
    agents = [Agent(holdings={'DOT': 100, 'USDT': 0})]

    init_intents_partial = [  # selling DOT for $10
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1000), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True}
    ]
    init_intents_full = [  # selling DOT for $10
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1000), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False}
    ]
    init_intents_partial_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': True}
    ]
    init_intents_full_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': False}
    ]

    initial_state, amm_list = get_markets_minimal()

    intents = copy.deepcopy(init_intents_partial)
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full)
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_partial_lrna)
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full_lrna)
    x = find_solution_outer_approx(initial_state, intents, amm_list)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), intents, intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0


def test_single_trade_partially_settles():
    intent_omnipool = {  # selling DOT for $4.00, too big to fully execute against AMMs
        'sell_quantity': mpf(1000000), 'buy_quantity': mpf(4000000), 'tkn_sell': 'DOT', 'tkn_buy': '2-Pool',
        'agent': Agent(enforce_holdings=False), 'partial': True  # uses 2-Pool instead of USDT
    }
    intent_stableswap = {  # selling DOT for $4.00, too big to fully execute against AMMs
        'sell_quantity': mpf(1000000), 'buy_quantity': mpf(4000000), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT',
        'agent': Agent(enforce_holdings=False), 'partial': True
    }

    asset_fee = 0.0025
    lrna_fee = 0.0005
    ss_fee = 0.0005

    intents_lists = [copy.deepcopy(intent_omnipool), copy.deepcopy(intent_stableswap), copy.deepcopy(intent_stableswap), copy.deepcopy(intent_stableswap)]
    initial_state1, amm_list1 = get_markets_even(  # only Omnipool
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        ss_fee=ss_fee,
        include_2_pool=False,
        include_4_pool=False
    )
    initial_state2, amm_list2 = get_markets_even(  # Omnipool + 2-Pool
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        ss_fee=ss_fee,
        include_4_pool=False
    )
    initial_state3, amm_list3 = get_markets_even(  # Omnipool + 4-Pool
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        ss_fee=ss_fee,
        include_2_pool=False
    )
    initial_state4, amm_list4 = get_markets_even(  # Omnipool + 2-Pool + 4-Pool
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        ss_fee=ss_fee
    )
    initial_states = [initial_state1, initial_state2, initial_state3, initial_state4]
    amm_lists = [amm_list1, amm_list2, amm_list3, amm_list4]
    tolerances = [0.001, 0.02, 0.02, 0.02]
    for i in range(len(initial_states)):
        initial_state = initial_states[i].copy()
        amm_list = copy.deepcopy(amm_lists[i])
        intent = copy.deepcopy(intents_lists[i])
        spot_limit = intent['buy_quantity'] / intent['sell_quantity'] / ((1 - asset_fee) * (1 - lrna_fee) * (1 - ss_fee))
        # do the DOT sale alone
        state_sale = initial_state.copy()
        intents_sale = [copy.deepcopy(intent)]
        amms = copy.deepcopy(amm_list)

        x = find_solution_outer_approx(state_sale, intents_sale, amms)
        amm_deltas = x['amm_deltas']
        sale_deltas = x['deltas']
        omnipool_deltas = x['omnipool_deltas']

        valid, profit = validate_and_execute_solution(state_sale, amms, intents_sale, sale_deltas, omnipool_deltas, amm_deltas)

        assert valid
        assert abs(sale_deltas[0][0]) > 0
        if len(amms) > 0:
            dot_price_2_pool = state_sale.price("DOT", "2-Pool")
            twopool_price_usdt = amms[0].share_price("USDT")
            price = dot_price_2_pool * twopool_price_usdt
            init_dot_price_2_pool = initial_state.price("DOT", "2-Pool")
            init_twopool_price_usdt = amm_list[0].share_price("USDT")
            init_price = init_dot_price_2_pool * init_twopool_price_usdt
        else:
            price = state_sale.price("DOT", "2-Pool")
            init_price = initial_state.price("DOT", "2-Pool")
        if abs(price - spot_limit)/spot_limit > tolerances[i]:
            raise AssertionError("Price after sale is not within tolerance of expected price.")


def test_matching_trades_execute_more():
    dot_sale, usdt_buy = mpf(1_000_000), mpf(4_000_000)
    usdt_sale, dot_buy = mpf(6_000_000 / 2), mpf(1_000_000 / 2)
    lrna_sale, usdt_lrna_buy = mpf(7_500_000), mpf(7_500_000 * 0.9)
    agents = [
        Agent(holdings={'DOT': dot_sale}), Agent(holdings={'USDT': usdt_sale}), Agent(holdings={'LRNA': lrna_sale})
    ]

    intent1 = {  # selling DOT for $4.00, too big to fully execute against AMMs
        'sell_quantity': dot_sale, 'buy_quantity': usdt_buy, 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True
    }
    intent2 = {  # buying DOT for $6.00
        'sell_quantity': usdt_sale, 'buy_quantity': dot_buy, 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1], 'partial': True
    }
    intent1_lrna = {  # selling LRNA for $0.90
        'sell_quantity': lrna_sale, 'buy_quantity': usdt_lrna_buy, 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[2], 'partial': True
    }

    initial_state, amm_list = get_markets_even()

    # do the DOT sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1)]
    amms = copy.deepcopy(amm_list)
    x = find_solution_outer_approx(state_sale, intents_sale, amms)
    amm_deltas = x['amm_deltas']
    sale_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_sale, amms, intents_sale, sale_deltas, omnipool_deltas, amm_deltas)

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    amms = copy.deepcopy(amm_list)
    x = find_solution_outer_approx(state_buy, intents_buy, amm_list)
    amm_buy_deltas = x['amm_deltas']
    buy_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_buy, amms, intents_buy, buy_deltas, omnipool_deltas, amm_buy_deltas)

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    amms = copy.deepcopy(amm_list)
    x = find_solution_outer_approx(state_match, intents_match, amm_list)
    amm_match_deltas = x['amm_deltas']
    match_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_match, amms, intents_match, match_deltas, omnipool_deltas, amm_match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) > 0
    assert abs(buy_deltas[0][0]) > 0
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])

    # do the LRNA sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1_lrna)]
    amms = copy.deepcopy(amm_list)
    x = find_solution_outer_approx(state_sale, intents_sale, amm_list)
    amm_deltas = x['amm_deltas']
    sale_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_sale, amms, intents_sale, sale_deltas, omnipool_deltas, amm_deltas)

    # do both LRNA sale & DOT buy together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1_lrna), copy.deepcopy(intent2)]
    amms = copy.deepcopy(amm_list)
    x = find_solution_outer_approx(state_match, intents_match, amm_list)
    amm_deltas = x['amm_deltas']
    match_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_match, amms, intents_match, match_deltas, omnipool_deltas, amm_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) > 0
    assert abs(buy_deltas[0][0]) > 0
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])


def test_matching_trades_execute_more_full_execution():
    agents = [Agent(holdings={'DOT': 1000, 'LRNA': 7500}), Agent(holdings={'USDT': 7600})]

    intent1 = {  # selling DOT for $7.49
        'sell_quantity': mpf(1000), 'buy_quantity': mpf(7470), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False
    }

    intent2 = {  # buying DOT for $7.51
        'sell_quantity': mpf(7530), 'buy_quantity': mpf(1000), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1], 'partial': False
    }

    intent1_lrna = {  # selling DOT for $7.49
        'sell_quantity': mpf(7500), 'buy_quantity': mpf(7480), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False
    }

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    # do the DOT sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_sale, [], intents_sale, sale_deltas, omnipool_deltas, [], "HDX")

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_buy, intents_buy)
    buy_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_buy, [], intents_buy, buy_deltas, omnipool_deltas, [], "HDX")

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_match, [], intents_match, match_deltas, omnipool_deltas, [], "HDX")

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) == 0
    assert abs(buy_deltas[0][0]) == 0
    assert abs(match_deltas[0][0]) > 0
    assert abs(match_deltas[1][0]) > 0

    # do the LRNA sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1_lrna)]
    x = find_solution_outer_approx(state_sale, intents_sale)
    sale_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_sale, [], intents_sale, sale_deltas, omnipool_deltas, [], "HDX")

    # do both LRNA sale & DOT buy together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1_lrna), copy.deepcopy(intent2)]
    x = find_solution_outer_approx(state_match, intents_match)
    match_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    assert validate_and_execute_solution(state_match, [], intents_match, match_deltas, omnipool_deltas, [], "HDX")

    # check that matching trades caused more execution than executing either alone
    assert abs(sale_deltas[0][0]) == 0
    assert abs(buy_deltas[0][0]) == 0
    assert abs(match_deltas[0][0]) > 0
    assert abs(match_deltas[1][0]) > 0


###############
# Other tests #
###############

@given(st.floats(min_value=1e-7, max_value=0.01))
@settings(verbosity=Verbosity.verbose, print_blob=True)
def test_fuzz_single_trade_settles(size_factor: float):

    # AMM setup

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    # initial_state.last_fee = {tkn: 0.00 for tkn in lrna}
    # initial_state.last_lrna_fee = {tkn: 0.00 for tkn in lrna}

    ss_fee = 0.0005

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=ss_fee,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]
    # amm_list = [stablepool, stablepool4]
    # amm_list = [stablepool, stablepool_btc]

    router = OmnipoolRouter([initial_state] + amm_list)

    # trade setup
    tkn_sell, tkn_buy = "DOT", "WETH"
    # size_factor = 0.001  # pct of total liquidity that is being traded
    partial = True
    # get buy amount, sell amount from size_factor
    total_buy_liq = initial_state.liquidity[tkn_buy] if tkn_buy in initial_state.liquidity else 0
    total_sell_liq = initial_state.liquidity[tkn_sell] if tkn_sell in initial_state.liquidity else 0
    for amm in amm_list:
        if amm.unique_id == tkn_buy:
            total_buy_liq = max(amm.shares, total_buy_liq)
        elif amm.unique_id == tkn_sell:
            total_sell_liq = max(amm.shares, total_sell_liq)
        else:
            total_buy_liq += amm.liquidity[tkn_buy] if tkn_buy in amm.liquidity else 0
            total_sell_liq += amm.liquidity[tkn_sell] if tkn_sell in amm.liquidity else 0
    if tkn_buy == "LRNA":
        total_buy_liq = initial_state.lrna["DOT"]
    elif tkn_sell == "LRNA":
        total_sell_liq = initial_state.lrna["DOT"]
    max_buy_amount = total_buy_liq * size_factor
    max_sell_amount = total_sell_liq * size_factor
    agent = Agent(holdings={tkn_sell: max_sell_amount})
    # get sell_amount by simulating swap
    test_state, test_agent = router.simulate_swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=max_buy_amount)
    if test_state.fail == '':  # swap was succesful
        sell_amt = agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
        intent = {'sell_quantity': sell_amt / 0.999, 'buy_quantity': max_buy_amount, 'tkn_sell': tkn_sell,
                  'tkn_buy': tkn_buy, 'agent': Agent(holdings={tkn_sell: sell_amt / 0.999}), 'partial': partial}
    else:  # swap was unsuccesful
        test_state, test_agent = router.simulate_swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=max_sell_amount)
        buy_amt = test_agent.holdings[tkn_buy]
        intent = {'sell_quantity': max_sell_amount, 'buy_quantity': buy_amt * 0.999, 'tkn_sell': tkn_sell,
                  'tkn_buy': tkn_buy, 'agent': Agent(holdings={tkn_sell: max_sell_amount}), 'partial': partial}

    intents = [intent]
    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    # intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x['deltas'], x['profit'], x['omnipool_deltas'], x['amm_deltas']
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")

    assert valid


def test_convex():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state, [], intents, intent_deltas, omnipool_deltas,
                                         amm_deltas, "HDX")

    pprint(intent_deltas)


def test_with_lrna_intent():
    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'LRNA': 1000}),
        Agent(holdings={'DOT': 1000000})
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(100), 'tkn_sell': 'LRNA', 'tkn_buy': 'DOT', 'agent': agents[4], 'partial': True},  # buying DOT for $10
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False},  # selling DOT for $7
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    assert validate_and_execute_solution(initial_state, [], intents, intent_deltas, omnipool_deltas,
                                         amm_deltas, "HDX")

    pprint(intent_deltas)


def test_small_trade():  # this is to test that rounding errors don't screw up small trades
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'CRU': 5}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        {'sell_quantity': mpf(1.150), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']

    assert validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents), intent_deltas,
                                         omnipool_deltas, amm_deltas, "HDX")
    assert intent_deltas[0][0] == -intents[0]['sell_quantity']
    assert intent_deltas[0][1] == intents[0]['buy_quantity']
    assert intent_deltas[1][0] == 0
    assert intent_deltas[1][1] == 0


@given(st.floats(min_value=1e-7, max_value=1e-3))
@settings(verbosity=Verbosity.verbose, print_blob=True)
def test_inclusion_problem_small_trade_fuzz(trade_size_pct: float):
    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    buy_tkn = 'DOT'
    selL_tkn = '2-Pool'
    buy_amt = trade_size_pct * liquidity[buy_tkn]
    # buy_amt = mpf(.01)
    price = initial_state.price(buy_tkn, selL_tkn)
    sell_amt = buy_amt * price * 1.01
    # sell_amt = mpf(.05)
    agents = [Agent(holdings={selL_tkn: sell_amt})]

    intents = [
        {'sell_quantity': sell_amt, 'buy_quantity': buy_amt, 'tkn_sell': selL_tkn, 'tkn_buy': buy_tkn, 'agent': agents[0], 'partial': False},
    ]

    x = find_solution_outer_approx(initial_state, intents)
    assert x['deltas'][0][0] == -intents[0]['sell_quantity']


# # TODO fix test
# @given(st.floats(min_value=1e-10, max_value=1e-3))
# @settings(verbosity=Verbosity.verbose, print_blob=True)
# def test_small_trade_fuzz(trade_size_pct: float):  # this is to test that rounding errors don't screw up small trades
#
#     liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
#                  'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
#                  'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
#                  'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
#                  '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
#                  'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
#     lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
#             'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
#             'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
#             'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
#             '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
#             'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }
#
#     initial_state = OmnipoolState(
#         tokens={
#             tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
#         },
#         asset_fee=mpf(0.0025),
#         lrna_fee=mpf(0.0005)
#     )
#
#     buy_tkn = 'DOT'
#     sell_tkn = '2-Pool'
#     buy_amt = trade_size_pct * liquidity[buy_tkn]
#     price = initial_state.price(buy_tkn, sell_tkn)
#     sell_amt = buy_amt * price * 1.01
#     agents = [Agent(holdings={sell_tkn: sell_amt})]
#
#     intents = [
#         {'sell_quantity': sell_amt, 'buy_quantity': buy_amt, 'tkn_sell': sell_tkn, 'tkn_buy': buy_tkn, 'agent': agents[0], 'partial': True},
#     ]
#
#     x = find_solution_outer_approx(initial_state, intents)
#     intent_deltas = x['deltas']
#     omnipool_deltas = x['omnipool_deltas']
#     amm_deltas = x['amm_deltas']
#
#     valid, profit = validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents), intent_deltas,
#                                          omnipool_deltas, amm_deltas, "HDX")
#     assert valid
#     assert intent_deltas[0][0] == -intents[0]['sell_quantity']
#     assert intent_deltas[0][1] == pytest.approx(intents[0]['buy_quantity'], rel=1e-10)


# TODO fix test
# def test_solver_with_real_omnipool_one_full():
#     agents = [
#         Agent(holdings={'HDX': 100}),
#         Agent(holdings={'HDX': 100}),
#     ]
#
#     intents = [
#         {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0],
#          'partial': False},
#         {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
#          'partial': True},
#     ]
#
#     liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
#                  'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
#                  'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
#                  'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
#                  '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
#                  'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
#     lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
#             'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
#             'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
#             'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
#             '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
#             'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }
#
#     initial_state = OmnipoolState(
#         tokens={
#             tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
#         },
#         asset_fee=mpf(0.0025),
#         lrna_fee=mpf(0.0005)
#     )
#
#     full_intent_indicators = [1]
#
#     p = ICEProblem(initial_state, intents, min_partial = 0)
#     p.set_up_problem(I = full_intent_indicators)
#
#     # amm_deltas, sell_deltas, _, _, _, _ = _find_solution_unrounded(p)
#     x = _find_solution_unrounded(p)
#     amm_deltas = x[-1]
#     sell_deltas = x[1]
#     for i in p.full_intents:
#         if full_intent_indicators.pop(0) == 1:
#             sell_deltas.append(-i['sell_quantity'])
#
#     sell_deltas = round_solution(p.partial_intents + p.full_intents, sell_deltas)
#     intent_deltas = add_buy_deltas(p.partial_intents + p.full_intents, sell_deltas)
#
#     assert sell_deltas[0] == -100
#     assert sell_deltas[1] == -100
#     assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(p.partial_intents + p.full_intents), intent_deltas)
#
#     pprint(intent_deltas)


def test_full_solver():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']

    valid, profit = validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents), intent_deltas,
                                         omnipool_deltas, amm_deltas, "HDX")
    assert valid

    pprint(intent_deltas)


@given(st.lists(st.floats(min_value=1e-10, max_value=0.5), min_size=3, max_size=3),
       st.lists(st.floats(min_value=0.9, max_value=1.1), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=18), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=17), min_size=3, max_size=3),
        st.lists(st.booleans(), min_size=3, max_size=3)
       )
@settings(print_blob=True, verbosity=Verbosity.verbose, deadline=None, phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target))
def test_solver_random_intents(sell_ratios, price_ratios, sell_is, buy_is, partial_flags):

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    good_indices = [i for i in range(len(sell_is)) if sell_is[i]-1 != buy_is[i]]
    intents = []
    for i in good_indices:
        sell_tkn = initial_state.asset_list[sell_is[i]-1] if sell_is[i] > 0 else "LRNA"
        buy_tkn = initial_state.asset_list[buy_is[i]]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    predicted_profit = x['profit']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']

    valid, profit = validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert valid
    abs_error = predicted_profit - profit
    if profit > 0:
        pct_error = abs_error/profit
        # if not (pct_error < 0.01 or abs_error < 1):
        #     raise AssertionError(f"Profit: {profit}, Predicted Profit: {predicted_profit}, Abs Error: {abs_error}, Pct Error: {pct_error}")
        if not (abs_error < 100 or pct_error < 0.10):
            raise AssertionError(f"Profit: {profit}, Predicted Profit: {predicted_profit}, Abs Error: {abs_error}, Pct Error: {pct_error}")
    elif abs_error != 0:
        raise AssertionError(f"Profit: {profit}, Predicted Profit: {predicted_profit}, Abs Error: {abs_error}")

    pprint(intent_deltas)


def test_more_random_intents():
    r = 50
    random.seed(r)
    np.random.seed(r)

    intent_ct = 500
    min_sell_ratio, max_sell_ratio = 1e-10, 0.01
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    min_price_ratio, max_price_ratio = 0.99, 1.01
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(initial_state.asset_list + ['LRNA'], 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})
        # intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
        #                 'tkn_buy': buy_tkn, 'agent': agent, 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 100000, 'tkn_sell': '2-Pool', 'tkn_buy': 'HDX', 'agent': Agent(holdings={"2-Pool": 10}), 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 4000, 'tkn_sell': 'DOT', 'tkn_buy': '2-Pool', 'agent': Agent(holdings={'DOT': 10}), 'partial': False})

    # intent_deltas, predicted_profit, Z_Ls, Z_Us = find_solution_outer_approx(initial_state, intents)
    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    predicted_profit = x['profit']
    Z_Ls = x['Z_L']
    Z_Us = x['Z_U']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    valid, profit = validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents),
                                                  intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_more_random_intents_with_small():
    r = 50
    random.seed(r)
    np.random.seed(r)

    intent_ct = 500
    min_sell_ratio, max_sell_ratio = 1e-5, 0.01
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    for i in range(int(intent_ct/2)):
        sell_ratios[i] /= 1000
    min_price_ratio, max_price_ratio = 0.99, 1.01
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(initial_state.asset_list + ['LRNA'], 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})
        # intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
        #                 'tkn_buy': buy_tkn, 'agent': agent, 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 100000, 'tkn_sell': '2-Pool', 'tkn_buy': 'HDX', 'agent': Agent(holdings={"2-Pool": 10}), 'partial': True})
    # intents.append({'sell_quantity': 1000, 'buy_quantity': 4000, 'tkn_sell': 'DOT', 'tkn_buy': '2-Pool', 'agent': Agent(holdings={'DOT': 10}), 'partial': False})

    x = find_solution_outer_approx(initial_state, intents)
    intent_deltas = x['deltas']
    predicted_profit = x['profit']
    omnipool_deltas = x['omnipool_deltas']
    Z_Ls = x['Z_L']
    Z_Us = x['Z_U']

    valid, profit = validate_and_execute_solution(initial_state.copy(), [], copy.deepcopy(intents), intent_deltas,
                                                  omnipool_deltas, [],"HDX")
    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_get_leftover_bounds():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'USDT': 100}),
        Agent(holdings={'USDC': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(80.0), 'tkn_sell': 'USDT', 'tkn_buy': 'USDC', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(10.0), 'tkn_sell': 'USDC', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},
    ]

    # liquidity = {'HDX': mpf(140474254.46393022), 'CRU': mpf(337868.26827475097),
    #              '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855)}
    # lrna = {'HDX': mpf(24725.8021660851), 'CRU': mpf(4744.442135139952),
    #         '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808)}

    liquidity = {'HDX': mpf(140474254.46393022), 'CRU': mpf(337868.26827475097)}
    lrna = {'HDX': mpf(24725.8021660851), 'CRU': mpf(4744.442135139952)}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    # sp_tokens = {
    #     "USDT": 7600000,
    #     "USDC": 9200000
    # }
    # stablepool = StableSwapPoolState(
    #     tokens=sp_tokens,
    #     amplification=1000,
    #     trade_fee=0.0,
    #     unique_id="2-Pool"
    # )

    amm_list = []

    init_i, exec_indices = [], []
    p = ICEProblem(initial_state, intents, amm_list=amm_list, init_i=init_i, apply_min_partial=False)
    p.set_up_problem(I=[])

    A3, b3 = _get_leftover_bounds(p, allow_loss=False)
    # b - A x >= 0
    # n = 2, sigma = 0, u = 0, m = 1, r = 0. k = 9
    x_real = np.array([
        -0.01760107,  # y_0
        0.01613315,  # y_1
        100,  # x_0
        -1.149/(1-0.0025),  # x_1
        0.01760107,  # lrna_lambda_0
        0,  # lrna_lambda_1
        0,  # lambda_0
        1.149,  # lambda_1
        100   # d_0
    ])

    x_scaled = p.get_scaled_x(x_real)

    leftovers = -A3 @ x_scaled * np.concatenate([[p._scaling['LRNA']], p._S])
    assert len(leftovers) == len(b3)
    for i in range(len(leftovers)):
        assert (leftovers[i] - b3[i]) >= -1e-5


def test_full_solver_stableswap():
    agents = [
        Agent(holdings={'HDX': 10000}),
        Agent(holdings={'HDX': 10000}),
        Agent(holdings={'USDT': 100}),
        Agent(holdings={'USDC': 100}),
        Agent(holdings={'2-Pool': 1000}),
        Agent(holdings={'2-Pool': 100}),
        Agent(holdings={'2-Pool': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(90.0), 'tkn_sell': 'USDT', 'tkn_buy': 'USDC', 'agent': agents[2], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(10.0), 'tkn_sell': 'USDC', 'tkn_buy': 'DOT', 'agent': agents[3], 'partial': False},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(2), 'tkn_sell': 'HDX', 'tkn_buy': 'DOT',
        #  'agent': agents[0], 'partial': True},
        # {'sell_quantity': mpf(10000), 'buy_quantity': mpf(2), 'tkn_sell': 'HDX', 'tkn_buy': 'DOT',
        #  'agent': agents[1], 'partial': False},
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(500.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
         'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(1000), 'buy_quantity': mpf(500.0), 'tkn_sell': '2-Pool', 'tkn_buy': '4-Pool',
        #  'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(90.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
        #  'agent': agents[5], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': '2-Pool', 'tkn_buy': 'USDC',
        #  'agent': agents[4], 'partial': False},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    # liquidity = {'4-Pool': mpf(1392263.9295618401 + 15000), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
    #              'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
    #              'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
    #              'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
    #              '2-Pool': mpf(14626788.977583803 - 15000), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
    #              'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    # lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
    #         'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
    #         'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
    #         'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
    #         '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
    #         'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {'4-Pool': mpf(1392263.9295618401 + 15000), 'HDX': mpf(140474254.46393022), '2-Pool-btc': mpf(80.37640742108785),
                 'CRU': mpf(337868.26827475097), '2-Pool': mpf(14626788.977583803 - 15000)}
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), '2-Pool-btc': mpf(197326.54331209575),
            'CRU': mpf(4744.442135139952), '2-Pool': mpf(523282.70722423657)}

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), '2-Pool-btc': mpf(80.37640742108785),
                 'CRU': mpf(337868.26827475097), '2-Pool': mpf(14626788.977583803)}
    lrna = {'4-Pool': mpf(50483.454258911326/10), 'HDX': mpf(24725.8021660851), '2-Pool-btc': mpf(197326.54331209575),
            'CRU': mpf(4744.442135139952), '2-Pool': mpf(523282.70722423657)}

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), '2-Pool': mpf(14626788.977583803)}
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), '2-Pool': mpf(523282.70722423657)}


    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )

    sp_tokens = {
        "USDT": 7600000 - 81080,
        "USDC": 9200000 + 74500
    }
    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000 - 74600,
        "USDT": 340000 + 81160,
        # "DAI": 365000,
        # "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.01,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]
    amm_list = [stablepool, stablepool4]
    # amm_list = [stablepool]

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    # intent_deltas, omnipool_deltas, amm_deltas = x[0], x[4], x[5]
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']

    # valid, profit =  validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents),
                                                  intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    assert valid

    pprint(intent_deltas)


def test_more_random_intents_with_stableswap():
    r = 52
    random.seed(r)
    np.random.seed(r)

    intent_ct = 5
    min_sell_ratio, max_sell_ratio = 1e-10, 0.001
    sell_ratios = min_sell_ratio + (max_sell_ratio - min_sell_ratio) * np.random.rand(intent_ct)
    min_price_ratio, max_price_ratio = 0.99, 1.01
    min_price_ratio, max_price_ratio = 0.5, 0.9
    price_ratios = min_price_ratio + (max_price_ratio - min_price_ratio) * np.random.rand(intent_ct)
    partial_flags = np.random.choice([True, False], size=intent_ct)
    partial_flags = np.array([False] * intent_ct)

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]

    total_asset_list = [tkn for tkn in initial_state.asset_list] + ['LRNA']
    for amm in amm_list:
        if amm.unique_id not in total_asset_list:
            total_asset_list.append(amm.unique_id)
        for tkn in amm.asset_list:
            if tkn not in total_asset_list:
                total_asset_list.append(tkn)

    lrna_prices = {}
    for tkn in total_asset_list:
        if tkn == 'LRNA':
            lrna_prices[tkn] = 1
        elif tkn in initial_state.asset_list:
            lrna_prices[tkn] = initial_state.price(tkn, 'LRNA')
        else:  # tkn is in a stableswap pool which has shares in Omnipool
            prices = []  # we will take average of several prices if token is in multiple stableswap pools
            for amm in amm_list:
                if tkn in amm.asset_list:
                    spot = amm.withdraw_asset_spot(tkn)  # spot is tkn / shares
                    prices.append(spot * initial_state.price(amm.unique_id, 'LRNA'))
            lrna_prices[tkn] = np.mean(prices)

    # Generate n pairs of elements without replacement
    asset_pairs = [random.sample(total_asset_list, 2) for _ in range(intent_ct)]
    for i in range(len(asset_pairs)):  # can't buy LRNA
        if asset_pairs[i][1] == 'LRNA':
            asset_pairs[i][1] = asset_pairs[i][0]
            asset_pairs[i][0] = 'LRNA'

    total_liquidity = {tkn: initial_state.liquidity[tkn] for tkn in initial_state.asset_list}
    for amm in amm_list:
        for tkn in amm.asset_list:
            if tkn not in total_liquidity:
                total_liquidity[tkn] = 0
            total_liquidity[tkn] += amm.liquidity[tkn]

    intents = []
    for i in range(intent_ct):
        sell_tkn = asset_pairs[i][0]
        buy_tkn = asset_pairs[i][1]
        if sell_tkn == "LRNA":
            if buy_tkn in initial_state.asset_list:
                sell_quantity = sell_ratios[i] * lrna[buy_tkn]
            else:
                sell_quantity = sell_ratios[i] * initial_state.lrna_total / 20
        else:
            sell_quantity = sell_ratios[i] * total_liquidity[sell_tkn]
        buy_quantity = sell_quantity * lrna_prices[sell_tkn] / lrna_prices[buy_tkn] * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    # intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    intent_deltas = x['deltas']
    predicted_profit = x['profit']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    z_l_archive = x['Z_L']
    z_u_archive = x['Z_U']
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    # valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas)

    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)


def test_temp_milp():

    agent1 = Agent(holdings={'vASTR': 1400})
    agent2 = Agent(holdings={'DOT': 1465})
    agent3 = Agent(holdings={'iBTC': .00072869})
    agent4 = Agent(holdings={'ZTG': 2046})
    agent5 = Agent(holdings={'HDX': 0.0078998})
    agent6 = Agent(holdings={'HDX': 10000})
    intents = [
        {'sell_quantity': 1400, 'buy_quantity': 15000, 'tkn_sell': 'vASTR', 'tkn_buy': 'HDX', 'agent': agent1, 'partial': False},
        {'sell_quantity': .00072869, 'buy_quantity': 2520, 'tkn_sell': 'iBTC', 'tkn_buy': 'INTR', 'agent': agent3, 'partial': False},
        {'sell_quantity': 2046, 'buy_quantity': 55, 'tkn_sell': 'ZTG', 'tkn_buy': 'CRU', 'agent': agent4, 'partial': False},
        # {'sell_quantity': 10000, 'buy_quantity': 100, 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agent6, 'partial': True},
        {'sell_quantity': 1465, 'buy_quantity': 1139472, 'tkn_sell': 'DOT', 'tkn_buy': 'HDX', 'agent': agent2, 'partial': False},
        {'sell_quantity': 0.0078998, 'buy_quantity': 2286, 'tkn_sell': '2-pool-btc', 'tkn_buy': 'GLMR', 'agent': agent5, 'partial': False}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    liquidity = {tkn: float(liquidity[tkn]) for tkn in liquidity}
    lrna = {tkn: float(lrna[tkn]) for tkn in lrna}

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.00,
        lrna_fee=0.00
    )

    sp_tokens = {
        "USDT": 7600000,
        "USDC": 9200000
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool"
    )

    sp4_tokens = {
        "USDC": 600000,
        "USDT": 340000,
        "DAI": 365000,
        "USDT2": 330000
    }
    stablepool4 = StableSwapPoolState(
        tokens=sp4_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="4-Pool"
    )

    sp_btc_tokens = {
        "iBTC": 27.9,
        "wBTC": 48.6
    }
    stablepool_btc = StableSwapPoolState(
        tokens=sp_btc_tokens,
        amplification=1000,
        trade_fee=0.0005,
        unique_id="2-Pool-btc"
    )

    amm_list = [stablepool, stablepool4, stablepool_btc]

    x = find_solution_outer_approx(initial_state, intents, amm_list=amm_list)
    # intent_deltas, predicted_profit, omnipool_deltas, amm_deltas = x[0], x[1], x[4], x[5]
    # z_l_archive = x[2]
    # z_u_archive = x[3]
    intent_deltas = x['deltas']
    omnipool_deltas = x['omnipool_deltas']
    amm_deltas = x['amm_deltas']
    predicted_profit = x['profit']
    z_l_archive = x['Z_L']
    z_u_archive = x['Z_U']
    valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents),
                                                  intent_deltas, omnipool_deltas, amm_deltas, "HDX")
    # valid, profit = validate_and_execute_solution(initial_state.copy(), copy.deepcopy(amm_list), copy.deepcopy(intents), intent_deltas, omnipool_deltas, amm_deltas)

    assert valid
    abs_error = predicted_profit - profit
    # if profit > 0:
    #     pct_error = abs_error / profit
    #     assert pct_error < 0.01 or abs_error < 1
    #     assert abs(pct_error) < 0.05 or abs(abs_error) < 100
    # else:
    #     assert abs_error == 0
    # assert abs_error < 100

    pprint(intent_deltas)