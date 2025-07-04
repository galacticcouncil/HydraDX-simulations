import pytest, copy, numpy as np
from hypothesis import given, strategies as st, assume

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.solver.amm_constraints import XykConstraints, StableswapConstraints, AmmIndexObject
from hydradx.model.amm.xyk_amm import ConstantProductPoolState


def test_get_asset_indicator_matrix():
    global_asset_list = ['S', 'XYK-1', 'B', 'C', "SS-1"]

    amm = ConstantProductPoolState(tokens={'A': 1000, 'B': 2000}, unique_id='XYK-1')
    constraints = XykConstraints(amm)
    share_mat, asset_mat = constraints.get_indicator_matrices(global_asset_list)
    expected_share_matrix = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    expected_asset_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert share_mat.tolist() == expected_share_matrix
    assert asset_mat.tolist() == expected_asset_matrix

    amm = StableSwapPoolState(tokens = {'A': 1000, 'B': 2000}, amplification=100, unique_id="SS-1")
    constraints = StableswapConstraints(amm)
    share_mat, asset_mat = constraints.get_indicator_matrices(global_asset_list)
    expected_share_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ]
    expected_asset_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert share_mat.tolist() == expected_share_matrix
    assert asset_mat.tolist() == expected_asset_matrix


def check_all_cone_feasibility(s, cones, cone_sizes, tol=2e-5):
    from hydradx.model.solver.omnix_solver import check_cone_feasibility
    res = check_cone_feasibility(s, cones, cone_sizes, tol)
    for _, _, cone_feas in res:
        if not cone_feas:
            return False
    return True


def test_get_amm_limits_A():
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = ConstantProductPoolState(tokens=xyk_liquidity)
    amm_directions = []
    last_amm_deltas = []

    for amm, trading_pairs in [[stablepool4, [list(range(ss_tkn_ct + 1)), [0, 2], [1, 3]]], [xyk, [[1, 2]]]]:
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        amm_i = amm_constraints.amm_i
        for trading_is in trading_pairs:
            trading_tkns = [all_trading_tkns[i] for i in trading_is]
            A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(amm_directions, last_amm_deltas,
                                                                                     trading_tkns)

            assert A_limits.shape[1] == amm_constraints.k
            # assert A_limits.shape[0] == 2 * len(trading_tkns) + 2 * (tkn_ct + 1 - len(trading_tkns))
            # in this case, A_limits should be enforcing that Li >= 0 for the restricted trading tokens
            # it should be enforcing that Xi == 0 and Li == 0 for the other tokens
            x = np.zeros(amm_constraints.k)
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


@given(st.lists(st.booleans(), min_size=5, max_size=5),
       st.booleans(),
       st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=5, unique=True))
def test_get_amm_limits_A_directions(ss_dirs, xyk_dir, trading_indices):
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = ConstantProductPoolState(tokens=xyk_liquidity)
    xyk_directions = ['none', 'buy' if xyk_dir else 'sell', 'sell' if xyk_dir else 'buy']
    ss_directions = ['buy' if dir else 'sell' for dir in ss_dirs]
    last_amm_deltas = []
    for amm, directions, trading_pairs in [[stablepool4, ss_directions, [list(range(ss_tkn_ct + 1)), trading_indices]],
                                           [xyk, xyk_directions, [[1, 2]]]]:
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        amm_i = amm_constraints.amm_i
        for trading_is in trading_pairs:
            trading_tkns = [all_trading_tkns[i] for i in trading_is]
            A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(directions, last_amm_deltas,
                                                                                      trading_tkns)
            assert A_limits.shape[1] == amm_constraints.k
            # in this case, A_limits should be enforcing that Li >= 0 for the restricted trading tokens
            # it should be enforcing that Xi == 0 and Li == 0 for the other tokens
            x = np.zeros(amm_constraints.k)
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
