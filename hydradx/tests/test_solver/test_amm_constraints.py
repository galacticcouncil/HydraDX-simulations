import pytest
from hypothesis import given, strategies as st, assume

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.solver.amm_constraints import XykConstraints, StableswapConstraints
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
