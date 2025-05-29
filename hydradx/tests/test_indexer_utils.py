import copy
import math

import pytest
from hypothesis import given, strategies as st, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.indexer_utils import get_latest_stableswap_data, get_stablepool_ids, get_fee_history, get_executed_trades, get_stableswap_liquidity_events


def test_get_latest_stableswap_data():
    """
    Test the get_latest_stableswap_data function.
    """
    pool_id = 102
    pool_data = get_latest_stableswap_data(pool_id)
    pool_id = 999
    with pytest.raises(IndexError):
        get_latest_stableswap_data(pool_id)


def test_get_stablepool_ids():
    """
    Test the get_stablepool_ids function.
    """
    pool_ids = get_stablepool_ids()
    assert 102 in pool_ids
    assert 690 in pool_ids


def test_get_fee_history():
    asset_id = 0
    min_block_id = 7400000
    max_block_id = 7400100
    data = get_fee_history(asset_id, min_block_id, max_block_id)
    print("done")


def test_get_executed_trades():
    asset_ids = [5, 10]
    min_block_id = 7400000
    max_block_id = 7401000
    data = get_executed_trades(asset_ids, min_block_id, max_block_id)
    print("done")


def test_get_stableswap_liquidity_events():
    pool_id = 102
    min_block_id = 7400000
    max_block_id = 7401000
    data = get_stableswap_liquidity_events(pool_id, min_block_id, max_block_id)
    print("done")


# def test_download_stableswap_exec_prices():
#     from hydradx.model.indexer_utils import download_stableswap_exec_prices
#     pool_id = 102
#     tkn_id = 10
#     min_block_id = 7111661
#     # max_block_id = 7654990  # this is desirable but takes too long for integration tests
#     max_block_id = min_block_id + 1000
#     print(os.getcwd())
#
#     path = "hydradx/apps/fees/data/"
#     download_stableswap_exec_prices(pool_id, tkn_id, min_block_id, max_block_id, path)
#     print("done")
#
#
# def test_download_omnipool_spot_prices():
#     from hydradx.model.indexer_utils import download_omnipool_spot_prices
#     denom_id = 102
#     tkn_id = 1000765
#     min_block_id = 7611662
#     # max_block_id = 7654990  # this is desirable but takes too long for integration tests
#     max_block_id = min_block_id + 1000
#     step_size = 50000
#     print(os.getcwd())
#
#     path = "hydradx/apps/fees/data/"
#     temp_max_block_id = min(min_block_id + step_size, max_block_id)
#     while min_block_id <= max_block_id:
#         download_omnipool_spot_prices(tkn_id, denom_id, min_block_id, temp_max_block_id, path)
#         min_block_id = temp_max_block_id + 1
#         temp_max_block_id = min(temp_max_block_id + step_size, max_block_id)
#     print("done")


def test_get_omnipool_swap_fees():
    from hydradx.model.indexer_utils import get_omnipool_swap_fees
    tkn_id = 5
    min_block_id = 7000000
    max_block_id = 7001000

    asset_fee_data, hub_fee_data = get_omnipool_swap_fees(tkn_id, min_block_id, max_block_id)
    print("done")


