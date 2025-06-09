import copy
import math
from pathlib import Path

import pytest
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hypothesis import given, strategies as st, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.indexer_utils import get_latest_stableswap_data, get_stablepool_ids, get_omnipool_liquidity, \
    get_omnipool_asset_data, get_current_block_height, get_asset_info_by_ids, get_current_omnipool, \
    get_current_omnipool_router, get_fee_history, get_executed_trades, get_stableswap_liquidity_events

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


def test_get_omnipool_data():
    info = get_asset_info_by_ids()
    current_block = get_current_block_height()
    omnipool_assets = get_omnipool_asset_data(max_block_id=current_block, min_block_id=current_block - 1000)
    ids = list(set([tkn['assetId'] for tkn in omnipool_assets]))
    # asset_data = get_asset_info_by_ids(ids)
    omnipool_liquidity = get_omnipool_liquidity(
        max_block_id=current_block, min_block_id=current_block - 100, asset_ids=ids
    )
    assert omnipool_liquidity is not None


def test_get_omnipool_state():
    omnipool = get_current_omnipool()
    assert isinstance(omnipool, OmnipoolState)


def test_get_omnipool_router():
    router = get_current_omnipool_router()
    assert router is not None


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


def test_download_stableswap_exec_prices():
    from hydradx.model.indexer_utils import download_stableswap_exec_prices
    pool_id = 102
    tkn_id = 10
    min_block_id = 6400000
    max_block_id = min_block_id + 1000

    base_dir = Path(__file__).resolve().parents[1]
    path = str(base_dir / "apps" / "fees" / "data") + "/"

    download_stableswap_exec_prices(pool_id, tkn_id, min_block_id, max_block_id, path)
    print("done")


def test_download_omnipool_spot_prices():
    from hydradx.model.indexer_utils import download_omnipool_spot_prices
    denom_id = 102
    # tkn_id = 1000765  # tBTC
    tkn_id = 1000624  # AAVE
    min_block_id = 7111661
    max_block_id = min_block_id + 1000
    step_size = 50000

    base_dir = Path(__file__).resolve().parents[1]
    path = str(base_dir / "apps" / "fees" / "data") + "/"

    temp_max_block_id = min(min_block_id + step_size, max_block_id)
    while min_block_id <= max_block_id:
        download_omnipool_spot_prices(tkn_id, denom_id, min_block_id, temp_max_block_id, path)
        min_block_id = temp_max_block_id + 1
        temp_max_block_id = min(temp_max_block_id + step_size, max_block_id)
    print("done")


def test_get_omnipool_swap_fees():
    from hydradx.model.indexer_utils import get_omnipool_swap_fees
    tkn_id = 5
    min_block_id = 7000000
    max_block_id = 7001000

    asset_fee_data, hub_fee_data = get_omnipool_swap_fees(tkn_id, min_block_id, max_block_id)
    print("done")


def test_get_asset_info_by_ids():
    from hydradx.model.indexer_utils import get_asset_info_by_ids
    ids = get_asset_info_by_ids()
    assert isinstance(ids, dict)


# def test_download_acct_trades():
#     from hydradx.model.indexer_utils import download_acct_trades
#
#     tkn_id = 5
#     path = "hydradx/apps/fees/data/"
#     acct = "0x7279fcf9694718e1234d102825dccaf332f0ea36edf1ca7c0358c4b68260d24b"
#     download_acct_trades(tkn_id, acct, path)
#     print("done")
