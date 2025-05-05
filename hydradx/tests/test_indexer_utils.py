import pytest
from hydradx.model.amm.omnipool_amm import OmnipoolState

from hydradx.model.indexer_utils import get_latest_stableswap_data, get_stablepool_ids, get_omnipool_liquidity, \
    get_omnipool_asset_data, get_current_block_height, get_asset_info, get_current_omnipool, get_omnipool_router


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
    info = get_asset_info()
    current_block = get_current_block_height()
    omnipool_assets = get_omnipool_asset_data(max_block_id=current_block, min_block_id=current_block - 10000)
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
    router = get_omnipool_router()
    assert router is not None
