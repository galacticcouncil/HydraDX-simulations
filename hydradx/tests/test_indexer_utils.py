import copy
import math

import pytest
from hypothesis import given, strategies as st, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.indexer_utils import get_latest_stableswap_data

def test_get_latest_stableswap_data():
    """
    Test the get_latest_stableswap_data function.
    """
    pool_id = 102
    pool_data = get_latest_stableswap_data(pool_id)
    pool_id = 103
    with pytest.raises(IndexError):
        get_latest_stableswap_data(pool_id)
