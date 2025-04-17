from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_latest_stableswap_data

pool_id = 102  # 2-pool, for now
pool_data = get_latest_stableswap_data(pool_id)
print(pool_data)