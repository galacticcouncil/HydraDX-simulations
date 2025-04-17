from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_latest_stableswap_data
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.apps.gigadot_modeling.display_utils import display_ss

pool_id = 102  # 2-pool, for now
pool_data = get_latest_stableswap_data(pool_id)
print(pool_data)

tokens = {'USDT': pool_data['liquidity'][10], 'USDC': pool_data['liquidity'][22]}
amp = pool_data['initialAmplification']
pool = StableSwapPoolState(tokens, amp, trade_fee = pool_data['fee'])
prices = {"USDT": 1, "USDC": 1}  # Dummy prices for display
fig = display_ss(pool.liquidity, prices, "")
st.sidebar.pyplot(fig)

# self,
# tokens: dict,
# amplification: float,
# precision: float = 0.0001,
# trade_fee: float = 0,
# unique_id: str = '',
# spot_price_precision: float = 1e-07,
# shares: float = 0,
# peg: float or list = None,
# peg_target: float or list = None,
# max_peg_update: float = float('inf')