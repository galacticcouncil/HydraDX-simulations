from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_latest_stableswap_data
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.apps.gigadot_modeling.display_utils import display_ss
from hydradx.model.amm.agents import Agent

pool_id = 102  # 2-pool, for now
pool_data = get_latest_stableswap_data(pool_id)
print(pool_data)

tokens = {'USDT': pool_data['liquidity'][10], 'USDC': pool_data['liquidity'][22]}
amp = pool_data['initialAmplification']
pool = StableSwapPoolState(tokens, amp, trade_fee = pool_data['fee'])
prices = {"USDT": 1, "USDC": 1}  # Dummy prices for display
fig = display_ss(pool.liquidity, prices, "")
st.sidebar.pyplot(fig)

usdt_buy_size = st.number_input(
    "USDT buy size",
    min_value=1.0, max_value=1000000.0, value=10000.0, step=1.0, key="usdt_buy_size", format="%.2f"
)

agent = Agent(enforce_holdings=False)
pool.swap(agent, tkn_buy='USDT', tkn_sell='USDC', buy_quantity=usdt_buy_size)
sell_amt = -agent.get_holdings('USDC')
st.text(f"sell amt: {sell_amt}")
