from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_asset_info_by_ids, get_omnipool_data_by_asset, get_omnipool_liquidity

asset_ids = [0, 12]
min_block_id = 6000000
max_block_id = 6000100

liquidity, hub_liquidity = get_omnipool_liquidity(min_block_id, max_block_id, asset_ids)

fig1, ax1 = plt.subplots()
ax1.plot(liquidity[0], label='HDX balances')
ax1.set_title("HDX balances")
# Format y-axis in millions
import matplotlib.ticker as ticker
def millions_formatter(x, pos):
    return f'{x/1000000:.2f}'  # Divide by 1 million and show 2 decimal places

ax1.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
ax1.set_ylabel("HDX balances, in millions")
ax1.set_xlabel("blocks")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(hub_liquidity[0], label='LRNA balances')
ax2.set_title("LRNA balances (HDX sub-pool)")
ax2.set_xlabel("blocks")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(liquidity[12], label='USDT balances')
ax3.set_title("USDT balances")
ax3.set_xlabel("blocks")
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
ax4.plot(hub_liquidity[12], label='LRNA balances')
ax4.set_title("LRNA balances (USDT sub-pool)")
ax4.set_xlabel("blocks")
st.pyplot(fig4)