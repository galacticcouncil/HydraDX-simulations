from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_fee_pcts, get_fee_history

c1, c2, c3 = st.columns(3)

with c1:
    asset_id = st.number_input(
        "Asset ID",
        min_value=0, max_value=99999999, value=0, step=1, key="asset_id", format="%d",
    )

with c2:
    min_block = st.number_input(
        "Minimum block ID",
        min_value=0, max_value=99999999, value=7480000, step=1, key="min_block", format="%d"
    )

with c3:
    max_block = st.number_input(
        "Maximum block ID",
        min_value=0, max_value=99999999, value=0, step=1, key="max_block", format="%d",
        help="set to 0 to get latest data available"
    )

data = get_fee_history(asset_id, min_block, max_block if max_block > 0 else None)
asset_fees_history = get_fee_pcts(data, asset_id)
# list of pairs in [block_id, fee] format
# need to convert to [unique_block_id, average_fee]
fees = {}
for i in range(len(asset_fees_history)):
    block_id = asset_fees_history[i][0]
    fee = asset_fees_history[i][1]
    if block_id not in fees:
        fees[block_id] = []
    fees[block_id].append(fee)
# calculate average fee for each block_id
avg_fees = {}
for block_id in fees:
    avg_fees[block_id] = sum(fees[block_id]) / len(fees[block_id])
# sort by block_id
sorted_avg_fees = sorted(avg_fees.items(), key=lambda x: x[0])
# separate block_id and fee into two lists
block_ids = [x[0] for x in sorted_avg_fees]
fees = [x[1] for x in sorted_avg_fees]
# plot the data
fig, ax = plt.subplots()
ax.scatter(block_ids, fees, label='asset fees', marker='x', s=1)
ax.set_title("asset fees history")
ax.set_xlabel("Block ID")
ax.set_ylabel("asset fees")
ax.legend()
st.pyplot(fig)
