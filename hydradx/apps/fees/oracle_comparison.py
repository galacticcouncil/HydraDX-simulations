from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_executed_trades

filename = 'DOTUSD_oracle_prices.csv'
file_path = os.path.join(project_root, 'hydradx', 'apps', 'fees', filename)
data = []
with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(
            {
                'block_number': int(row['block_number']),
                'id': int(row['id']),
                'oracle_price': int(row['oracle_price']) / 1e8,
                'tkn_pair': row['tkn_pair'],
                'timestamp': int(row['time'])
            }
        )
        if len(data) > 1 and data[-1]['block_number'] > data[-2]['block_number']:
            raise ValueError("Data is not sorted by block number, descending")

max_block_no = data[0]['block_number']
min_block_no = data[-1]['block_number']

# HDX: 0
# H20: 1
# DOT: 5
# USDT: 10
# 2-Pool: 102

c1, c2 = st.columns(2)
with c1:
    min_block = st.number_input(label='min block number', value=7030000, min_value=min_block_no, max_value=max_block_no)
    denom_asset_id = st.number_input(label='Denomination asset id', value=10, min_value=0)
with c2:
    max_block = st.number_input(label='max block number', value=7040000, min_value=min_block_no, max_value=max_block_no)
    tkn_asset_id = st.number_input(label='Token asset id', value=5, min_value=0)
assert denom_asset_id != tkn_asset_id, "Asset IDs must be different"
all_trade_data = get_executed_trades([tkn_asset_id, denom_asset_id], min_block, max_block)
trade_data = []
for x in all_trade_data:
    try:
        involved_ids = set([int(y) for y in x['all_involved_asset_ids']])
        if involved_ids.issubset([denom_asset_id, tkn_asset_id, 102, 1, 22]):
            trade_data.append(x)
    except ValueError:
        pass
for x in trade_data:
    x['price'] = x['output_amount'] / x['input_amount'] if x['output_asset_id'] == denom_asset_id else x['input_amount'] / x['output_amount']

restricted_data = [x for x in data if x['block_number'] >= min_block and x['block_number'] <= max_block]
# calculate oracle price in each block present in data
oracle_prices = {}
for x in restricted_data:
    if x['block_number'] not in oracle_prices:
        oracle_prices[x['block_number']] = []
    oracle_prices[x['block_number']].append(x['oracle_price'])
# calculate average oracle price for each block
avg_oracle_prices = {}
for block_no in oracle_prices:
    avg_oracle_prices[block_no] = sum(oracle_prices[block_no]) / len(oracle_prices[block_no])
# sort by block_number
sorted_avg_oracle_prices = sorted(avg_oracle_prices.items(), key=lambda x: x[0])

# calculate execution price in each block present in data
execution_prices = {}
for x in trade_data:
    if x['block_number'] not in execution_prices:
        execution_prices[x['block_number']] = []
    execution_prices[x['block_number']].append(x['price'])
# calculate average execution price for each block
avg_execution_prices = {}
for block_no in execution_prices:
    avg_execution_prices[block_no] = sum(execution_prices[block_no]) / len(execution_prices[block_no])
# sort by block_number
sorted_avg_execution_prices = sorted(avg_execution_prices.items(), key=lambda x: x[0])

# plot
fig, ax = plt.subplots()
ax.step([x[0] for x in sorted_avg_oracle_prices], [x[1] for x in sorted_avg_oracle_prices], label='oracle price')
ax.step([x[0] for x in sorted_avg_execution_prices], [x[1] for x in sorted_avg_execution_prices], label='execution price')
ax.set_title("Oracle price vs execution price")
ax.set_xlabel("Block number")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
