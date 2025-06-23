from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv, json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_asset_info_by_ids

tkn = 'DOT'
oracle_data_filename = 'DOTUSD_oracle_prices.csv'
tkn_id = 5
file_path = os.path.join(project_root, 'hydradx', 'apps', 'fees', 'data', oracle_data_filename)
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
st.text(f"Min block: {min_block_no}, Max block: {max_block_no}")

acct = "0x7279fcf9694718e1234d102825dccaf332f0ea36edf1ca7c0358c4b68260d24b"
acct_swaps_filename = f"acct_swaps_{tkn_id}_{acct}.json"
data_path = os.path.join(project_root, 'hydradx', 'apps', 'fees', 'data')
with open(os.path.join(data_path, acct_swaps_filename), 'r') as f:
    arb_data = json.load(f)

tkn_asset_id = tkn_id
denom_asset_id = 10
assert denom_asset_id != tkn_asset_id, "Asset IDs must be different"
min_block = min_block_no
max_block = max_block_no

asset_info = get_asset_info_by_ids([tkn_asset_id, 102, 1, denom_asset_id])

restricted_data = [x for x in data if min_block <= x['block_number'] <= max_block]
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
interp_oracle_prices = []
last_block_no = min_block
last_price = sorted_avg_oracle_prices[0][1]
for (block_no, price) in sorted_avg_oracle_prices:
    interp_oracle_prices.extend([last_price] * (block_no - last_block_no))
    last_block_no = block_no
    last_price = price
interp_oracle_prices.extend([last_price] * (max_block - last_block_no + 1))
assert len(interp_oracle_prices) == (max_block - min_block + 1)

# get oracle values in blocks when arbs happened
buy_data = []
sell_data = []
for arb in arb_data:
    if min_block <= arb['block_number'] <= max_block:
        if arb['output_asset_id'] == tkn_id:
            buy_data.append(arb)
            buy_data[-1]['oracle_price'] = interp_oracle_prices[arb['block_number'] - min_block]
            buy_data[-1]['exec_price'] = arb['input_amount'] / arb['output_amount']
        elif arb['input_asset_id'] == tkn_id:
            sell_data.append(arb)
            sell_data[-1]['oracle_price'] = interp_oracle_prices[arb['block_number'] - min_block]
            sell_data[-1]['exec_price'] = arb['output_amount'] / arb['input_amount']
            sell_data[-1]['pct_diff'] = 1 - sell_data[-1]['exec_price'] / sell_data[-1]['oracle_price']
            sell_data[-1]['fees'] = 0.0015 + sell_data[-1]['hub_fee'] / sell_data[-1]['hub_amount']
        else:
            raise ValueError("Invalid asset ID")

fig, ax = plt.subplots()
ax.plot(range(len(sell_data)), [x['oracle_price'] for x in sell_data], label='oracle price')
ax.plot(range(len(sell_data)), [x['exec_price'] for x in sell_data], label='execution price')
ax.set_title("Oracle price vs arbitrager execution price")
ax.set_xlabel("Block number")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(range(len(sell_data)), [x['pct_diff'] for x in sell_data], label='pct diff')
ax.set_title("Arbitrager execution price vs oracle price pct diff")
ax.set_xlabel("Block number")
ax.set_ylabel("Pct diff")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist([x['pct_diff'] for x in sell_data], bins=50)
ax.set_title("Arbitrager execution price vs oracle price pct diff")
ax.set_ylabel("Frequency")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist([x['pct_diff'] - x['fees'] for x in sell_data], bins=50)
ax.set_title("Arbitrager execution price vs oracle price pct diff, minus fees")
ax.set_ylabel("Frequency")
st.pyplot(fig)
