from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv, json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_stableswap_liquidity_events, get_omnipool_asset_data, get_asset_info_by_ids

oracle_data_filename = 'DOTUSD_oracle_prices.csv'
file_path = os.path.join(project_root, 'hydradx', 'apps', 'fees', oracle_data_filename)
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

omnipool_data_file_prefix = 'omnipool_spot_prices'
stableswap_data_file_prefix = 'stableswap_exec_prices'

data_path = os.path.join(project_root, 'hydradx', 'apps', 'fees', 'data')
omnipool_data_files = [f for f in os.listdir(data_path) if f.startswith(omnipool_data_file_prefix)]
omnipool_data = {}
for file in omnipool_data_files:
    with open(os.path.join(data_path, file), 'r') as f:
        omnipool_data.update(json.load(f))

price_of_tkn = [0] * (max_block_no - min_block_no + 1)
last_price = 0
i = 0
while last_price == 0:
    str_block_no = str(min_block_no + i)
    if str_block_no in omnipool_data:
        price_of_tkn[i] = omnipool_data[str_block_no]
        for j in range(i):
            price_of_tkn[j] = price_of_tkn[i]
        last_price = omnipool_data[str_block_no]
    i += 1
assert last_price > 0, "last_price must be > 0"
for j in range(i, len(price_of_tkn)):
    str_block_no = str(min_block_no + j)
    if str_block_no in omnipool_data:
        price_of_tkn[j] = omnipool_data[str_block_no]
        last_price = omnipool_data[str_block_no]
    else:
        price_of_tkn[j] = last_price

stableswap_data_files = [f for f in os.listdir(data_path) if f.startswith(stableswap_data_file_prefix)]
stableswap_data = {}
for file in stableswap_data_files:
    with open(os.path.join(data_path, file), 'r') as f:
        stableswap_data.update(json.load(f))

ss_exec_prices = [0] * (max_block_no - min_block_no + 1)
last_price = 0
i = 0
while last_price == 0:
    str_block_no = str(min_block_no + i)
    if str_block_no in stableswap_data:
        ss_exec_prices[i] = stableswap_data[str_block_no]
        for j in range(i):
            ss_exec_prices[j] = ss_exec_prices[i]
        last_price = stableswap_data[str_block_no]
    i += 1
assert last_price > 0, "last_price must be > 0"
for j in range(i, len(ss_exec_prices)):
    str_block_no = str(min_block_no + j)
    if str_block_no in stableswap_data:
        ss_exec_prices[j] = stableswap_data[str_block_no]
        last_price = stableswap_data[str_block_no]
    else:
        ss_exec_prices[j] = last_price

# HDX: 0
# H20: 1
# DOT: 5
# USDT: 10
# 2-Pool: 102

tkn_asset_id = 5
denom_asset_id = 10
assert denom_asset_id != tkn_asset_id, "Asset IDs must be different"
min_block = min_block_no
max_block = max_block_no

asset_info = get_asset_info_by_ids([tkn_asset_id, 102, 1, denom_asset_id])

# price of TKN denominated in USDT
price_of_tkn_in_usdt = [price_of_tkn[i] * ss_exec_prices[i] for i in range(max_block - min_block + 1)]

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


# plot
fig, ax = plt.subplots()
ax.plot(range(min_block, max_block + 1), interp_oracle_prices, label='oracle price')
ax.plot(range(min_block, max_block + 1), price_of_tkn_in_usdt, label='spot price')
ax.set_title("Oracle price vs spot price")
ax.set_xlabel("Block number")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

spot_prices = np.array(price_of_tkn_in_usdt)
oracle_prices = np.array(interp_oracle_prices)
correlations = []
max_tau = 200
for tau in range(1, max_tau + 1):
    n = len(spot_prices) - tau
    D = np.log(oracle_prices[:n] / spot_prices[:n])
    R = np.log(spot_prices[tau:] / spot_prices[:n])
    corr = np.corrcoef(D, R)[0, 1]
    correlations.append(corr)

# plot
fig, ax = plt.subplots()
ax.plot(range(1, max_tau + 1), correlations)
ax.set_title('Cross-Correlation: Oracle Returns → Future AMM Returns')
ax.set_xlabel('Lag (time steps)')
ax.set_ylabel('Correlation coefficient')
ax.grid(True)
st.pyplot(fig)