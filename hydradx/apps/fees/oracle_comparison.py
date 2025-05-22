from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_stableswap_liquidity_events, get_omnipool_asset_data, get_asset_info_by_ids

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
st.text(f"Min block: {min_block_no}, Max block: {max_block_no}")

# HDX: 0
# H20: 1
# DOT: 5
# USDT: 10
# 2-Pool: 102

c1, c2, c3 = st.columns(3)
with c1:
    tkn_asset_id = st.number_input(label='Token asset id', value=5, min_value=0)
with c2:
    min_block = st.number_input(label='min block number', value=7030000, min_value=min_block_no, max_value=max_block_no)
with c3:
    max_block = st.number_input(label='max block number', value=7035000, min_value=min_block_no, max_value=max_block_no)
denom_asset_id = 10
assert denom_asset_id != tkn_asset_id, "Asset IDs must be different"

asset_info = get_asset_info_by_ids([tkn_asset_id, 102, 1, 10])

stableswap_execution_data = get_stableswap_liquidity_events(102, min_block, max_block)
usdt_execution_data = [
    x for x in stableswap_execution_data if (
            len(x['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes']) == 1
            and x['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes'][0]['assetId'] == '10'
            and int(x['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes'][0]['amount']) > 0
    )
]

# calculate oracle price in each block present in data
usdt_exec_prices = {}
for x in usdt_execution_data:
    if x['paraBlockHeight'] not in usdt_exec_prices:
        usdt_exec_prices[x['paraBlockHeight']] = []
    delta_shares = int(x['sharesAmount']) / (10 ** asset_info[102].decimals)
    delta_usdt = int(x['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes'][0]['amount']) / (10 ** asset_info[10].decimals)
    usdt_exec_prices[x['paraBlockHeight']].append(delta_usdt / delta_shares)
# calculate average oracle price for each block
avg_exec_prices = {}
for block_no in usdt_exec_prices:
    avg_exec_prices[block_no] = sum(usdt_exec_prices[block_no]) / len(usdt_exec_prices[block_no])
# sort by block_number
sorted_avg_exec_prices = sorted(avg_exec_prices.items(), key=lambda x: x[0])

ss_exec_prices = []
last_block_no = min_block
last_price = sorted_avg_exec_prices[0][1]
for (block_no, price) in sorted_avg_exec_prices:
    ss_exec_prices.extend([last_price] * (block_no - last_block_no))
    last_block_no = block_no
    last_price = price
ss_exec_prices.extend([last_price] * (max_block - last_block_no + 1))
assert len(ss_exec_prices) == (max_block - min_block + 1)

omnipool_data = get_omnipool_asset_data(min_block, max_block, [tkn_asset_id, 102])
hub_price_of_tkn = []
hub_price_of_stable = []
for x in omnipool_data:
    if x['assetId'] == tkn_asset_id:
        assert x['paraChainBlockHeight'] == min_block + len(hub_price_of_tkn), "Block number must be sequential"
        hub_price_of_tkn.append(
            (int(x['assetState']['hubReserve']) / 10 ** asset_info[1].decimals)
            / (int(x['balances']['free']) / 10 ** asset_info[tkn_asset_id].decimals)
        )
    elif x['assetId'] == 102:
        assert x['paraChainBlockHeight'] == min_block + len(hub_price_of_stable), "Block number must be sequential"
        hub_price_of_stable.append(
            (int(x['assetState']['hubReserve']) / 10 ** asset_info[1].decimals)
            / (int(x['balances']['free']) / 10 ** asset_info[102].decimals)
        )
    else:
        raise ValueError("Invalid asset ID")
# price of TKN denominated in 2-pool
price_of_tkn = [hub_price_of_tkn[i] / hub_price_of_stable[i] for i in range(max_block - min_block + 1)]
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
max_tau = 100
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