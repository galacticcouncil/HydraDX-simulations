from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv, json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_asset_info_by_ids, get_omnipool_swap_fees

tkn = 'tBTC'
tkn = 'DOT'
if tkn == 'DOT':
    oracle_data_filename = 'DOTUSD_oracle_prices.csv'
    tkn_id = 5
elif tkn == 'tBTC':
    oracle_data_filename = 'tBTCUSD_oracle_prices.csv'
    tkn_id = 1000765
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

omnipool_data_file_prefix = 'omnipool_spot_prices_102_' + str(tkn_id)
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

tkn_asset_id = tkn_id
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

b = 0.0001  # buffer
m = 0.9  # pct of arb to take
min_asset_fees = [
    max([0, 1 - (spot_prices[i] * (1 + b) / oracle_prices[i])]) * m
    for i in range(len(spot_prices))
]
min_protocol_fees = [
    max([0, 1 - (oracle_prices[i] * (1 + b) / spot_prices[i])]) * m
    for i in range(len(spot_prices))
]

# plot
fig, ax = plt.subplots()
ax.plot(range(min_block, max_block + 1), min_asset_fees, label='min asset fees')
ax.plot(range(min_block, max_block + 1), min_protocol_fees, label='min protocol fees')
ax.set_title("Minimum asset and protocol fees")
ax.set_xlabel("Block number")
ax.set_ylabel("Fee")
ax.legend()
st.pyplot(fig)

min_block = max_block - 100000  # temporary
# min_block = 6_800_000
asset_fees, hub_fees = get_omnipool_swap_fees(tkn_id, min_block, max_block)

feeless_outs_asset = [x['fee_amount'] + x['output_amount'] for x in asset_fees]
feeless_outs_hub = [x['output_amount'] for x in hub_fees]

real_fees_asset = [(x['block_number'], x['fee_amount']) for x in asset_fees]
real_fees_hub = [(x['block_number'], x['fee_amount']) for x in hub_fees]

oracle_asset_fees_pcts = [(x['block_number'], max(x['fee_pct'], min_asset_fees[x['block_number'] - min_block])) for x in asset_fees]
oracle_hub_fees_pcts = [(x['block_number'], max(x['fee_pct'], min_protocol_fees[x['block_number'] - min_block])) for x in hub_fees]
oracle_asset_fees = [(oracle_asset_fees_pcts[i][0], oracle_asset_fees_pcts[i][1] * feeless_outs_asset[i]) for i in range(len(asset_fees))]
oracle_hub_fees = [(oracle_hub_fees_pcts[i][0], oracle_hub_fees_pcts[i][1] * feeless_outs_hub[i]) for i in range(len(hub_fees))]

# calculate average fee accrual per block in buckets
# bucket_size = 5*60  # 1 hour buckets
def bucket_fees(bucket_ct: int, min_block: int, max_block: int, oracle_asset_fees, oracle_hub_fees, real_fees_asset,
                real_fees_hub):
    bucket_size = (max_block - min_block) // bucket_ct + 1
    bucketed_asset_fees = [0] * bucket_ct
    for x in oracle_asset_fees:
        bucket_index = (x[0] - min_block) // bucket_size
        bucketed_asset_fees[bucket_index] += x[1] / bucket_size

    bucketed_hub_fees = [0] * bucket_ct
    for x in oracle_hub_fees:
        bucket_index = (x[0] - min_block) // bucket_size
        bucketed_hub_fees[bucket_index] += x[1] / bucket_size

    bucketed_asset_fees_real = [0] * bucket_ct
    for x in real_fees_asset:
        bucket_index = (x[0] - min_block) // bucket_size
        bucketed_asset_fees_real[bucket_index] += x[1] / bucket_size

    bucketed_hub_fees_real = [0] * bucket_ct
    for x in real_fees_hub:
        bucket_index = (x[0] - min_block) // bucket_size
        bucketed_hub_fees_real[bucket_index] += x[1] / bucket_size

    return bucketed_asset_fees, bucketed_hub_fees, bucketed_asset_fees_real, bucketed_hub_fees_real

bucketed_asset_fees, bucketed_hub_fees, bucketed_asset_fees_real, bucketed_hub_fees_real = bucket_fees(
    30, min_block, max_block, oracle_asset_fees, oracle_hub_fees, real_fees_asset, real_fees_hub
)

add_oracle_asset_fees = [bucketed_asset_fees[i] - bucketed_asset_fees_real[i] for i in range(len(bucketed_asset_fees))]
add_oracle_hub_fees = [bucketed_hub_fees[i] - bucketed_hub_fees_real[i] for i in range(len(bucketed_hub_fees))]
if min(add_oracle_asset_fees) < -1e-12:
    raise ValueError("Oracle asset fees are negative, something is wrong")
if min(add_oracle_hub_fees) < -1e-12:
    raise ValueError("Oracle hub fees are negative, something is wrong")


# plot oracle asset fees
fig, ax = plt.subplots()
ax.bar(np.arange(bucket_ct), bucketed_asset_fees_real, label='Real asset fees')
ax.bar(np.arange(bucket_ct), add_oracle_asset_fees, bottom=bucketed_asset_fees_real, label='With oracle fees')

ax.set_xticks(np.arange(bucket_ct))
ax.set_xticklabels(range(min_block, max_block + 1, bucket_size), rotation=90)
ax.set_ylabel(f'Fee per block, in {tkn}')
ax.set_title("Asset fee comparison")
ax.set_xlabel("Block number")
ax.legend()
st.pyplot(fig)

# plot oracle hub fees
fig, ax = plt.subplots()
ax.bar(np.arange(bucket_ct), bucketed_hub_fees_real, label='Real hub fees')
ax.bar(np.arange(bucket_ct), add_oracle_hub_fees, bottom=bucketed_hub_fees_real, label='With oracle fees')

ax.set_xticks(np.arange(bucket_ct))
ax.set_xticklabels(range(min_block, max_block + 1, bucket_size), rotation=90)
ax.set_ylabel(f'Fee per block, in H2O')
ax.set_title("Hub fee comparison")
ax.set_xlabel("Block number")
ax.legend()
st.pyplot(fig)
