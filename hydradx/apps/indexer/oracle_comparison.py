from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_asset_info_by_ids, get_omnipool_data_by_asset, get_omnipool_liquidity, \
    get_stableswap_asset_data
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

filename = 'DOTUSD_oracle_prices.csv'
file_path = os.path.join(project_root, 'hydradx', 'apps', 'indexer', filename)
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

liquidity, lrna = get_omnipool_liquidity(min_block_no, min_block_no+5, [0, 5, 102])
stableswap_data = get_stableswap_asset_data(102, min_block_no, min_block_no+5)

# price of DOT denominated in 2-pool
omnipool_spots_dot = [liquidity[102][i] * lrna[5][i] / (lrna[102][i] * liquidity[5][i]) for i in range(len(liquidity[102]))]
# price of 2-pool denominated in USDT
stableswap_objs = []
for i in range(len(stableswap_data)):
    liquidity_data = stableswap_data[i]['stablepoolAssetDataByPoolId']['nodes']
    tokens = {int(x['assetId']): int(x['balances']['free']) for x in liquidity_data}
    pool = StableSwapPoolState(
        tokens,
        amplification=int(stableswap_data[i]['initialAmplification'])
    )
    stableswap_objs.append(
        {
            'id': stableswap_data[i]['pool_id'],
            'price': stableswap_data[i]['price'],
            'liquidity': stableswap_data[i]['liquidity']
        }
    )

print(len(liquidity))
