from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_asset_info_by_ids, get_omnipool_data_by_asset, get_omnipool_liquidity

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

# liquidity, lrna_liquidity = get_omnipool_liquidity(min_block_no, max_block_no, [0, 5, 10, 102])
liquidity, lrna_liquidity = get_omnipool_liquidity(min_block_no, min_block_no+5, [0, 5, 102])

print(liquidity)
print(lrna_liquidity)