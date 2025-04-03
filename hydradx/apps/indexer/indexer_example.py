from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

url1 = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'
url2 = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'

query = """
{
  omnipoolAssetData(first: 10, orderBy: PARA_CHAIN_BLOCK_HEIGHT_DESC) {
    nodes {
      id
      paraChainBlockHeight
      poolAddress
      balances
      assetState
      assetId
    }
  }
}
"""

asset_query = """
{
  assets {
    nodes {
      decimals
      id
    }
  }
}
"""

# Send POST request to the GraphQL API
asset_response = requests.post(url1, json={'query': asset_query})
if asset_response.status_code != 200:
    raise ValueError(f"Query failed with status code {asset_response.status_code}")
asset_data = asset_response.json()['data']

response = requests.post(url1, json={'query': query})
if response.status_code != 200:
    raise ValueError(f"Query failed with status code {response.status_code}")

data = response.json()['data']
print(data)

