from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_fee_history, query_indexer

c1, c2 = st.columns(2)


with c1:
    center_block = st.number_input(
        "center block ID",
        min_value=0, max_value=99999999, value=6973967, step=1, key="center_block", format="%d"
    )

with c2:
    total_blocks = st.number_input(
        "total block count",
        min_value=0, max_value=99999999, value=400000, step=1, key="total_blocks", format="%d",
        help="set to 0 to get latest data available"
    )

min_block = center_block - total_blocks // 2
max_block = center_block + total_blocks // 2

url = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'
buyback_query = """
query buyback_query($first: Int!, $after: Cursor, $minBlock: Int!, $maxBlock: Int!) {
  swaps(
    first: $first,
    after: $after,
    filter: {
      swapperId: {equalToInsensitive: "0x6d6f646c726566657272616c0000000000000000000000000000000000000000"},
      allInvolvedAssetIds: {contains: "0"},
      paraBlockHeight: {greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock}
    }
    orderBy: PARA_BLOCK_HEIGHT_ASC
  ) {
    nodes {
      swapOutputs {
        nodes {
          amount
          assetId
        }
      }
      allInvolvedAssetIds
      paraBlockHeight
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


variables = {"minBlock": min_block, "maxBlock": max_block}

data_all = []
has_next_page = True
after_cursor = None
page_size = 10000
variables["first"] = page_size

while has_next_page:
    variables["after"] = after_cursor
    data = query_indexer(url, buyback_query, variables)
    page_data = data['data']['swaps']['nodes']
    data_all.extend(page_data)
    page_info = data['data']['swaps']['pageInfo']
    has_next_page = page_info['hasNextPage']
    after_cursor = page_info['endCursor']

buckets = 100
bucket_size = (max_block - min_block) // buckets

buybacks_per_bucket = {}
first_half_total = 0
second_half_total = 0
for i in range(len(data_all)):
    swap_assets = data_all[i]['allInvolvedAssetIds']
    if len(swap_assets) != 2 or '0' not in swap_assets or '1' not in swap_assets:
        raise ValueError("Invalid swap assets")
    hour_id = data_all[i]['paraBlockHeight'] // bucket_size
    amt = int(data_all[i]['swapOutputs']['nodes'][0]['amount']) / 1e12
    if hour_id not in buybacks_per_bucket:
        buybacks_per_bucket[hour_id] = 0
    buybacks_per_bucket[hour_id] += amt
    if data_all[i]['paraBlockHeight'] < center_block:
        first_half_total += amt
    else:
        second_half_total += amt

st.text("HDX buybacks in first half of data:")
st.text(f"{first_half_total:.2f} HDX")
st.text("HDX buybacks in second half of data:")
st.text(f"{second_half_total:.2f} HDX")

# plot the data
fig, ax = plt.subplots()
blocks = [x * bucket_size for x in buybacks_per_bucket.keys()]
ax.plot(blocks, list(buybacks_per_bucket.values()), label='buybacks')
ax.set_title(f"HDX buybacks every {bucket_size} blocks")
ax.set_xlabel("Block ID")
ax.legend()
st.pyplot(fig)

print("done")