import requests

from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.processing import get_current_money_market
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState


class AssetInfo:
    def __init__(
            self,
            asset_type: str = None,
            decimals: int = None,
            existential_deposit: int = None,
            id: int = None,
            is_sufficient: bool = None,
            name: str = None,
            symbol: str = None,
            xcm_rate_limit = None
    ):
        self.asset_type = asset_type
        self.decimals = decimals
        self.existential_deposit = existential_deposit
        self.id = id
        self.is_sufficient = is_sufficient
        self.name = name
        self.symbol = symbol
        self.xcm_rate_limit = xcm_rate_limit


def query_indexer(url: str, query: str, variables: dict = None) -> dict:
    response = requests.post(url, json={'query': query, 'variables': variables})
    if response.status_code != 200:
        raise ValueError(f"Query failed with status code {response.status_code}")
    return_val = response.json()
    if 'errors' in return_val:
        raise ValueError(return_val['errors'][0]['message'])
    return return_val


def get_asset_info(asset_ids: list[int] = None) -> dict:
    url1 = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'

    asset_query = f"""
    query assetInfoByAssetIds{'($assetIds: [String!]!)' if asset_ids else ''} {{
      assets(filter: {{{
        'id: {in: $assetIds}, ' if asset_ids else ''
      }symbol: {{notEqualTo: "none"}}}}) {{
        nodes {{
          assetType
          decimals
          existentialDeposit
          id
          isSufficient
          name
          symbol
          xcmRateLimit
        }}
      }}
    }}
    """

    # Send POST request to the GraphQL API
    variables = {}
    if asset_ids:
        variables['assetIds'] = [str(i) for i in asset_ids]
    return_val = query_indexer(url1, asset_query, variables)
    asset_data = return_val['data']['assets']['nodes']
    dict_data = {}
    for asset in asset_data:
        asset_info = AssetInfo(
            asset_type=asset['assetType'],
            decimals=int(asset['decimals'] if asset['decimals'] else 0),
            existential_deposit=int(asset['existentialDeposit']),
            id=int(asset['id']),
            is_sufficient=asset['isSufficient'],
            name=asset['name'],
            symbol=asset['symbol']
        )
        dict_data[int(asset['id'])] = asset_info
    return dict_data


def get_omnipool_asset_data(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list[int] = None
) -> list:
    url = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'

    variables = {
        "minBlock": min_block_id,
        "maxBlock": max_block_id
    }

    query = f"""
    query AssetBalancesByBlockHeight($first: Int!, $after: Cursor, $minBlock: Int!, $maxBlock: Int!{
    ', $assetIds: [Int!]!' if asset_ids else ''
    }) {{
      omnipoolAssetData(
        first: $first,
        after: $after,
        orderBy: PARA_CHAIN_BLOCK_HEIGHT_ASC,
        filter: {{
          paraChainBlockHeight: {{ greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }} {
            'assetId: { in: $assetIds }' if asset_ids else ''
          }
        }}
      ) {{
        nodes {{
          paraChainBlockHeight
          assetId
          balances
          assetState
        }}
        pageInfo {{
          hasNextPage
          endCursor
        }}
      }}
    }}
    """

    variables["assetIds"] = asset_ids

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(url, query, variables)
        page_data = data['data']['omnipoolAssetData']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['omnipoolAssetData']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']
    return data_all


def get_omnipool_data_by_asset(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list = None,
        validate: bool = True
) -> dict:
    data = get_omnipool_asset_data(min_block_id, max_block_id, asset_ids)
    data_by_asset = {}
    for item in data:
        if item['assetId'] not in data_by_asset:
            data_by_asset[item['assetId']] = []
        elif validate:
            last_block_id = data_by_asset[item['assetId']][-1]['paraChainBlockHeight']
            next_block_id = item['paraChainBlockHeight']
            assert last_block_id + 1 == next_block_id
        data_by_asset[item['assetId']].append(item)
    return data_by_asset


def get_omnipool_liquidity(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list[int] = None,
        asset_info: list[AssetInfo] = None
):
    if asset_info is None:
        asset_dict = get_asset_info(asset_ids + [1])
    else:
        asset_dict = {int(asset.id): asset for asset in asset_info}
    if asset_ids is None:
        asset_ids = list(asset_dict.keys())

    data = get_omnipool_data_by_asset(min_block_id, max_block_id, asset_ids)
    liquidity = {}
    hub_liquidity = {}
    for asset_id in asset_ids:
        tkn_balances = [int(block['balances']['free']) / (10 ** asset_dict[asset_id].decimals) for block in data[asset_id]]
        hub_balances = [int(block['assetState']['hubReserve']) / (10 ** asset_dict[1].decimals) for block in data[asset_id]]
        liquidity[asset_id] = tkn_balances
        hub_liquidity[asset_id] = hub_balances
    return liquidity, hub_liquidity


def get_current_block_height():
    url = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'

    latest_block_query = """
        query BlockHeight {
            blocks(last: 1) {
                nodes {
                    height
                }
            }
        }
    """

    data = query_indexer(url, latest_block_query)
    latest_block = data['data']['blocks']['nodes'][0]['height']
    return latest_block


def get_stableswap_asset_data(
        pool_id: int,
        min_block_id: int,
        max_block_id: int
) -> list:
    url = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:stablepool/api/graphql'

    stableswap_query = """
    query MyQuery($pool_id: Int!, $min_block: Int!, $max_block: Int!) {
      stablepools(
        orderBy: PARA_CHAIN_BLOCK_HEIGHT_ASC,
        filter: {poolId: {equalTo: $pool_id}, paraChainBlockHeight: {greaterThanOrEqualTo: $min_block, lessThanOrEqualTo: $max_block}}
      ) {
        nodes {
          id
          poolId
          stablepoolAssetDataByPoolId {
            nodes {
              assetId
              balances
              id
            }
          }
          fee
          finalAmplification
          finalBlock
          initialAmplification
          initialBlock
        }
      }
    }
    """
    variables = {"pool_id": pool_id, "min_block": min_block_id, "max_block": max_block_id}
    data = query_indexer(url, stableswap_query, variables)
    return data['data']['stablepools']['nodes']


def get_stableswap_data_by_block(
        pool_id: int,
        block_no: int
):
    return get_stableswap_asset_data(pool_id, block_no, block_no)[0]


def get_latest_stableswap_data(
        pool_id: int
):

    latest_block = get_current_block_height()
    pool_data = get_stableswap_data_by_block(pool_id, latest_block)
    pool_data_formatted = {
        "pool_id": pool_data['poolId'],
        "block": latest_block,
        'fee': pool_data['fee'] / 1000000,
        'finalAmplification': pool_data['finalAmplification'],
        'finalBlock': pool_data['finalBlock'],
        'initialAmplification': pool_data['initialAmplification'],
        'initialBlock': pool_data['initialBlock'],
        'liquidity': {}
    }

    asset_id_list = [asset['assetId'] for asset in pool_data['stablepoolAssetDataByPoolId']['nodes']]
    asset_dict = get_asset_info(asset_id_list)

    for asset in pool_data['stablepoolAssetDataByPoolId']['nodes']:
        asset_id = asset['assetId']
        balance = int(asset['balances']['free']) / (10 ** asset_dict[asset_id].decimals)
        pool_data_formatted['liquidity'][asset_id] = balance
    return pool_data_formatted


def get_stablepool_ids():
    url = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:stablepool/api/graphql'

    stablepool_query = """
    query MyQuery {
      stablepools {
        groupedAggregates(groupBy: POOL_ID) {
          keys
        }
      }
    }
    """

    data = query_indexer(url, stablepool_query)
    pool_ids = [int(pool['keys'][0]) for pool in data['data']['stablepools']['groupedAggregates']]
    return pool_ids


def get_current_omnipool():
    asset_info = get_asset_info()
    current_block = get_current_block_height()
    omnipool_data = get_omnipool_asset_data(min_block_id=current_block - 10000, max_block_id=current_block)
    liquidity = {}
    lrna = {}
    for item in reversed(omnipool_data):
        if asset_info[item['assetId']].symbol not in liquidity:
            liquidity[asset_info[item['assetId']].symbol] = int(item['balances']['free']) / (10 ** asset_info[item['assetId']].decimals)
            lrna[asset_info[item['assetId']].symbol] = int(item['assetState']['hubReserve']) / (10 ** asset_info[1].decimals)
        if len(liquidity) == len(asset_info):
            break

    omnipool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in liquidity
        },
        asset_fee=DynamicFee(
            minimum=0.0025,
            maximum=0.05,
            amplification=2,
            decay=0.001
        ),
        lrna_fee=DynamicFee(
            minimum=0.0005,
            maximum=0.01,
            amplification=1,
            decay=0.0005
        )
    )
    return omnipool

def get_omnipool_router():
    omnipool = get_current_omnipool()
    stableswap_data = {
        pool: get_latest_stableswap_data(pool)
        for pool in get_stablepool_ids()
    }
    asset_info = get_asset_info()
    # money_market = get_current_money_market()
    stableswap_pools = []
    for pool in stableswap_data.values():
        if min(pool['liquidity'].values()) > 0:
            stableswap_pools.append(
                StableSwapPoolState(
                    tokens={asset_info[tkn_id].symbol: pool['liquidity'][tkn_id] for tkn_id in pool['liquidity']},
                    amplification=pool['finalAmplification'],
                    trade_fee=pool['fee'],
                    unique_id=asset_info[pool['pool_id']].symbol,
                )
            )
    return OmnipoolRouter(
        exchanges=[
            omnipool, *stableswap_pools
        ]
    )
