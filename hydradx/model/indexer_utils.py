import requests

URL_UNIFIED_PROD = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'
URL_OMNIPOOL_STORAGE = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'
URL_STABLEPOOL_STORAGE = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:stablepool/api/graphql'

class AssetInfo:
    def __init__(
            self,
            asset_type: str = None,
            decimals: int = None,
            existential_deposit: int = None,
            id: str = None,
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


def get_asset_info_by_ids(asset_ids: list) -> dict:

    asset_query = """
    query assetInfoByAssetIds($assetIds: [String!]!) {
      assets(filter: {id: {in: $assetIds}}) {
        nodes {
          assetType
          decimals
          existentialDeposit
          id
          isSufficient
          name
          symbol
          xcmRateLimit
        }
      }
    }
    """

    # Send POST request to the GraphQL API
    variables = {'assetIds': [f"{asset_id}" for asset_id in asset_ids]}
    return_val = query_indexer(URL_OMNIPOOL_STORAGE, asset_query, variables)
    asset_data = return_val['data']['assets']['nodes']
    dict_data = {}
    for asset in asset_data:
        asset_info = AssetInfo(
            asset_type=asset['assetType'],
            decimals=int(asset['decimals']),
            existential_deposit=int(asset['existentialDeposit']),
            id=asset['id'],
            is_sufficient=asset['isSufficient'],
            name=asset['name'],
            symbol=asset['symbol']
        )
        dict_data[int(asset['id'])] = asset_info
    return dict_data


def get_omnipool_asset_data(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list = None
) -> list:

    variables = {
        "minBlock": min_block_id,
        "maxBlock": max_block_id
    }

    if asset_ids is None:
        query = """
        query AssetBalancesByBlockHeight($first: Int!, $after: Cursor, $minBlock: Int!, $maxBlock: Int!) {
          omnipoolAssetData(
            first: $first,
            after: $after,
            orderBy: PARA_CHAIN_BLOCK_HEIGHT_ASC,
            filter: {
              paraChainBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
            }
          ) {
            nodes {
              paraChainBlockHeight
              assetId
              balances
              assetState
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """

    else:
        query = """
        query AssetBalancesByBlockHeight($first: Int!, $after: Cursor, $minBlock: Int!, $maxBlock: Int!, $assetIds: [Int!]!) {
          omnipoolAssetData(
            first: $first,
            after: $after,
            orderBy: PARA_CHAIN_BLOCK_HEIGHT_ASC,
            filter: {
              paraChainBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock },
              assetId: { in: $assetIds }
            }
          ) {
            nodes {
              paraChainBlockHeight
              assetId
              balances
              assetState
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """
        variables["assetIds"] = asset_ids

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_OMNIPOOL_STORAGE, query, variables)
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
        asset_ids: list,
        asset_info: list[AssetInfo] = None
):
    if asset_info is None:
        asset_dict = get_asset_info_by_ids(asset_ids + [1])
    else:
        asset_dict = {int(asset.id): asset for asset in asset_info}

    data = get_omnipool_data_by_asset(min_block_id, max_block_id, asset_ids)
    liquidity = {}
    hub_liquidity = {}
    for asset_id in asset_ids:
        tkn_balances = [int(block['balances']['free']) / (10 ** asset_dict[asset_id].decimals) for block in data[asset_id]]
        hub_balances = [int(block['assetState']['hubReserve']) / (10 ** asset_dict[1].decimals) for block in data[asset_id]]
        liquidity[asset_id] = tkn_balances
        hub_liquidity[asset_id] = hub_balances
    return liquidity, hub_liquidity


def get_stableswap_asset_data(
        pool_id: int,
        min_block_id: int,
        max_block_id: int
) -> list:

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
    data = query_indexer(URL_STABLEPOOL_STORAGE, stableswap_query, variables)
    return data['data']['stablepools']['nodes']


def get_stableswap_data_by_block(
        pool_id: int,
        block_no: int
):
    return get_stableswap_asset_data(pool_id, block_no, block_no)[0]


def get_latest_stableswap_data(
        pool_id: int
):

    latest_block_query = """
    query MaxHeightQuery {
      maxHeightResult: stablepools(first: 1, orderBy: PARA_CHAIN_BLOCK_HEIGHT_DESC) {
        nodes {
          paraChainBlockHeight
        }
      }
    }
    """

    data = query_indexer(URL_STABLEPOOL_STORAGE, latest_block_query)

    latest_block = int(data['data']['maxHeightResult']['nodes'][0]['paraChainBlockHeight'])
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
    asset_dict = get_asset_info_by_ids(asset_id_list)

    for asset in pool_data['stablepoolAssetDataByPoolId']['nodes']:
        asset_id = asset['assetId']
        balance = int(asset['balances']['free']) / (10 ** asset_dict[asset_id].decimals)
        pool_data_formatted['liquidity'][asset_id] = balance
    return pool_data_formatted


def get_stablepool_ids():

    stablepool_query = """
    query MyQuery {
      stablepools {
        groupedAggregates(groupBy: POOL_ID) {
          keys
        }
      }
    }
    """

    data = query_indexer(URL_STABLEPOOL_STORAGE, stablepool_query)
    pool_ids = [int(pool['keys'][0]) for pool in data['data']['stablepools']['groupedAggregates']]
    return pool_ids


def get_fee_history(asset_id: int, min_block: int, max_block: int = None):

    if max_block is None:
        latest_block_query = """
        query MaxHeightQuery {
          maxHeightResult: swaps(first: 1, orderBy: PARA_BLOCK_HEIGHT_DESC) {
            nodes {
              paraBlockHeight
            }
          }
        }
        """
        data = query_indexer(URL_UNIFIED_PROD, latest_block_query)
        max_block = int(data['data']['maxHeightResult']['nodes'][0]['paraBlockHeight'])

    fee_query = """
    query fees_query($first: Int!, $after: Cursor, $assetId: String!, $minBlock: Int!, $maxBlock: Int!) {
      swaps(
        first: $first,
        after: $after,
        filter: {
          allInvolvedAssetIds: {contains: [$assetId, "1"]},
          paraBlockHeight:{greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock}
        }
        orderBy: PARA_BLOCK_HEIGHT_ASC
      ) {
          nodes {
            id
            swapOutputs {
              nodes {
                amount
                assetId
              }
            }
            swapInputs {
              nodes {
                amount
                assetId
              }
            }
            swapFees {
              nodes {
                amount
                assetId
                recipientId
              }
            }
            paraBlockHeight
          }
          pageInfo {
            hasNextPage
            endCursor
          }
        }
    }
    """

    variables = {"assetId": str(asset_id), "minBlock": min_block, "maxBlock": max_block}
    # data = query_indexer(url, hdx_fee_query, variables)

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, fee_query, variables)
        page_data = data['data']['swaps']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['swaps']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']

    return data_all


def get_fee_pcts(data, asset_id):
    fee_pcts = [
        [int(x['paraBlockHeight']),
        (sum([int(y['amount']) for y in x['swapFees']['nodes'] if y['assetId'] == str(asset_id)])
         / int(x['swapOutputs']['nodes'][0]['amount']))]
        for x in data if x['swapOutputs']['nodes'][0]['assetId'] == str(asset_id)
    ]
    return fee_pcts


def get_executed_trades(asset_ids, min_block: int, max_block: int):

    executed_trades_query = """
    query executed_trade_query(
        $first: Int!, $after: Cursor,
        $assetIds: [String!]!, $minBlock: Int!, $maxBlock: Int!
    ) {
      routedTrades(
        first: $first,
        after: $after,
        filter: {
            allInvolvedAssetIds: {contains: $assetIds},
            paraBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
        }
        orderBy: PARA_BLOCK_HEIGHT_ASC
      ) {
        nodes {
          routeTradeInputs {
            nodes {
              amount
              assetId
            }
          }
          routeTradeOutputs {
            nodes {
              amount
              assetId
            }
          }
          paraBlockHeight
          allInvolvedAssetIds
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    variables = {"assetIds": [str(id) for id in asset_ids], "minBlock": min_block, "maxBlock": max_block}
    # data = query_indexer(url, hdx_fee_query, variables)

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, executed_trades_query, variables)
        page_data = data['data']['routedTrades']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['routedTrades']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']

    asset_info = get_asset_info_by_ids(asset_ids)

    trade_data = [
        {
            'block_number': int(x['paraBlockHeight']),
            'input_asset_id': int(x['routeTradeInputs']['nodes'][0]['assetId']),
            'input_amount': int(x['routeTradeInputs']['nodes'][0]['amount']) / (10 ** asset_info[int(x['routeTradeInputs']['nodes'][0]['assetId'])].decimals),
            'output_asset_id': int(x['routeTradeOutputs']['nodes'][0]['assetId']),
            'output_amount': int(x['routeTradeOutputs']['nodes'][0]['amount']) / (10 ** asset_info[int(x['routeTradeOutputs']['nodes'][0]['assetId'])].decimals),
            'all_involved_asset_ids': [y for y in x['allInvolvedAssetIds']]
        }
        for x in data_all if (x['routeTradeOutputs']['nodes'][0]['assetId'] in variables['assetIds']
                              and x['routeTradeInputs']['nodes'][0]['assetId'] in variables['assetIds'])
    ]

    return trade_data


def get_stableswap_liquidity_events(pool_id: int, min_block: int, max_block: int):

    events_query = """
    query MyQuery(
      $first: Int!, $after: Cursor,
      $poolId: String!, $minBlock: Int!, $maxBlock: Int!
    ) {
      stableswapLiquidityEvents(
        first: $first,
        after: $after,
        orderBy: PARA_BLOCK_HEIGHT_ASC
        filter: {poolId: {equalTo: $poolId}, paraBlockHeight: {greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock}}
      ) {
        nodes {
          sharesAmount
          actionType
          paraBlockHeight
          stableswapAssetLiquidityAmountsByLiquidityActionId {
            nodes {
              amount
              assetId
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    variables = {"poolId": str(pool_id), "minBlock": min_block, "maxBlock": max_block}

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, events_query, variables)
        page_data = data['data']['stableswapLiquidityEvents']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['stableswapLiquidityEvents']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']

    return data_all
