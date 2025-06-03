import json
import requests

from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.processing import get_stableswap_data


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
    while response.status_code == 504:
        print("504 error, retrying...")
        response = requests.post(url, json={'query': query, 'variables': variables})
    if response.status_code != 200:
        raise ValueError(f"Query failed with status code {response.status_code}")
    return_val = response.json()
    if 'errors' in return_val:
        raise ValueError(return_val['errors'][0]['message'])
    return return_val


def get_asset_info_by_ids(asset_ids: list = None) -> dict:

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
    variables = {'assetIds': [f"{asset_id}" for asset_id in asset_ids]} if asset_ids else None
    return_val = query_indexer(URL_OMNIPOOL_STORAGE, asset_query, variables)
    asset_data = return_val['data']['assets']['nodes']
    dict_data = {}
    for asset in asset_data:
        asset_info = AssetInfo(
            asset_type=asset['assetType'],
            decimals=int(asset['decimals'] if asset['decimals'] else 0),
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
        asset_ids: list[int] = None
) -> list:

    variables: dict[str, object] = {
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

    if asset_ids is not None:
        variables["assetIds"] = [int(asset_id) for asset_id in asset_ids]

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 2000
    variables["first"] = page_size

    i = 0
    while has_next_page:
        print(f'Fetching page {i}...')
        variables["after"] = after_cursor
        data = query_indexer(URL_OMNIPOOL_STORAGE, query, variables)
        print(f'Processing page {i}...')
        page_data = data['data']['omnipoolAssetData']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['omnipoolAssetData']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']
        i += 1
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
        asset_dict = get_asset_info_by_ids(asset_ids + [1])
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


def get_current_omnipool_assets():
    query = """
    query assetInfoByAssetIds {
      omnipoolAssets(filter: {isRemoved: {equalTo: false}}) {
        nodes {
          assetId
        }
      }
    }
    """
    data = query_indexer(URL_UNIFIED_PROD, query)
    return [node['assetId'] for node in data['data']['omnipoolAssets']['nodes']]


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


def get_current_stableswap_pools(asset_info: dict[int: AssetInfo] = None):
    stableswap_data = {
        pool: get_latest_stableswap_data(pool)
        for pool in get_stablepool_ids()
    }
    if asset_info is None:
        asset_info = get_asset_info_by_ids()
    stableswap_pools = []
    for pool in stableswap_data.values():
        if min(pool['liquidity'].values()) > 0:
            stableswap_pools.append(
                StableSwapPoolState(
                    tokens={asset_info[tkn_id].symbol: pool['liquidity'][tkn_id] for tkn_id in pool['liquidity']},
                    amplification=pool['finalAmplification'],
                    trade_fee=pool['fee'],
                    unique_id=asset_info[pool['pool_id']].name,
                )
            )
    return stableswap_pools

def get_current_omnipool():
    asset_ids = get_current_omnipool_assets()
    asset_info = get_asset_info_by_ids(asset_ids)
    asset_ids.remove('1')  # Remove hub asset ID
    for asset in asset_info.values():
        if asset.asset_type == 'StableSwap':
            asset.symbol = asset.name
    current_block = get_current_block_height()
    asset_ids_remaining = asset_ids.copy()
    liquidity = {}
    lrna = {}
    while asset_ids_remaining:
        omnipool_data = get_omnipool_asset_data(
            min_block_id=current_block - 100,
            max_block_id=current_block,
            asset_ids=asset_ids_remaining
        )
        for item in reversed(omnipool_data):
            asset = asset_info[item['assetId']]
            if asset.symbol not in liquidity:
                liquidity[asset.symbol] = int(item['balances']['free']) / (
                            10 ** asset.decimals)
                lrna[asset.symbol] = int(item['assetState']['hubReserve']) / (
                            10 ** asset_info[1].decimals)
                asset_ids_remaining.remove(asset.id)
        current_block -= 100

    asset_fee, lrna_fee = get_current_omnipool_fees(asset_info)
    for tkn in liquidity:
        if tkn not in asset_fee.current:
            asset_fee.current[tkn] = asset_fee.minimum
            asset_fee.last_updated[tkn] = current_block
        if tkn not in lrna_fee.current:
            lrna_fee.current[tkn] = asset_fee.minimum
            lrna_fee.last_updated[tkn] = current_block
    omnipool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in liquidity
        },
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    omnipool.time_step = current_block
    return omnipool


def get_current_omnipool_fees(
        asset_info: dict[int: AssetInfo] = None
) -> tuple[DynamicFee, DynamicFee]:

    if asset_info is None:
        asset_info = get_asset_info_by_ids()
    url = URL_UNIFIED_PROD
    query = """
        query MyQuery {
            events(
                first: 10000
                orderBy: PARA_BLOCK_HEIGHT_DESC
                filter: {name: {includes: "Omnipool"}}
            ) {
                nodes {
                  name
                  args
                  id
                  paraBlockHeight
                }
            }
        }
    """
    transaction_data = query_indexer(url, query)['data']['events']['nodes']
    asset_fee = DynamicFee(
        minimum=0.0015,
        maximum=0.05,
        amplification=2,
        decay=0.001
    )
    lrna_fee = DynamicFee(
        minimum=0.0005,
        maximum=0.01,
        amplification=1,
        decay=0.005
    )
    for trade in transaction_data:
        args = {
            arg[:arg.index(':')].strip('"'): arg[arg.index(':') + 1:].strip('"')
            for arg in trade['args'].strip('}').strip('{').split(',')
        }
        block = trade['paraBlockHeight']
        tkn_sell = asset_info[int(args['assetIn'])].symbol
        tkn_buy = asset_info[int(args['assetOut'])].symbol
        if tkn_sell not in lrna_fee.current and float(args['hubAmountOut']) > 0:
            lrna_fee.current[tkn_sell] = float(args['protocolFeeAmount']) / float(args['hubAmountOut'])
            lrna_fee.last_updated[tkn_sell] = block
        if tkn_buy not in asset_fee.current and float(args['amountIn']) > 0:
            asset_fee.current[tkn_buy] = float(args['assetFeeAmount']) / (float(args['amountOut']) + float(args['assetFeeAmount']))
            asset_fee.last_updated[tkn_buy] = block
    return asset_fee, lrna_fee


def get_current_omnipool_router():
    omnipool = get_current_omnipool()
    asset_info = get_asset_info_by_ids()
    stable_swap_data = get_stableswap_data()
    stableswap_pools = []
    for pool in stable_swap_data.values():
        pool_name = asset_info[pool.pool_id].name
        if pool.pool_id == 690:
            peg = omnipool.lrna_price('vDOT') / omnipool.lrna_price('DOT')
            shares = pool.shares / 10 ** 18
            tokens = {
                'DOT': shares * peg / (1 + peg),
                'vDOT': shares / (1 + peg),
            }
        else:
            tokens={
                asset_info[tkn_id].symbol:
                    int(pool.reserves[tkn_id]) / 10 ** asset_info[tkn_id].decimals
                for tkn_id in pool.reserves
            }
            peg = None
        stableswap_pools.append(
            StableSwapPoolState(
                tokens=tokens,
                peg=peg,
                amplification=pool.final_amplification,
                trade_fee=pool.fee,
                unique_id=pool_name,
            )
        )
    # money_market = get_current_money_market()

    return OmnipoolRouter(
        exchanges=[
            omnipool, *stableswap_pools
        ]
    )


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
    page_size = 1000
    variables["first"] = page_size

    i = 0
    while has_next_page:
        print(f'Fetching page {i}...')
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, events_query, variables)
        page_data = data['data']['stableswapLiquidityEvents']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['stableswapLiquidityEvents']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']
        i += 1

    return data_all


def download_stableswap_exec_prices(pool_id: int, tkn_id: int, min_block: int, max_block: int, path: str):
    data = get_stableswap_liquidity_events(pool_id, min_block, max_block)
    asset_info = get_asset_info_by_ids([tkn_id, pool_id])
    decimals_tkn_id = asset_info[tkn_id].decimals
    decimals_pool_id = asset_info[pool_id].decimals
    prices_by_block = {}
    for tx in data:
        liq_data = tx['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes']
        if len(liq_data) == 1 and int(liq_data[0]['assetId']) == tkn_id and int(liq_data[0]['amount']) > 0:
            block_no = int(tx['paraBlockHeight'])
            if block_no not in prices_by_block:
                prices_by_block[block_no] = []
            price = int(liq_data[0]['amount']) / int(tx['sharesAmount']) * (10 ** (decimals_pool_id - decimals_tkn_id))
            prices_by_block[block_no].append(price)
    avg_price_by_block = {}
    for block_no in prices_by_block:
        avg_price_by_block[block_no] = sum(prices_by_block[block_no]) / len(prices_by_block[block_no])
    with open(f"{path}stableswap_exec_prices_{pool_id}_{tkn_id}_{min_block}_{max_block}.json", "w") as f:
        json.dump(avg_price_by_block, f)


def download_omnipool_spot_prices(tkn_id: int, denom_id: int, min_block: int, max_block: int, path: str):
    data = get_omnipool_asset_data(min_block, max_block, [denom_id, tkn_id])
    # assert len(data) == 2 * (max_block - min_block + 1)
    if len(data) != 2 * (max_block - min_block + 1):
        print(f'Warning: len(data) is {len(data)}, expected {2 * (max_block - min_block + 1)}')
    asset_info = get_asset_info_by_ids([tkn_id, denom_id])
    tkn_decimals = asset_info[tkn_id].decimals
    denom_decimals = asset_info[denom_id].decimals
    prices = {}
    prev_block = None
    for i in range(max_block - min_block + 1):
        tkn_i = 2 * i + (0 if data[2 * i]['assetId'] == tkn_id else 1)
        denom_i = 2 * i + (0 if data[2 * i]['assetId'] == denom_id else 1)
        data_tkn = data[tkn_i]
        data_denom = data[denom_i]
        assert data_tkn['paraChainBlockHeight'] == data_denom['paraChainBlockHeight'], "Block number must match"
        block_no = int(data_tkn['paraChainBlockHeight'])
        assert block_no == min_block + i, "Block number must be sequential"
        hub_price_of_tkn = int(data_tkn['assetState']['hubReserve']) / int(data_tkn['balances']['free'])
        hub_price_of_denom = int(data_denom['assetState']['hubReserve']) / int(data_denom['balances']['free'])
        price = hub_price_of_tkn / hub_price_of_denom * 10 ** (tkn_decimals - denom_decimals)
        if prev_block is not None and prices[prev_block] == price:
            continue
        prices[block_no] = price
        prev_block = block_no

    with open(f"{path}omnipool_spot_prices_{denom_id}_{tkn_id}_{min_block}_{max_block}.json", "w") as f:
        json.dump(prices, f)


def get_omnipool_swap_fees_one_query(tkn_id: int, min_block: int, max_block: int):

    query = """
    query OmnipoolSwapsByAssetId(
      $assetId: String!,
      $minBlock: Int!, $maxBlock: Int!
    ) {
      swaps(
        filter: {
          allInvolvedAssetIds: {contains: ["1", $assetId], containedBy: ["1", $assetId]},
          paraBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
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
          swapFees {
            nodes {
              amount
              assetId
            }
          }
          paraBlockHeight
        }
      }
    }

    """

    variables = {"assetId": str(tkn_id), "minBlock": min_block, "maxBlock": max_block}
    data = query_indexer(URL_UNIFIED_PROD, query, variables)

    asset_info = get_asset_info_by_ids([tkn_id, 1])  # 1 is the hub asset ID

    asset_fee_data = []
    hub_fee_data = []
    for x in data['data']['swaps']['nodes']:
        block_number = int(x['paraBlockHeight'])
        asset_id = int(x['swapOutputs']['nodes'][0]['assetId'])
        decimals = asset_info[asset_id].decimals
        output_amount = int(x['swapOutputs']['nodes'][0]['amount']) / (10 ** decimals)
        fee_amount = sum([int(y['amount']) for y in x['swapFees']['nodes']]) / (10 ** decimals)
        if output_amount > 0:
            fee_dict = {
                    'block_number': block_number,
                    'output_amount': output_amount,
                    'fee_amount': fee_amount
                }
            if asset_id == tkn_id:
                fee_pct = fee_amount / (output_amount + fee_amount)
                fee_dict['fee_pct'] = fee_pct
                asset_fee_data.append(fee_dict)
            elif asset_id == 1:  # 1 is the hub asset ID
                fee_pct = fee_amount / output_amount if output_amount > 0 else 0
                fee_dict['fee_pct'] = fee_pct
                hub_fee_data.append(fee_dict)
    return asset_fee_data, hub_fee_data


def get_omnipool_swap_fees(tkn_id: int, min_block: int, max_block: int, max_block_count: int = 50000):
    min_block_temp = min_block
    max_block_temp = min(min_block + max_block_count - 1, max_block)
    asset_fee_data = []
    hub_fee_data = []
    while min_block_temp <= max_block:
        print(f"Querying from block {min_block_temp} to {max_block_temp}...")
        asset_data, hub_data = get_omnipool_swap_fees_one_query(tkn_id, min_block_temp, max_block_temp)
        asset_fee_data.extend(asset_data)
        hub_fee_data.extend(hub_data)
        min_block_temp = max_block_temp + 1
        max_block_temp = min(min_block_temp + max_block_count - 1, max_block)
    return asset_fee_data, hub_fee_data


def download_omnipool_swap_fees(tkn_id: int, min_block: int, max_block: int, path: str):
    asset_fees, hub_fees = get_omnipool_swap_fees(tkn_id, min_block, max_block)
    with open(f"{path}omnipool_swap_fees_{tkn_id}_{min_block}_{max_block}.json", "w") as f:
        json.dump({
            'asset_fees': asset_fees,
            'hub_fees': hub_fees
        }, f)
