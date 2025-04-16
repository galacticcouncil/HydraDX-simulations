import requests

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

def get_asset_info_by_ids(asset_ids: list) -> dict:
    url1 = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'

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
    asset_response = requests.post(url1, json={'query': asset_query, 'variables': variables})
    if asset_response.status_code != 200:
        raise ValueError(f"Query failed with status code {asset_response.status_code}")
    return_val = asset_response.json()
    if 'errors' in return_val:
        raise ValueError(return_val['errors'][0]['message'])
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
    url = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql'

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
        response = requests.post(url, json={"query": query, "variables": variables})
        if response.status_code != 200:
            raise ValueError(f"Query failed with status code {response.status_code}")

        data = response.json()
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
