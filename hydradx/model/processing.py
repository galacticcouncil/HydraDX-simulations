from web3 import Web3
import math
import json
from csv import reader
import requests
from zipfile import ZipFile
import datetime
import os
from hydradxapi import HydraDX
import time
import base64
from pprint import pprint
from dotenv import load_dotenv

from .amm.centralized_market import OrderBook, CentralizedMarket
from .amm.global_state import GlobalState, value_assets
from .amm.stableswap_amm import StableSwapPoolState
from .amm.omnipool_amm import OmnipoolState
from .amm.amm import basic_fee

cash_out = GlobalState.cash_out
impermanent_loss = GlobalState.impermanent_loss
pool_val = GlobalState.pool_val
deposit_val = GlobalState.deposit_val


def postprocessing(events: list, optional_params: list[str] = ()) -> list:
    """
    Definition:
    Compute more abstract metrics from the simulation

    Optional parameters:
    'withdraw_val': tracks the actual value of each agent's assets if they were withdrawn from the pool at each step
    'deposit_val': tracks the theoretical value of each agent's original assets at each step's current spot prices,
        if they had been held outside the pool from the beginning
    'holdings_val': the total value of the agent's outside holdings
    'pool_val': tracks the value of all assets held in the pool
    'impermanent_loss': computes loss for LPs due to price movements in either direction
    """
    # save initial state
    initial_state: GlobalState = events[0]
    withdraw_state: GlobalState = initial_state.copy()

    optional_params = set(optional_params)
    if 'impermanent_loss' in optional_params:
        optional_params.add('deposit_val')
        optional_params.add('withdraw_val')

    agent_params = {
        'deposit_val',
        'withdraw_val',
        'holdings_val',
        'impermanent_loss',
        'token_count',
        'trade_volume'
    }
    exchange_params = {
        'pool_val',
        # 'usd_price'
    }
    unrecognized_params = optional_params.difference(agent_params | exchange_params)
    if unrecognized_params:
        raise ValueError(f'Unrecognized parameter {unrecognized_params}')

    for step in events:
        state: GlobalState = step

        for pool in state.pools.values():
            if 'pool_val' in optional_params:
                pool.pool_val = state.pool_val(pool)

        # agents
        for agent in state.agents.values():
            if 'deposit_val' in optional_params:
                # what are this agent's original holdings theoretically worth at current spot prices?
                agent.deposit_val = value_assets(
                    state.market_prices(agent.holdings),
                    withdraw_state.agents[agent.unique_id].holdings
                )
            if 'withdraw_val' in optional_params:
                # what are this agent's holdings worth if sold?
                agent.withdraw_val = state.cash_out(agent)
            if 'holdings_val' in optional_params:
                agent.holdings_val = sum([quantity * state.price(asset) for asset, quantity in agent.holdings.items()])
            if 'impermanent_loss' in optional_params:
                agent.impermanent_loss = agent.withdraw_val / agent.deposit_val - 1
            if 'token_count' in optional_params:
                agent.token_count = sum(agent.holdings.values())
            if 'trade_volume' in optional_params:
                agent.trade_volume = 0
                if state.time_step > 0:
                    previous_agent = events[state.time_step - 1].agents[agent.unique_id]
                    agent.trade_volume += (
                        sum([
                            abs(previous_agent.holdings[tkn] - agent.holdings[tkn]) * state.price(tkn)
                            for tkn in agent.holdings])
                    )

    return events


def import_binance_prices(
    assets: list[str], start_date: str, days: int, interval: int = 12,
    stablecoin: str = 'USDT', return_as_dict: bool = False
) -> dict[str: list[float]]:

    start_date = datetime.datetime.strptime(start_date, "%b %d %Y")
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i), "%Y-%m-%d") for i in range(days)]

    # find the data folder
    while not os.path.exists("./data"):
        cwd = os.getcwd()
        os.chdir("..")
        if cwd == os.getcwd():
            raise FileNotFoundError("Could not find the data folder")

    # check that the files are all there, and if not, download them
    for tkn in assets:
        for date in dates:
            file = f"{tkn}{stablecoin}-1s-{date}"
            if os.path.exists(f'./data/{file}.csv'):
                continue
            else:
                print(f'Downloading {file}')
                url = f"https://data.binance.vision/data/spot/daily/klines/{tkn}{stablecoin}/1s/{file}.zip"
                response = requests.get(url)
                with open(f'./data/{file}.zip', 'wb') as f:
                    f.write(response.content)
                with ZipFile(f"./data/{file}.zip", 'r') as zipObj:
                    zipObj.extractall(path='./data')
                os.remove(f"./data/{file}.zip")
                # strip out everything except close price
                with open(f'./data/{file}.csv', 'r') as input_file:
                    rows = input_file.readlines()
                with open(f'./data/{file}.csv', 'w', newline='') as output_file:
                    output_file.write('\n'.join([row.split(',')[1] for row in rows]))

    # now that we have all the files, read them in
    price_data = {tkn: [] for tkn in assets}
    for tkn in assets:
        for date in dates:
            file = f"{tkn}{stablecoin}-1s-{date}"
            with open(f'./data/{file}.csv', 'r') as input_file:
                csvreader = reader(input_file)
                price_data[tkn] += [float(row[0]) for row in csvreader][::interval]

    if not return_as_dict:
        data_length = min([len(price_data[tkn]) for tkn in assets])
        price_data = [
            {tkn: price_data[tkn][i] for tkn in assets} for i in range(data_length)
        ]
    return price_data


def import_monthly_binance_prices(
        assets: list[str], start_month: str, months: int, interval: int = 12, return_as_dict: bool = False
) -> dict[str: list[float]]:
    start_mth, start_year = start_month.split(' ')

    start_date = datetime.datetime.strptime(start_mth + ' 15 ' + start_year, "%b %d %Y")
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i * 30), "%Y-%m") for i in range(months)]

    # find the data folder
    while not os.path.exists("./data"):
        os.chdir("..")

    # check that the files are all there, and if not, download them
    for tkn in assets:
        for date in dates:
            file = f"{tkn}BUSD-1s-{date}"
            if os.path.exists(f'./data/{file}.csv'):
                continue
            else:
                print(f'Downloading {file}')
                url = f"https://data.binance.vision/data/spot/monthly/klines/{tkn}BUSD/1s/{file}.zip"
                response = requests.get(url)
                with open(f'./data/{file}.zip', 'wb') as f:
                    f.write(response.content)
                with ZipFile(f"./data/{file}.zip", 'r') as zipObj:
                    zipObj.extractall(path='./data')
                os.remove(f"./data/{file}.zip")
                # strip out everything except close price
                with open(f'./data/{file}.csv', 'r') as input_file:
                    rows = input_file.readlines()
                with open(f'./data/{file}.csv', 'w', newline='') as output_file:
                    output_file.write('\n'.join([row.split(',')[1] for row in rows]))

    # now that we have all the files, read them in
    price_data = {tkn: [] for tkn in assets}
    for tkn in assets:
        for date in dates:
            file = f"{tkn}BUSD-1s-{date}"
            with open(f'./data/{file}.csv', 'r') as input_file:
                csvreader = reader(input_file)
                price_data[tkn] += [float(row[0]) for row in csvreader][::interval]

    if not return_as_dict:
        price_data = [
            {tkn: price_data[tkn][i] for tkn in assets} for i in range(len(price_data[assets[0]]))
        ]
    return price_data


def convert_kraken_orderbook(x: dict) -> OrderBook:
    result = x['result']
    ks = list(result.keys())
    if len(ks) > 1:
        raise ValueError('Multiple keys in result')
    k = ks[0]
    ob = x['result'][k]

    ob_obj = OrderBook(
        bids=[[float(bid[0]), float(bid[1])] for bid in ob['bids']],
        asks=[[float(ask[0]), float(ask[1])] for ask in ob['asks']]
    )
    return ob_obj


def convert_binance_orderbook(tkn_pair: tuple, x: dict) -> OrderBook:
    orderbook = x

    ob_obj = OrderBook(
        bids=[[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
        asks=[[float(ask[0]), float(ask[1])] for ask in orderbook['asks']]
    )
    return ob_obj


def get_orderbooks_from_file(input_path: str) -> dict:
    file_ls = os.listdir(input_path)
    ob_dict = {'kraken': {}, 'binance': {}}
    for filename in file_ls:
        if filename.startswith('kraken_orderbook'):
            tkn_pair = tuple(filename.split('_')[2].split('-'))
            filepath = input_path + filename
            with open(filepath, newline='') as input_file:
                y = json.load(input_file)
                ob_dict['kraken'][tkn_pair] = convert_kraken_orderbook(y)
        elif filename.startswith('binance_orderbook'):
            tkn_pair = tuple(filename.split('_')[2].split('-'))
            filepath = input_path + filename
            with open(filepath, newline='') as input_file:
                y = json.load(input_file)
                ob_dict['binance'][tkn_pair] = convert_binance_orderbook(tkn_pair, y)

    return ob_dict


def get_kraken_orderbook(tkn_pair: tuple, archive: bool = False) -> OrderBook:
    orderbook_url = 'https://api.kraken.com/0/public/Depth?pair=' + tkn_pair[0] + tkn_pair[1]
    resp = requests.get(orderbook_url)
    y = resp.json()
    if 'error' in y and y['error']:
        print(y['error'])
    elif archive:
        ts = time.time()
        with open(f'./archive/kraken_orderbook_{tkn_pair[0]}-{tkn_pair[1]}_{ts}.json', 'w') as output_file:
            json.dump(y, output_file)
    return convert_kraken_orderbook(y)


def get_binance_orderbook(tkn_pair: tuple, archive: bool = False) -> OrderBook:
    orderbook_url = 'https://api.binance.com/api/v3/depth?symbol=' + tkn_pair[0] + tkn_pair[1]
    resp = requests.get(orderbook_url)
    y = resp.json()
    if 'msg' in y:
        print(y['msg'])
    elif archive:
        ts = time.time()
        with open(f'./archive/binance_orderbook_{tkn_pair[0]}-{tkn_pair[1]}_{ts}.json', 'w') as output_file:
            json.dump(y, output_file)
    return convert_binance_orderbook(tkn_pair, y)


def get_unique_name(ls: list[str], name: str) -> str:
    if name not in ls:
        return name
    else:
        c = 1
        while name + str(c).zfill(3) in ls:
            c += 1
        return name + str(c).zfill(3)


def get_omnipool_data(rpc: str = 'wss://rpc.hydradx.cloud', archive: bool = False):
    with HydraDX(rpc) as chain:

        asset_list = []
        fees = {}
        tokens = {}
        asset_map = {}

        op_state = chain.api.omnipool.state()

        for asset_id in op_state:

            fee = op_state[asset_id].fees
            decimals = op_state[asset_id].asset.decimals
            symbol = op_state[asset_id].asset.symbol

            tkn = get_unique_name(asset_list, symbol)
            asset_list.append(tkn)
            asset_map[asset_id] = tkn
            tokens[tkn] = {
                'liquidity': op_state[asset_id].reserve / 10 ** decimals,
                'LRNA': op_state[asset_id].hub_reserve / 10 ** 12
            }
            fees[tkn] = {"asset_fee": fee.asset_fee / 100, "protocol_fee": fee.protocol_fee / 100}

    if archive:
        ts = time.time()
        with open(f'./archive/omnipool_data_tokens_{ts}.json', 'w') as output_file:
            json.dump(tokens, output_file)
        with open(f'./archive/omnipool_data_fees_{ts}.json', 'w') as output_file:
            json.dump(fees, output_file)
        with open(f'./archive/omnipool_data_assetmap_{ts}.json', 'w') as output_file:
            json.dump(asset_map, output_file)

    return asset_list, asset_map, tokens, fees


def get_stableswap_data(rpc: str = 'wss://rpc.hydradx.cloud', archive: bool = False) -> list[StableSwapPoolState]:
    with HydraDX(rpc) as chain:
        pools = []
        data = chain.api.stableswap.pools()
        for pool_name, pool_data in {'4-Pool': data[100], '2-Pool': data[101]}.items():
            symbols = [asset.symbol for asset in pool_data.assets]
            repeats = [symbol for symbol in symbols if symbols.count(symbol) > 1]
            pools.append(StableSwapPoolState(
                tokens = {
                    f"{asset.symbol}{asset.asset_id}" if asset.symbol in repeats else asset.symbol:
                        int(pool_data.reserves[asset.asset_id]) / 10 ** asset.decimals
                    for asset in pool_data.assets
                },
                amplification=float(pool_data.final_amplification),
                trade_fee=float(pool_data.fee) / 100,
                unique_id=pool_name
            ))
    if archive:
        save_stableswap_data(pools)
    return pools


def save_stableswap_data(pools: list[StableSwapPoolState], path: str = './archive/'):
    ts = time.time()
    for state in pools:
        json_state = {
            'tokens': state.liquidity,
            'amplification': state.amplification,
            'precision': state.precision,
            'trade_fee': state.trade_fee
        }
        with open(f'{path}stableswap_data_{state.unique_id}_{ts}.json', 'w') as output_file:
            json.dump(json_state, output_file)


def load_stableswap_data(path: str = './archive/') -> list[StableSwapPoolState]:
    file_ls = os.listdir(path)
    pools = []
    pool_names = ['4-Pool', '2-Pool']
    for filename in file_ls:
        # return with the first likely-looking file
        if filename.startswith('stableswap_data'):
            pool_name = filename.split('_')[2]
            if pool_name in pool_names:
                pool_names.remove(pool_name)
                with open(path + filename, 'r') as input_file:
                    json_state = json.load(input_file)
                pools.append(StableSwapPoolState(
                    tokens=json_state['tokens'],
                    amplification=json_state['amplification'],
                    precision=json_state['precision'],
                    trade_fee=json_state['trade_fee']
                ))
        if len(pool_names) == 0:
            return pools
    raise FileNotFoundError(f'Stableswap data not found: {pool_names}')


def get_omnipool_data_from_file(path: str):
    file_ls = os.listdir(path)
    tokens = {}
    asset_map = {}
    fees = {}
    for filename in file_ls:
        if filename.startswith('omnipool_data'):
            if filename.split('_')[2] == 'tokens':
                with open(path + filename, newline='') as json_file:
                    tokens = json.load(json_file)
            elif filename.split('_')[2] == 'fees':
                with open(path + filename, newline='') as json_file:
                    fees = json.load(json_file)
            elif filename.split('_')[2] == 'assetmap':
                with open(path + filename, newline='') as json_file:
                    asset_map_str = json.load(json_file)
                    # print(asset_map_str)
                    asset_map = {int(k): v for k, v in asset_map_str.items()}

    asset_list = list(asset_map.values())
    return asset_list, asset_map, tokens, fees


def get_omnipool(rpc='wss://rpc.hydradx.cloud') -> OmnipoolState:
    with HydraDX(rpc) as chain:
        # get omnipool and subpool data
        op_state = chain.api.omnipool.state()
        sub_pools = chain.api.stableswap.pools()
        # collect assets
        assets = [(tkn.asset.asset_id, tkn.asset.symbol) for tkn in op_state.values()]
        sub_pool_assets = [(tkn.asset_id, tkn.symbol) for pool in sub_pools.values() for tkn in pool.assets]
        # get a unique symbol for each asset
        symbols = [asset[1] for asset in assets + sub_pool_assets]
        repeats = set([symbol for symbol in symbols if symbols.count(symbol) > 1])
        symbol_map = {
            tkn_id: f"{tkn_name}{tkn_id}" if tkn_name in repeats else tkn_name
            for tkn_id, tkn_name in assets + sub_pool_assets
        }
        omnipool = OmnipoolState(
            tokens={
                symbol_map[tkn_id]: {
                    'liquidity': int(op_state[tkn_id].reserve) / 10 ** op_state[tkn_id].asset.decimals,
                    'LRNA': int(op_state[tkn_id].hub_reserve) / 10 ** 12
                }
                for tkn_id in op_state.keys()  # base_tkn_ids
            },
            asset_fee={
                symbol_map[tkn_id]: op_state[tkn_id].fees.asset_fee / 100 if tkn_id in op_state else 0
                for tkn_id, tkn in [asset for asset in assets]
            },
            lrna_fee={
                symbol_map[tkn_id]: op_state[tkn_id].fees.protocol_fee / 100 if tkn_id in op_state else 0
                for tkn_id, tkn in [asset for asset in assets]
            },
            unique_id='omnipool'
            # preferred_stablecoin='USDT10'
        )
        for pool_id, pool_data in sub_pools.items():
            subpool = StableSwapPoolState(
                tokens={
                    symbol_map[asset.asset_id]: int(pool_data.reserves[asset.asset_id]) / 10 ** asset.decimals
                    for asset in pool_data.assets
                },
                amplification=float(pool_data.final_amplification),
                trade_fee=float(pool_data.fee) / 100,
                unique_id=symbol_map[pool_id]
            )
            omnipool.sub_pools[subpool.unique_id] = subpool

        return omnipool


def save_omnipool(omnipool: OmnipoolState, path: str = './archive'):
    ts = time.time()
    with open(os.path.join(path, f'omnipool_savefile_{ts}.json'), 'w+') as output_file:
        json.dump(
        {
                'liquidity': omnipool.liquidity,
                'LRNA': omnipool.lrna,
                'asset_fee': {tkn: (fee.fee if hasattr(fee, 'fee') else str(fee)) for tkn, fee in omnipool.asset_fee.items()},
                'lrna_fee': {tkn: (fee.fee if hasattr(fee, 'fee') else str(fee)) for tkn, fee in omnipool.lrna_fee.items()},
                'sub_pools': [
                    {
                        'tokens': pool.liquidity,
                        'amplification': pool.amplification,
                        'trade_fee': pool.trade_fee,
                        'unique_id': pool.unique_id
                    } for pool in omnipool.sub_pools.values()
                ]
            },
            output_file
        )


def load_omnipool(path: str = './archive', filename: str = '') -> OmnipoolState:
    if filename:
        file_ls = [filename]
    else:
        file_ls = list(filter(lambda file: file.startswith('omnipool_savefile'), os.listdir(path)))
    for filename in reversed(sorted(file_ls)):  # by default, load the latest first
        with open (os.path.join(path, filename), 'r') as input_file:
            json_state = json.load(input_file)
            # pprint(json_state)
        omnipool = OmnipoolState(
            tokens={
                tkn: {
                    'liquidity': json_state['liquidity'][tkn],
                    'LRNA': json_state['LRNA'][tkn]
                }
                for tkn in json_state['liquidity']
            },
            asset_fee={tkn: basic_fee(float(fee)) for tkn, fee in json_state['asset_fee'].items()},
            lrna_fee={tkn: basic_fee(float(fee)) for tkn, fee in json_state['lrna_fee'].items()}
        )
        for pool in json_state['sub_pools']:
            omnipool.sub_pools[pool['unique_id']] = StableSwapPoolState(
                tokens=pool['tokens'],
                amplification=pool['amplification'],
                trade_fee=float(pool['trade_fee']),
                unique_id=pool['unique_id']
            )
        return omnipool
    raise FileNotFoundError(f'Omnipool file not found in {path}.')


def get_centralized_market(
        config: list,
        exchange_name: str,
        trade_fee: float,
        archive: bool
) -> CentralizedMarket:

    order_books = {}
    for arb_cfg in config:
        exchanges = arb_cfg['exchanges'].keys()
        if exchange_name in exchanges:
            tkn_pair = tuple(arb_cfg['exchanges'][exchange_name])
            if tkn_pair not in order_books:
                if exchange_name == 'kraken':
                    order_books[tkn_pair] = get_kraken_orderbook(tkn_pair, archive=archive)
                elif exchange_name == 'binance':
                    order_books[tkn_pair] = get_binance_orderbook(tkn_pair, archive=archive)
                else:
                    raise ValueError(f"Exchange {exchange_name} not supported")

    return CentralizedMarket(
        unique_id=exchange_name,
        order_book=order_books,
        trade_fee=trade_fee
    )


def convert_config(cfg: list[dict]) -> list[dict]:
    """
    Convert the config from the format used in the UI to the format used in the backend.
    """

    asset_map = {
        100: '4-Pool',
        0: 'HDX',
        10: 'USDT',
        20: 'WETH',
        16: 'GLMR',
        11: 'iBTC',
        14: 'BNC',
        19: 'WBTC',
        15: 'vDOT',
        13: 'CFG',
        5: 'DOT',
        8: 'PHA',
        12: 'ZTG',
        17: 'INTR',
        9: 'ASTR'
    }
    # asset_map = get_omnipool_data("wss://rpc.hydradx.cloud")[1]

    return [
        {
            'exchanges': {
                'omnipool': tuple(asset_map[tkn_id] for tkn_id in cfg_item['tkn_ids']),
                cfg_item['exchange']: cfg_item['order_book']
            },
            'buffer': cfg_item['buffer']
        }
        for cfg_item in cfg
    ]

def load_config(filename, path='archive'):
    with open(os.path.join(path, filename), 'r') as input_file:
        config = json.load(input_file)
    for cfg_item in config:
        for exchange in cfg_item['exchanges']:
            cfg_item['exchanges'][exchange] = tuple(cfg_item['exchanges'][exchange])
    return config


def get_omnipool_balance_history():
    chunk_size = 10000
    chunks_per_file = 100

    load_dotenv()
    username = os.getenv('SQLPAD_USERNAME')
    password = os.getenv('PASSWORD')

    # Encode the username and password in Base64
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'  # This is typically required for JSON payloads
    }

    def insert_data_chunk(position: int, data_list: list):
        request = {
            'connectionId': "4a34594e-efa6-4f6e-a594-655ca20f2881",
            'batchText': (
                f"with hdx_changes as ("
                f"  select"
                f"    block_id,"
                f"    '0' as asset_id,"
                f"    (args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name like 'Balances.Transfer'"
                f"    and args->>'to' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"  union all"
                f"  select"
                f"    block_id,"
                f"    '0' as asset_id,"
                f"    -(args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name like 'Balances.Transfer'"
                f"    and args->>'from' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"),"
                f"tokens_changes as ("
                f"  select"
                f"    block_id,"
                f"    args->>'currencyId' as asset_id,"
                f"    (args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name = 'Tokens.Transfer'"
                f"    and args->>'to' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"  union all"
                f"  select"
                f"    block_id,"
                f"    args->>'currencyId' as asset_id,"
                f"    -(args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name = 'Tokens.Transfer'"
                f"    and args->>'from' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"  union all"
                f"  select"
                f"    block_id,"
                f"    args->>'currencyId' as asset_id,"
                f"    (args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name = 'Tokens.Deposited'"
                f"    and args->>'who' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"  union all"
                f"  select"
                f"    block_id,"
                f"    args->>'currencyId' as asset_id,"
                f"    -(args->>'amount')::numeric as amount"
                f"  from event"
                f"  where"
                f"    name = 'Tokens.Withdrawn'"
                f"    and args->>'who' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'"
                f"),"
                f"balance_changes as ("
                f"  select * from hdx_changes"
                f"  union all"
                f"  select * from tokens_changes"
                f"),"
                f"balance_history as ("
                f"  select"
                f"    height,"
                f"    timestamp,"
                f"    block_id,"
                f"    asset_id,"
                f"    symbol,"
                f"    sum(amount) over (partition by asset_id order by block_id) / 10 ^ decimals as balance"
                f"  from balance_changes"
                f"  inner join block on block_id = block.id"
                f"  inner join token_metadata on asset_id = token_metadata.id::text"
                f")"
                f"select timestamp, symbol, balance as liquidity "
                f"from balance_history "
                f"order by timestamp asc "
                f"limit {chunk_size} offset {chunk_size} * {position}"
            )
        }
        response = requests.post(
            url='https://sqlpad.play.hydration.cloud/api/batches',
            headers=headers,
            data=json.dumps(request)
        )

        # todo: this needs some work. It should use query_sqlPad,
        # and it needs to handle the end of the available data gracefully.

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            batchID = data['statements'][0]['batchId']
            try:
                statement = requests.get(
                    url=f'https://sqlpad.play.hydration.cloud/api/batches/{batchID}/statements',
                    headers=headers
                ).json()
                statementID = statement[0]['id']

                print(f'waiting for query page {position + 1}...')
                if response.status_code == 200:
                    # this is the response we get from the server if the query isn't finished yet.
                    # loop until we get a different response
                    response = {'title': 'Not found'}
                    while 'title' in response and response['title'] == 'Not found':
                        response = requests.get(
                            url=f'https://sqlpad.play.hydration.cloud/api/statements/{statementID}/results',
                            headers=headers
                        ).json()
                        time.sleep(1)
                    print("finished.")
                    response = list(response)
                    # tag a record number to each entry, so we can go back and see if anything is missing
                    new_data = []
                    for i in range(len(response)):
                        new_data.append([position * chunk_size + i] + response[i])
                    # insert at the correct position
                    data_list = data_list[:position * chunk_size] + new_data + data_list[position * chunk_size:]

            except Exception as e:
                print(f"There was a problem with your request: {str(e)}")
        else:
            pprint(response)
        return data_list

    def load_history_file(file_name: str):
        with open(f'./data/{file_name}', 'r') as file:
            file_data = json.loads('[' + file.read() + ']')
        return file_data

    def save_history_file(data_list: list, n: int):
        with open(f'./data/omnipool_history_{str(n).zfill(2)}', 'w') as file:
            file.write(', '.join([json.dumps(line) for line in
                                  data_list[chunk_size * chunks_per_file * (n - 1): chunk_size * chunks_per_file * n]]))
        print(f'Saved {filename}')

    def check_errors(data_list):
        errors = []
        for i in range(int(len(data_list) / chunk_size)):
            correctIndex = (i + len(errors)) * chunk_size
            if correctIndex >= len(data_list):
                break
            if data_list[i * chunk_size][0] != correctIndex:
                errors.append(i)
        return errors

    def fix_errors(data_list):
        # error checking and correction
        # this works in the specific case where a piece of chunk_size length failed to download, which is typical
        # other types of errors would require different handling
        errors = check_errors(data_list)
        if not errors:
            print('Data looks error-free.')
        while errors:
            print(f'Error detected at: {errors[0]}')
            data_list = insert_data_chunk(position=errors[0], data_list=data_list)
            errors = check_errors(data_list)

        return data_list

    # load what we have so far
    all_data = []
    file_ls = os.listdir('./data')
    for filename in file_ls:
        if filename.startswith('omnipool_history'):
            print(f'loading {filename}')
            all_data += load_history_file(filename)

    # continue downloading and check for errors
    while True:
        fix_errors(all_data)
        file_number = round(len(all_data) / chunk_size / chunks_per_file) + 1
        start_at = round(len(all_data) / chunk_size)
        for n in range(start_at, start_at + chunks_per_file):
            data_length = len(all_data)
            all_data = insert_data_chunk(position=n, data_list=all_data)
            print(data_length, len(all_data))
            if 0 < len(all_data) - data_length < chunk_size:
                # probably means we're finished. There might be a better way to detect this, but I think it'll do
                return all_data

        print(f'saving omnipool_history_{str(file_number).zfill(2)}')
        save_history_file(all_data, file_number)

async def query_sqlPad(query: str):
    """
    input: sql query string
    output: result of the query as a list of lists
    requirement: .env file with SQLPAD_USERNAME and PASSWORD in /model/ folder
    """

    load_dotenv()
    username = os.getenv('SQLPAD_USERNAME')
    password = os.getenv('PASSWORD')
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'  # This is typically required for JSON payloads
    }

    request = {
        'connectionId': "4a34594e-efa6-4f6e-a594-655ca20f2881",
        'batchText': query
    }

    try:
        response = requests.post(
            url='https://sqlpad.play.hydration.cloud/api/batches',
            headers=headers,
            data=json.dumps(request)
        )

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            batchID = data['statements'][0]['batchId']
            try:
                statement = requests.get(
                    url=f'https://sqlpad.play.hydration.cloud/api/batches/{batchID}/statements',
                    headers=headers
                ).json()
                statementID = statement[0]['id']

                print('waiting for query to finish...')
                if response.status_code == 200:
                    data = {'title': 'Not found'}
                    while 'title' in data and data['title'] == 'Not found':
                        data = requests.get(
                            url=f'https://sqlpad.play.hydration.cloud/api/statements/{statementID}/results',
                            headers=headers
                        ).json()
                    return data

            except Exception as e:
                print(f"There was a problem with your request: {str(e)}")
        else:
            print(f"Request failed with status code {response.status_code}")
            return []

    except Exception as e:
        print(f"There was a problem with your request: {str(e)}")


tkn_pair_abi = json.loads('[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"int24","name":"tickLower","type":"int24"},{"indexed":true,"internalType":"int24","name":"tickUpper","type":"int24"},{"indexed":false,"internalType":"uint128","name":"amount","type":"uint128"},{"indexed":false,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1","type":"uint256"}],"name":"Burn","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":false,"internalType":"address","name":"recipient","type":"address"},{"indexed":true,"internalType":"int24","name":"tickLower","type":"int24"},{"indexed":true,"internalType":"int24","name":"tickUpper","type":"int24"},{"indexed":false,"internalType":"uint128","name":"amount0","type":"uint128"},{"indexed":false,"internalType":"uint128","name":"amount1","type":"uint128"}],"name":"Collect","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"sender","type":"address"},{"indexed":true,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"uint128","name":"amount0","type":"uint128"},{"indexed":false,"internalType":"uint128","name":"amount1","type":"uint128"}],"name":"CollectProtocol","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"sender","type":"address"},{"indexed":true,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"paid0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"paid1","type":"uint256"}],"name":"Flash","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint16","name":"observationCardinalityNextOld","type":"uint16"},{"indexed":false,"internalType":"uint16","name":"observationCardinalityNextNew","type":"uint16"}],"name":"IncreaseObservationCardinalityNext","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"indexed":false,"internalType":"int24","name":"tick","type":"int24"}],"name":"Initialize","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"sender","type":"address"},{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"int24","name":"tickLower","type":"int24"},{"indexed":true,"internalType":"int24","name":"tickUpper","type":"int24"},{"indexed":false,"internalType":"uint128","name":"amount","type":"uint128"},{"indexed":false,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1","type":"uint256"}],"name":"Mint","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint8","name":"feeProtocol0Old","type":"uint8"},{"indexed":false,"internalType":"uint8","name":"feeProtocol1Old","type":"uint8"},{"indexed":false,"internalType":"uint8","name":"feeProtocol0New","type":"uint8"},{"indexed":false,"internalType":"uint8","name":"feeProtocol1New","type":"uint8"}],"name":"SetFeeProtocol","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"sender","type":"address"},{"indexed":true,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"int256","name":"amount0","type":"int256"},{"indexed":false,"internalType":"int256","name":"amount1","type":"int256"},{"indexed":false,"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"indexed":false,"internalType":"uint128","name":"liquidity","type":"uint128"},{"indexed":false,"internalType":"int24","name":"tick","type":"int24"}],"name":"Swap","type":"event"},{"inputs":[{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"amount","type":"uint128"}],"name":"burn","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"amount0Requested","type":"uint128"},{"internalType":"uint128","name":"amount1Requested","type":"uint128"}],"name":"collect","outputs":[{"internalType":"uint128","name":"amount0","type":"uint128"},{"internalType":"uint128","name":"amount1","type":"uint128"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint128","name":"amount0Requested","type":"uint128"},{"internalType":"uint128","name":"amount1Requested","type":"uint128"}],"name":"collectProtocol","outputs":[{"internalType":"uint128","name":"amount0","type":"uint128"},{"internalType":"uint128","name":"amount1","type":"uint128"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"factory","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"fee","outputs":[{"internalType":"uint24","name":"","type":"uint24"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"feeGrowthGlobal0X128","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"feeGrowthGlobal1X128","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"flash","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"}],"name":"increaseObservationCardinalityNext","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"}],"name":"initialize","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"liquidity","outputs":[{"internalType":"uint128","name":"","type":"uint128"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"maxLiquidityPerTick","outputs":[{"internalType":"uint128","name":"","type":"uint128"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"amount","type":"uint128"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"mint","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"observations","outputs":[{"internalType":"uint32","name":"blockTimestamp","type":"uint32"},{"internalType":"int56","name":"tickCumulative","type":"int56"},{"internalType":"uint160","name":"secondsPerLiquidityCumulativeX128","type":"uint160"},{"internalType":"bool","name":"initialized","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint32[]","name":"secondsAgos","type":"uint32[]"}],"name":"observe","outputs":[{"internalType":"int56[]","name":"tickCumulatives","type":"int56[]"},{"internalType":"uint160[]","name":"secondsPerLiquidityCumulativeX128s","type":"uint160[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"name":"positions","outputs":[{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},{"internalType":"uint128","name":"tokensOwed0","type":"uint128"},{"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"protocolFees","outputs":[{"internalType":"uint128","name":"token0","type":"uint128"},{"internalType":"uint128","name":"token1","type":"uint128"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint8","name":"feeProtocol0","type":"uint8"},{"internalType":"uint8","name":"feeProtocol1","type":"uint8"}],"name":"setFeeProtocol","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"slot0","outputs":[{"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"internalType":"int24","name":"tick","type":"int24"},{"internalType":"uint16","name":"observationIndex","type":"uint16"},{"internalType":"uint16","name":"observationCardinality","type":"uint16"},{"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},{"internalType":"uint8","name":"feeProtocol","type":"uint8"},{"internalType":"bool","name":"unlocked","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"}],"name":"snapshotCumulativesInside","outputs":[{"internalType":"int56","name":"tickCumulativeInside","type":"int56"},{"internalType":"uint160","name":"secondsPerLiquidityInsideX128","type":"uint160"},{"internalType":"uint32","name":"secondsInside","type":"uint32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"bool","name":"zeroForOne","type":"bool"},{"internalType":"int256","name":"amountSpecified","type":"int256"},{"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"swap","outputs":[{"internalType":"int256","name":"amount0","type":"int256"},{"internalType":"int256","name":"amount1","type":"int256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"int16","name":"","type":"int16"}],"name":"tickBitmap","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"tickSpacing","outputs":[{"internalType":"int24","name":"","type":"int24"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"int24","name":"","type":"int24"}],"name":"ticks","outputs":[{"internalType":"uint128","name":"liquidityGross","type":"uint128"},{"internalType":"int128","name":"liquidityNet","type":"int128"},{"internalType":"uint256","name":"feeGrowthOutside0X128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthOutside1X128","type":"uint256"},{"internalType":"int56","name":"tickCumulativeOutside","type":"int56"},{"internalType":"uint160","name":"secondsPerLiquidityOutsideX128","type":"uint160"},{"internalType":"uint32","name":"secondsOutside","type":"uint32"},{"internalType":"bool","name":"initialized","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"token0","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"token1","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}]')
erc20_abi = [{"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"}]

class UniswapToken:
    def __init__(self, symbol: str, address: str, w3: Web3):
        self.address = address
        self.symbol = symbol
        # print('Getting decimals for token: ', symbol)
        self.contract = w3.eth.contract(address=self.address, abi=erc20_abi)
        self.decimals = self.contract.functions.decimals().call()

    def __str__(self):
        return (
            f"symbol: {self.symbol}"
            f"address: ({self.address}"
            f"decimals: {self.decimals}"
        )

class UniswapPool:
    def __init__(self, tkn1: UniswapToken, tkn2: UniswapToken, fee: float, address, quoter_contract, w3: Web3):
        self.quoter_contract = quoter_contract
        self.fee = fee
        if tkn1.address.lower() < tkn2.address.lower():
            self.tkn1 = tkn1
            self.tkn2 = tkn2
        else:
            self.tkn1 = tkn2
            self.tkn2 = tkn1
        print('new pool: ', tkn1.symbol, tkn2.symbol)
        self.name = f"{tkn1.symbol}-{tkn2.symbol}"
        self.address = address
        print('pool address: ', self.address)
        self.contract = w3.eth.contract(address=self.address, abi=tkn_pair_abi)
        self.price96 = 0
        self.price = self.get_price()
        self.sqrt_price = self.price ** 0.5
        self.tick_spacing = self.contract.functions.tickSpacing().call()
        self.current_tick = self.get_active_tick()

    def get_price(self):
        slot0 = self.contract.functions.slot0().call()
        sqrt_price = slot0[0]
        self.price96 = sqrt_price
        price = (2 ** 192) / sqrt_price ** 2 # Square the sqrtPriceX96 and adjust for fixed point precision
        # account for the different decimals of the tokens
        # price = price / (10 ** (self.tkn1.decimals - self.tkn2.decimals))
        return price

    def get_active_tick(self):
        price_tick = self.contract.functions.slot0().call()[1]
        return price_tick // self.tick_spacing * self.tick_spacing

    def liquidity_at_tick(self, tick):
        return_val = self.contract.functions.ticks(tick).call()
        return return_val[1]

    def get_active_liquidity(self):
        return self.contract.functions.liquidity().call()

    def get_liquidity_distribution(self):
        """ get liquidity at the current tick and at 10 ticks above and below """
        starting_tick = self.get_active_tick()
        liquidity = {starting_tick: self.liquidity_at_tick(starting_tick)}
        for i in range(1, 10):
            tick = starting_tick + i * self.tick_spacing
            liquidity[tick] = self.liquidity_at_tick(tick)
            tick = starting_tick - i * self.tick_spacing
            liquidity[tick] = self.liquidity_at_tick(tick)
        return {k: v for k, v in sorted(liquidity.items())}

    def get_quote(
            self, tkn_sell: str = None, tkn_buy: str = None, sell_quantity: float = 0, buy_quantity: float = 0
    ) -> float:
        """ get quote for a token pair """
        if tkn_sell is None:
            tkn_sell = self.tkn1.symbol if tkn_buy == self.tkn2.symbol else self.tkn2.symbol
        elif tkn_buy is None:
            tkn_buy = self.tkn1.symbol if tkn_sell == self.tkn2.symbol else self.tkn2.symbol
        if tkn_sell not in (self.tkn1.symbol, self.tkn2.symbol):
            raise ValueError(f"Token {tkn_sell} not in pool")
        if tkn_buy not in (self.tkn1.symbol, self.tkn2.symbol):
            raise ValueError(f"Token {tkn_buy} not in pool")
        # sell_quantity *= 10 ** self.tkn1.decimals if tkn_sell == self.tkn1.symbol else 10 ** self.tkn2.decimals
        # buy_quantity *= 10 ** self.tkn1.decimals if tkn_buy == self.tkn1.symbol else 10 ** self.tkn2.decimals
        if sell_quantity:
            if tkn_sell == self.tkn1.symbol:
                buy_quantity = self.quoter_contract.functions.quoteExactInputSingle(
                    self.tkn1.address, self.tkn2.address, self.fee, sell_quantity, 0).call()
            else:
                buy_quantity = self.quoter_contract.functions.quoteExactInputSingle(
                    self.tkn2.address, self.tkn1.address, self.fee, sell_quantity, 0).call()
            # buy_quantity /= 10 ** self.tkn1.decimals if tkn_buy == self.tkn1.symbol else 10 ** self.tkn2.decimals
            return buy_quantity
        elif buy_quantity:
            if tkn_sell == self.tkn1.symbol:
                sell_quantity = self.quoter_contract.functions.quoteExactOutputSingle(
                    self.tkn1.address, self.tkn2.address, self.fee, buy_quantity, 0).call()
            else:
                sell_quantity = self.quoter_contract.functions.quoteExactOutputSingle(
                    self.tkn2.address, self.tkn1.address, self.fee, buy_quantity, 0).call()
            # sell_quantity /= 10 ** self.tkn1.decimals if tkn_sell == self.tkn1.symbol else 10 ** self.tkn2.decimals
            return sell_quantity
        return 0.0


def get_uniswap_pool_data(tkn_pairs: list[tuple]) -> dict[str, UniswapPool]:
    """ get pool data from uniswap """
    # Connect to an Ethereum node
    provider = 'https://eth-mainnet.g.alchemy.com/v2/wWrLtJw3ZHgitVEI0hOcrQ5UxAuemN6f'
    # provider = 'https://moonbeam-rpc.dwellir.com'
    w3 = Web3(Web3.HTTPProvider(provider))

    # Uniswap V3 Pool address and ABI
    uniswap_pool_address = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    # stellaswap_factory_address = "0xabE1655110112D0E45EF91e94f8d757e4ddBA59C"
    uniswap_pool_abi = json.loads('[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint24","name":"fee","type":"uint24"},{"indexed":true,"internalType":"int24","name":"tickSpacing","type":"int24"}],"name":"FeeAmountEnabled","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"oldOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnerChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"token0","type":"address"},{"indexed":true,"internalType":"address","name":"token1","type":"address"},{"indexed":true,"internalType":"uint24","name":"fee","type":"uint24"},{"indexed":false,"internalType":"int24","name":"tickSpacing","type":"int24"},{"indexed":false,"internalType":"address","name":"pool","type":"address"}],"name":"PoolCreated","type":"event"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"}],"name":"createPool","outputs":[{"internalType":"address","name":"pool","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickSpacing","type":"int24"}],"name":"enableFeeAmount","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint24","name":"","type":"uint24"}],"name":"feeAmountTickSpacing","outputs":[{"internalType":"int24","name":"","type":"int24"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint24","name":"","type":"uint24"}],"name":"getPool","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"parameters","outputs":[{"internalType":"address","name":"factory","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickSpacing","type":"int24"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"_owner","type":"address"}],"name":"setOwner","outputs":[],"stateMutability":"nonpayable","type":"function"}]')
    uniswap_pool_contract = w3.eth.contract(address=uniswap_pool_address, abi=uniswap_pool_abi)

    uniswap_quoter_address = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6'
    uniswap_quoter_abi = [{"inputs": [{"internalType": "address", "name": "_factory", "type": "address"}, {"internalType": "address", "name": "_WETH9", "type": "address"}],"stateMutability": "nonpayable", "type": "constructor"}, {"inputs": [], "name": "WETH9","outputs": [{"internalType": "address","name": "","type": "address"}],"stateMutability": "view","type": "function"},{"inputs": [], "name": "factory","outputs": [{"internalType": "address", "name": "", "type": "address"}],"stateMutability": "view", "type": "function"}, {"inputs": [{"internalType": "bytes", "name": "path", "type": "bytes"},{"internalType": "uint256", "name": "amountIn", "type": "uint256"}],"name": "quoteExactInput","outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],"stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "tokenIn", "type": "address"},{"internalType": "address", "name": "tokenOut", "type": "address"},{"internalType": "uint24", "name": "fee", "type": "uint24"},{"internalType": "uint256", "name": "amountIn", "type": "uint256"},{"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}],"name": "quoteExactInputSingle","outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],"stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "bytes", "name": "path", "type": "bytes"},{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],"name": "quoteExactOutput","outputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}],"stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "address", "name": "tokenIn", "type": "address"},{"internalType": "address", "name": "tokenOut", "type": "address"}, {"internalType": "uint24", "name": "fee", "type": "uint24"}, {"internalType": "uint256", "name": "amountOut", "type": "uint256"}, {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}], "name": "quoteExactOutputSingle", "outputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}], "stateMutability": "nonpayable", "type": "function"}, {"inputs": [{"internalType": "int256", "name": "amount0Delta", "type": "int256"}, {"internalType": "int256", "name": "amount1Delta", "type": "int256"}, {"internalType": "bytes", "name": "path", "type": "bytes"}], "name": "uniswapV3SwapCallback", "outputs": [], "stateMutability": "view", "type": "function"}]
    quoter_contract = w3.eth.contract(address=uniswap_quoter_address, abi=uniswap_quoter_abi)

    tkn_addr = {
        "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "dai": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "eth": "0x0000000000000000000000000000000000000000",
        "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "bat": "0x0D8775F648430679A709E98d2b0Cb6250d2887EF",
    }

    # create token objects
    tokens = {
        symbol: UniswapToken(symbol, tkn_addr[symbol], w3)
        for symbol in {tkn for pair in tkn_pairs for tkn in pair}
    }

    # get all available pools for each token pair at every fee level
    fee_levels = [100, 500, 3000, 10000]
    pools = {}
    for tkn1, tkn2, fee in [(tkn1, tkn2, fee) for tkn1, tkn2 in tkn_pairs for fee in fee_levels]:
        pool_address = uniswap_pool_contract.functions.getPool(tokens[tkn1].address, tokens[tkn2].address, fee).call()
        if int(pool_address, 16) > 0:
            pools[f"{tkn1}-{tkn2}-{fee}"] = UniswapPool(
                tokens[tkn1], tokens[tkn2], fee, pool_address, quoter_contract, w3
            )

    return pools