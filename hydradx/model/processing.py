import json
from csv import reader
import requests
from zipfile import ZipFile
import datetime
import os
from hydradxapi import HydraDX
import time

from .amm.centralized_market import OrderBook
from .amm.global_state import GlobalState, AMM, value_assets

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
    if archive:
        ts = time.time()
        with open(f'./archive/kraken_orderbook_{tkn_pair[0]}-{tkn_pair[1]}_{ts}.json', 'w') as output_file:
            json.dump(y, output_file)
    return convert_kraken_orderbook(y)


def get_binance_orderbook(tkn_pair: tuple, archive: bool = False) -> OrderBook:
    orderbook_url = 'https://api.binance.com/api/v3/depth?symbol=' + tkn_pair[0] + tkn_pair[1]
    resp = requests.get(orderbook_url)
    y = resp.json()
    if archive:
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


def get_omnipool_data(rpc: str, archive: bool = False):
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
                    asset_map = {int(k): v for k, v in asset_map_str.items()}

    asset_list = list(asset_map.values())
    return asset_list, asset_map, tokens, fees


# def import_prices(input_path: str, input_filename: str) -> list[PriceTick]:
#     price_data = []
#     with open(input_path + input_filename, newline='') as input_file:
#         fieldnames = ['timestamp', 'price']
#         reader = DictReader(input_file, fieldnames=fieldnames)
#         next(reader)  # skip header
#         for row in reader:
#             price_data.append(PriceTick(int(row["timestamp"]), float(row["price"])))
#
#     price_data.sort(key=lambda x: x.timestamp)
#     return price_data
