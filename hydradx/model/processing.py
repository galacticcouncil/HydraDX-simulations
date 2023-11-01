from csv import reader
import requests
from zipfile import ZipFile
import datetime
import os
from hydradxapi import HydraDX
import json

from .amm.centralized_market import OrderBook, CentralizedMarket
from .amm.omnipool_amm import OmnipoolState
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


def get_kraken_orderbook(tkn_pair: tuple, orderbook_url: str) -> OrderBook:
    resp = requests.get(orderbook_url)
    y = resp.json()
    orderbook = y['result'][tkn_pair[0] + tkn_pair[1]]

    ob_obj = OrderBook(
        bids=[[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
        asks=[[float(ask[0]), float(ask[1])] for ask in orderbook['asks']]
    )
    return ob_obj


def get_unique_name(ls: list[str], name: str) -> str:
    if name not in ls:
        return name
    else:
        c = 1
        while name + str(c).zfill(3) in ls:
            c += 1
        return name + str(c).zfill(3)


def get_omnipool_data(rpc: str, n: int):
    with HydraDX(rpc) as chain:

        asset_list = []
        fees = {}
        tokens = {}
        asset_map = {}

        for i in range(n):
            try:
                md = chain.api.registry.asset_metadata(i)
                state = chain.api.omnipool.asset_state(md.asset_id)
                fee = chain.api.fees.asset_fees(md.asset_id)

            except:
                continue

            tkn = get_unique_name(asset_list, md.symbol)
            asset_list.append(tkn)
            asset_map[i] = tkn
            tokens[tkn] = {
                'liquidity': state.reserve / 10 ** md.decimals,
                'LRNA': state.hub_reserve / 10 ** 12
            }
            fees[tkn] = {"asset_fee": fee.asset_fee / 100, "protocol_fee": fee.protocol_fee / 100}

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


def save_market_config():
    asset_list, asset_map, tokens, fees = get_omnipool_data("wss://hydradx-rpc.dwellir.com", 24)
    lrna_fee = {asset: fees[asset]['protocol_fee'] for asset in asset_list}
    asset_fee = {asset: fees[asset]['asset_fee'] for asset in asset_list}

    arb_list = [
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 10), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 23), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 2), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 7), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 18), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 21), "order_book": ("HDX", "USD")},
        {"tkns": ("HDX", "USDT"), "tkn_ids": (0, 22), "order_book": ("HDX", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 10), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 23), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 2), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 7), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 18), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 21), "order_book": ("DOT", "USD")},
        {"tkns": ("DOT", "USDT"), "tkn_ids": (5, 22), "order_book": ("DOT", "USD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 10), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 23), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 2), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 7), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 18), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 21), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (4, 22), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 10), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 23), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 2), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 7), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 18), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 21), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("WETH", "USDT"), "tkn_ids": (20, 22), "order_book": ("XETH", "ZUSD")},
        {"tkns": ("DOT", "WETH"), "tkn_ids": (5, 4), "order_book": ("DOT", "ETH")},
        {"tkns": ("DOT", "WETH"), "tkn_ids": (5, 20), "order_book": ("DOT", "ETH")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 10), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 23), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 2), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 7), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 18), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 21), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("WBTC", "USDT"), "tkn_ids": (19, 22), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 10), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 23), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 2), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 7), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 18), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 21), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("IBTC", "USDT"), "tkn_ids": (11, 22), "order_book": ("XXBT", "ZUSD")},
        {"tkns": ("DOT", "WBTC"), "tkn_ids": (5, 19), "order_book": ("DOT", "XBT")},
        {"tkns": ("DOT", "IBTC"), "tkn_ids": (5, 11), "order_book": ("DOT", "XBT")},
        {"tkns": ("WETH", "WBTC"), "tkn_ids": (4, 19), "order_book": ("XETH", "XXBT")},
        {"tkns": ("WETH", "IBTC"), "tkn_ids": (4, 11), "order_book": ("XETH", "XXBT")},
        {"tkns": ("WETH", "WBTC"), "tkn_ids": (20, 19), "order_book": ("XETH", "XXBT")},
        {"tkns": ("WETH", "IBTC"), "tkn_ids": (20, 11), "order_book": ("XETH", "XXBT")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 10), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 23), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 2), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 7), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 18), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 21), "order_book": ("ASTR", "USD")},
        {"tkns": ("ASTR", "USDT"), "tkn_ids": (9, 22), "order_book": ("ASTR", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 10), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 23), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 2), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 7), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 18), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 21), "order_book": ("CFG", "USD")},
        {"tkns": ("CFG", "USDT"), "tkn_ids": (13, 22), "order_book": ("CFG", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 10), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 23), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 2), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 7), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 18), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 21), "order_book": ("BNC", "USD")},
        {"tkns": ("BNC", "USDT"), "tkn_ids": (14, 22), "order_book": ("BNC", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 10), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 23), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 2), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 7), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 18), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 21), "order_book": ("GLMR", "USD")},
        {"tkns": ("GLMR", "USDT"), "tkn_ids": (16, 22), "order_book": ("GLMR", "USD")}
    ]

    ob_objs = {}
    order_book_asset_list = []

    for arb_cfg in arb_list:
        tkn_pair = arb_cfg['order_book']
        if tkn_pair not in ob_objs:
            order_book_url = f'https://api.kraken.com/0/public/Depth?pair={tkn_pair[0]}{tkn_pair[1]}'
            ob_objs[tkn_pair] = get_kraken_orderbook(tkn_pair, order_book_url)
            for tkn in tkn_pair:
                if tkn not in order_book_asset_list:
                    order_book_asset_list.append(tkn)

    cex_fee = 0.0016
    # buffer = 0.0010

    order_book_map = {}
    for i in range(len(arb_list)):
        base_id, quote_id = arb_list[i]['tkn_ids']
        orderbook_tkn_pair = arb_list[i]['order_book']
        if base_id in asset_map and quote_id in asset_map:
            tkn_pair = (asset_map[base_id], asset_map[quote_id])
            order_book_map[f"{tkn_pair[0]}, {tkn_pair[1]}"] = orderbook_tkn_pair

    save_data = {
        'tokens': tokens,
        'lrna_fees': lrna_fee,
        'asset_fees': asset_fee,
        'order_books': {
            f"{tkn_pair[0]}, {tkn_pair[1]}": {'bids': orderbook.bids, 'asks': orderbook.asks}
            for tkn_pair, orderbook in ob_objs.items()
        },
        'order_book_map': order_book_map,
        'cex_fee': cex_fee,
    }
    with open('./config.txt', 'w') as outfile:
        json.dump(save_data, outfile)


def load_market_config() -> (OmnipoolState, CentralizedMarket, dict):
    with open('./config.txt', 'r') as openfile:
        data = json.load(openfile)

    tokens = data['tokens']
    asset_fee = data['asset_fees']
    lrna_fee = data['lrna_fees']
    order_books = {
        tuple([tkn for tkn in tkn_pair.split(', ')]): OrderBook(
            bids=[[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
            asks=[[float(ask[0]), float(ask[1])] for ask in orderbook['asks']]
        )
        for tkn_pair, orderbook in data['order_books'].items()
    }
    order_book_map = {
        tuple([tkn for tkn in pair1.split(', ')]): tuple(pair2)
        for pair1, pair2 in data['order_book_map'].items()
    }
    cex_fee = data['cex_fee']
    return (
        OmnipoolState(
            tokens=tokens,
            asset_fee=asset_fee,
            lrna_fee=lrna_fee,
            preferred_stablecoin='USDT',
        ),
        CentralizedMarket(
            order_book=order_books,
            trade_fee=cex_fee,
        ),
        order_book_map
    )
