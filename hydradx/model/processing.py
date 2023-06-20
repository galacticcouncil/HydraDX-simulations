import pickle
from csv import DictReader, writer, reader
from dataclasses import dataclass
import requests
from zipfile import ZipFile
import datetime
import os

from .amm.global_state import GlobalState, withdraw_all_liquidity, AMM, value_assets
# from .amm.agents import Agent

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

    # a little pre-processing
    if 'deposit_val' in optional_params:
        # move the agents' liquidity deposits back into holdings, as something to compare against later
        for agent_id in initial_state.agents:
            # do it this convoluted way because we're pretending each agent withdrew their assets alone,
            # isolated from any effects of the other agents withdrawing *their* assets
            withdraw_state.agents[agent_id] = withdraw_all_liquidity(initial_state.copy(), agent_id).agents[agent_id]

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
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i), ("%Y-%m-%d")) for i in range(days)]

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

    data_length = min([len(price_data[tkn]) for tkn in assets])
    if not return_as_dict:
        data_length = min([len(price_data[tkn]) for tkn in assets])
        price_data = [
            {tkn: price_data[tkn][i] for tkn in assets} for i in range(data_length)
        ]
    return price_data


def import_monthly_binance_prices(
        assets: list[str], start_month: str, months: int, interval: int = 12,
        stablecoin: str = 'USDT', return_as_dict: bool = False
) -> dict[str: list[float]]:
    start_mth, start_year = start_month.split(' ')

    start_date = datetime.datetime.strptime(start_mth + ' 15 ' + start_year, "%b %d %Y")
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i * 30), ("%Y-%m")) for i in range(months)]

    # find the data folder
    while not os.path.exists("./data"):
        os.chdir("..")

    # check that the files are all there, and if not, download them
    for tkn in assets:
        for date in dates:
            file = f"{stablecoin}{tkn}-1s-{date}"
            if os.path.exists(f'./data/{file}.csv'):
                continue
            else:
                print(f'Downloading {file}')
                url = f"https://data.binance.vision/data/spot/monthly/klines/{tkn}{stablecoin}/1s/{file}.zip"
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


def save_output(data: object, filename: str, folder: str = 'output') -> None:
    """Save the output data to a file.

    Args:
        data (object): The data to be saved.
        filename (str): The name of the file to be saved.
        folder (str, optional): The name of the folder to save the file in. Defaults to 'output'.
    """
    if not os.path.exists(f'./{folder}'):
        os.mkdir(f'./{folder}')
    with open(f'./{folder}/{filename}.pickle', 'wb') as f:
        pickle.dump(data, f)


def load_output(filename: str, folder: str = 'output') -> object:
    """Load the output data from a file.

    Args:
        filename (str): The name of the file to be loaded.
        folder (str, optional): The name of the folder to load the file from. Defaults to 'output'.

    Returns:
        object: The data that was saved in the file.
    """
    with open(f'./{folder}/{filename}.pickle', 'rb') as f:
        data = pickle.load(f)
    return data