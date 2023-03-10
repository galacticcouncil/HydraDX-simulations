import csv, json
from csv import DictReader, writer
from dataclasses import dataclass

from .amm.global_state import GlobalState, withdraw_all_liquidity, AMM
from .amm.agents import Agent

cash_out = GlobalState.cash_out
value_assets = GlobalState.value_assets
impermanent_loss = GlobalState.impermanent_loss
pool_val = GlobalState.pool_val
deposit_val = GlobalState.deposit_val
market_prices = GlobalState.market_prices

TOKEN_METADATA = [
    ('HDX', 12),
    ('LRNA', 12),
    ('DAI', 18),
    ('WBTC', 8),
    ('WETH', 18),
    ('DOT', 10),
    ('APE', 18),
    ('USDC', 6),
    ('PHA', 12),
]
LRNA_I = 1


def postprocessing(events: list[dict], optional_params: list[str] = ()) -> list[dict]:
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
    initial_state: GlobalState = events[0]['state']
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

    # print(f'processing {optional_params}')
    #
    # a little pre-processing
    if 'deposit_val' in optional_params:
        # move the agents' liquidity deposits back into holdings, as something to compare against later
        for agent_id in initial_state.agents:
            # do it this convoluted way because we're pretending each agent withdrew their assets alone,
            # isolated from any effects of the other agents withdrawing *their* assets
            withdraw_state.agents[agent_id] = withdraw_all_liquidity(initial_state.copy(), agent_id).agents[agent_id]

    for step in events:
        state: GlobalState = step['state']

        for pool in state.pools.values():
            if 'pool_val' in optional_params:
                pool.pool_val = state.pool_val(pool)
            # if 'usd_price' in optional_params:
            #     pool.usd_price = {tkn: pool.price(tkn) for tkn in pool.asset_list}

        # agents
        for agent in state.agents.values():
            if 'deposit_val' in optional_params:
                # what are this agent's original holdings theoretically worth at current spot prices?
                agent.deposit_val = state.value_assets(
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
                    previous_agent = events[state.time_step - 1]['state'].agents[agent.unique_id]
                    agent.trade_volume += (
                        sum([
                            abs(previous_agent.holdings[tkn] - agent.holdings[tkn]) * state.price(tkn)
                            for tkn in agent.holdings])
                    )

    return events


@dataclass
class PriceTick:
    timestamp: int
    price: float


def write_price_data(price_data: list[PriceTick], output_filename: str) -> None:
    with open(output_filename, 'w', newline='') as output_file:
        fieldnames = ['timestamp', 'price']
        csvwriter = writer(output_file)
        csvwriter.writerow(fieldnames)
        for row in price_data:
            csvwriter.writerow([row.timestamp, row.price])


def import_price_data(input_filename: str) -> list[PriceTick]:
    price_data = []
    with open(input_filename, newline='') as input_file:
        reader = DictReader(input_file)
        for row in reader:
            price_data.append(PriceTick(int(row["timestamp"]), float(row["price"])))
    return price_data


def import_binance_prices(input_path: str, input_filenames: list[str]) -> list[PriceTick]:
    price_data = []
    for input_filename in input_filenames:
        with open(input_path + input_filename, newline='') as input_file:
            fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            reader = DictReader(input_file, fieldnames=fieldnames)
            # reader = DictReader(input_file)
            for row in reader:
                price_data.append(PriceTick(int(row["timestamp"]), float(row["open"])))

    price_data.sort(key=lambda x: x.timestamp)
    return price_data


def read_csvs_to_list_of_dicts(paths: list[str]) -> list:
    dicts = []
    for path in paths:
        with open(path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            dicts.extend(list(reader))
    return dicts


def import_trades(paths: list[str]) -> list[dict]:
    rows = read_csvs_to_list_of_dicts(paths)

    token_names = [token[0] for token in TOKEN_METADATA]
    token_decimals = [token[1] for token in TOKEN_METADATA]

    trades = []
    for row in rows:
        name = row['call_name'].split('.')[1]
        if row['asset_out'] == '':
            raise
        out_i = int(row['asset_out'])
        in_i = int(row['asset_in'])
        limit_decimals = token_decimals[out_i] if name == 'sell' else token_decimals[in_i]
        trade_dict = {
            'id': row['extrinsic_id'],
            'block_no': int(row['extrinsic_id'].split('-')[0]),
            'tx_no': int(row['extrinsic_id'].split('-')[1]),
            'type': name,
            'success': bool(row['call_success']),
            'who': row['who'],
            'fee': int(row['fee']),
            'tip': int(row['tip']),
            'asset_in': token_names[in_i],
            'asset_out': token_names[out_i],
            'amount_in': int(row['amount_in']) / 10 ** token_decimals[in_i],
            'amount_out': int(row['amount_out']) / 10 ** token_decimals[out_i],
            'limit_amount': int(row['limit_amount']) / 10 ** limit_decimals,
        }
        trades.append(trade_dict)
    return trades


def import_add_liq(paths: list[str]) -> list[dict]:
    rows = read_csvs_to_list_of_dicts(paths)

    token_names = [token[0] for token in TOKEN_METADATA]
    token_decimals = [token[1] for token in TOKEN_METADATA]

    txs = []
    for row in rows:
        tx_dict = {
            'id': row['extrinsic_id'],
            'block_no': int(row['extrinsic_id'].split('-')[0]),
            'tx_no': int(row['extrinsic_id'].split('-')[1]),
            'type': row['call_name'].split('.')[1],
            'success': bool(row['call_success']),
            'who': row['who'],
            'fee': int(row['fee']),
            'tip': int(row['tip']),
            'asset': token_names[int(row['asset'])],
            'amount': int(row['amount']) / 10 ** token_decimals[int(row['asset'])],
            'price': int(row['price']) / 10 ** (18 + token_decimals[LRNA_I] - token_decimals[int(row['asset'])]),
            'shares': int(row['shares']) / 10 ** token_decimals[int(row['asset'])],
            'position_id': row['position_id'],
        }
        txs.append(tx_dict)
    return txs


def import_remove_liq(paths: list[str]) -> list[dict]:
    rows = read_csvs_to_list_of_dicts(paths)

    token_names = [token[0] for token in TOKEN_METADATA]
    token_decimals = [token[1] for token in TOKEN_METADATA]

    txs = []
    for row in rows:
        tx_dict = {
            'id': row['extrinsic_id'],
            'block_no': int(row['extrinsic_id'].split('-')[0]),
            'tx_no': int(row['extrinsic_id'].split('-')[1]),
            'type': row['call_name'].split('.')[1],
            'success': bool(row['call_success']),
            'who': row['who'],
            'fee': int(row['fee']),
            'tip': int(row['tip']),
            'asset': token_names[int(row['asset'])],
            'amount': int(row['amount']) / 10 ** token_decimals[int(row['asset'])],
            'lrna_amount': int(row['lrna_amount']) / 10 ** token_decimals[LRNA_I] if row['lrna_amount'] != '' else 0,
            'shares': int(row['shares']) / 10 ** token_decimals[int(row['asset'])],
            'position_id': row['position_id'],
        }
        txs.append(tx_dict)
    return txs


def import_tx_data():
    trade_paths = ['input/trades.csv']
    liq_add_paths = ['input/add_liquidity.csv']
    liq_rem_paths = ['input/remove_liquidity.csv']

    trades = import_trades(trade_paths)
    liq_adds = import_add_liq(liq_add_paths)
    liq_rems = import_remove_liq(liq_rem_paths)

    txs = trades + liq_adds + liq_rems
    txs.sort(key=lambda x: (x['block_no'], x['tx_no']))

    return txs
