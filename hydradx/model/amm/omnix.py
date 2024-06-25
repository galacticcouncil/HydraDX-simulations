from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
import copy
from mpmath import mp, mpf


def validate_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    return validate_and_execute_solution(
        omnipool=omnipool.copy(),
        intents=copy.deepcopy(intents),
        amt_processed=amt_processed,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        tolerance=tolerance
    )


def calculate_transfers(
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict  # key: token, value: price
) -> (list, dict):
    transfers = []
    deltas = {tkn: {"in": 0, "out": 0} for tkn in buy_prices}
    for i in range(len(intents)):
        intent = intents[i]
        amt = amt_processed[i]
        if 'sell_quantity' in intent:
            buy_amt = amt * sell_prices[intent['tkn_sell']] / buy_prices[intent['tkn_buy']]
            transfer = {
                'agent': intent['agent'],
                'buy_quantity': buy_amt,
                'sell_quantity': amt,
                'tkn_buy': intent['tkn_buy'],
                'tkn_sell': intent['tkn_sell']
            }
            transfers.append(transfer)
            deltas[intent['tkn_buy']]["out"] += buy_amt
            deltas[intent['tkn_sell']]["in"] += amt
        elif 'buy_quantity' in intent:
            sell_amt = amt * buy_prices[intent['tkn_buy']] / sell_prices[intent['tkn_sell']]
            transfer = {
                'agent': intent['agent'],
                'buy_quantity': amt,
                'sell_quantity': sell_amt,
                'tkn_buy': intent['tkn_buy'],
                'tkn_sell': intent['tkn_sell']
            }
            transfers.append(transfer)
            deltas[intent['tkn_buy']]["out"] += amt
            deltas[intent['tkn_sell']]["in"] += sell_amt

    return transfers, deltas


def validate_and_execute_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    assert buy_prices.keys() == sell_prices.keys(), "buy_prices and sell_prices are not provided for same tokens"

    validate_intents(intents, amt_processed)
    transfers, deltas = calculate_transfers(intents, amt_processed, buy_prices, sell_prices)
    validate_transfer_amounts(transfers)
    pool_agent, lrna_deltas = execute_solution(omnipool, transfers, deltas)
    if not validate_remainder(pool_agent):
        raise Exception("agent has negative holdings")
    if not validate_prices(omnipool, deltas, lrna_deltas, buy_prices, sell_prices, tolerance):
        raise Exception("prices are not valid")

    update_intents(intents, transfers)

    return True


def validate_intents(intents: list, amt_processed: list):
    for i in range(len(intents)):
        intent = intents[i]
        amt = amt_processed[i]
        if 'sell_quantity' in intent:
            assert "buy_limit" in intent, "buy_limit not in sell intent"
            assert "buy_quantity" not in intent, "intent has both buy_quantity and sell_quantity"
            assert "sell_limit" not in intent, "intent has both buy_quantity and sell_limit"
            assert 0 <= amt <= intent['sell_quantity'], "amt processed is not in range"
        elif 'buy_quantity' in intent:
            assert "sell_limit" in intent, "sell_limit not in buy intent"
            assert "buy_limit" not in intent, "intent has both sell_quantity and buy_limit"
            assert 0 <= amt <= intent['buy_quantity'], "amt processed is not in range"
        else:
            raise Exception("intent does not have sell_quantity or buy_quantity")

    return True


def validate_transfer_amounts(transfers: list):
    for transfer in transfers:
        if not transfer['agent'].is_holding(transfer['tkn_sell'], transfer['sell_quantity']):
            raise Exception(f"agent does not have enough holdings in token {transfer['tkn_sell']}")

    return True


def execute_solution(
        omnipool: OmnipoolState,
        transfers: list,
        deltas: dict  # note that net_deltas can be easily reconstructed from transfers
):
    pool_agent = Agent()

    # transfer assets in from agents whose intents are being executed
    for transfer in transfers:
        if transfer['tkn_sell'] not in pool_agent.holdings:
            pool_agent.holdings[transfer['tkn_sell']] = 0
        pool_agent.holdings[transfer['tkn_sell']] += transfer['sell_quantity']
        transfer['agent'].holdings[transfer['tkn_sell']] -= transfer['sell_quantity']

    # execute swaps against Omnipool
    lrna_deltas = {tkn: 0 for tkn in deltas}
    for tkn in deltas:  # first do sells to accumulate LRNA
        if deltas[tkn]["in"] > deltas[tkn]["out"]:
            init_lrna = omnipool.lrna[tkn]
            omnipool.lrna_swap(pool_agent, delta_ra=deltas[tkn]["out"] - deltas[tkn]["in"], tkn=tkn)
            lrna_deltas[tkn] = omnipool.lrna[tkn] - init_lrna
    for tkn in deltas:  # next sell LRNA back for tokens
        if deltas[tkn]["in"] < deltas[tkn]["out"]:
            init_lrna = omnipool.lrna[tkn]
            omnipool.lrna_swap(pool_agent, delta_ra=deltas[tkn]["out"] - deltas[tkn]["in"], tkn=tkn)
            lrna_deltas[tkn] = omnipool.lrna[tkn] - init_lrna

    # transfer assets out to intent agents
    for transfer in transfers:
        pool_agent.holdings[transfer['tkn_buy']] -= transfer['buy_quantity']
        if transfer['tkn_buy'] not in transfer['agent'].holdings:
            transfer['agent'].holdings[transfer['tkn_buy']] = 0
        transfer['agent'].holdings[transfer['tkn_buy']] += transfer['buy_quantity']

    return pool_agent, lrna_deltas


def validate_remainder(pool_agent: Agent):
    for tkn, amt in pool_agent.holdings.items():
        if amt < 0:
            # return False
            raise
    return True


def validate_prices(
        omnipool: OmnipoolState,
        deltas: dict,  # key: token, value: delta_change of token w.r.t. Omnipool
        lrna_deltas: dict, # key: token, value: delta_change of LRNA w.r.t. Omnipool in tkn trade
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    for tkn, delta in deltas.items():  # calculate net of Omnipool leg and other leg in same direction
        if delta["out"] > delta["in"]:  # LRNA being sold to Omnipool for TKN
            agent_sell_amt = delta['in']
            omnipool_sell_amt = delta["out"] - delta['in']
            agent_lrna_amt = agent_sell_amt * sell_prices[tkn]
            omnipool_lrna_amt = lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_sell_amt + omnipool_sell_amt)
            if not abs(buy_prices[tkn] - net_price)/net_price < tolerance:
                raise Exception("price not valid for " + tkn)
        elif delta["out"] < delta["in"]:  # LRNA being bought from Omnipool for TKN
            agent_buy_amt = delta['out']
            omnipool_buy_amt = delta['in'] - delta["out"]
            agent_lrna_amt = agent_buy_amt * buy_prices[tkn]
            omnipool_lrna_amt = -lrna_deltas[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_buy_amt + omnipool_buy_amt)
            if not abs(sell_prices[tkn] - net_price)/net_price < tolerance:
                raise Exception("price not valid for " + tkn)
        elif delta["out"] != 0:
            spot_price = omnipool.price(omnipool, tkn, "LRNA")
            assert abs(buy_prices[tkn] - spot_price) / spot_price < tolerance, "price not valid for " + tkn
            assert abs(sell_prices[tkn] - spot_price) / spot_price < tolerance, "price not valid for " + tkn

    return True


def update_intents(intents: list, transfers: list):
    for i in range(len(intents)):
        intent, transfer = intents[i], transfers[i]
        if 'sell_quantity' in intent:
            intent['sell_quantity'] -= transfer['sell_quantity']
            if intent['sell_quantity'] == 0:
                intent['buy_limit'] = 0
            else:
                intent['buy_limit'] -= transfer['buy_quantity']
        else:
            intent['buy_quantity'] -= transfer['buy_quantity']
            if intent['buy_quantity'] == 0:
                intent['sell_limit'] = 0
            else:
                intent['sell_limit'] -= transfer['sell_quantity']
