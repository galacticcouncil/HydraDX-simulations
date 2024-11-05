from hydradx.model.amm.omnipool_amm import OmnipoolState, simulate_swap
from hydradx.model.amm.agents import Agent
import copy
from mpmath import mp, mpf
import math


def validate_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        intent_deltas: list,  # list of deltas for each intent
):
    return validate_and_execute_solution(
        omnipool=omnipool.copy(),
        intents=copy.deepcopy(intents),
        intent_deltas=intent_deltas
    )


def calculate_transfers(
        intents: list,  # swap desired to be processed
        intent_deltas: list  # list of deltas for each intent
) -> (list, dict):
    transfers = []
    deltas = {}
    for i in range(len(intents)):
        intent = intents[i]
        sell_amt = -intent_deltas[i][0]
        buy_amt = intent_deltas[i][1]
        transfer = {
            'agent': intent['agent'],
            'buy_quantity': buy_amt,
            'sell_quantity': sell_amt,
            'tkn_buy': intent['tkn_buy'],
            'tkn_sell': intent['tkn_sell']
        }
        transfers.append(transfer)
        if intent['tkn_buy'] not in deltas:
            deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        deltas[intent['tkn_buy']]["out"] += buy_amt
        if intent['tkn_sell'] not in deltas:
            deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        deltas[intent['tkn_sell']]["in"] += sell_amt

    return transfers, deltas


def validate_and_execute_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        intent_deltas: list,  # list of deltas for each intent
        tkn_profit: str = None
):

    validate_intents(intents, intent_deltas)
    transfers, deltas = calculate_transfers(intents, intent_deltas)
    validate_transfer_amounts(transfers)
    pool_agent, fee_agent, lrna_deltas = execute_solution(omnipool, transfers, deltas, exit_to=tkn_profit)
    if not validate_remainder(pool_agent):
        raise Exception("agent has negative holdings")

    update_intents(intents, transfers)
    if tkn_profit is not None:
        tkn_list = [tkn for tkn in pool_agent.holdings if tkn != tkn_profit]
        for tkn in tkn_list:
            omnipool.swap(pool_agent, tkn_profit, tkn, sell_quantity=pool_agent.holdings[tkn])
        return True, (pool_agent.holdings[tkn_profit] if tkn_profit in pool_agent.holdings else 0)
    else:
        return True


def validate_intents(intents: list, intent_deltas: list):
    for i in range(len(intents)):
        intent = intents[i]
        sell_amt = -intent_deltas[i][0]
        buy_amt = intent_deltas[i][1]
        if not 0 <= sell_amt <= intent['sell_quantity']:
            raise Exception("amt processed is not in range")
        if intent['partial'] == False and sell_amt not in [0, intent['sell_quantity']]:
            raise Exception("intent should not be partially executed")
        if intent['partial'] == False and 0 < buy_amt < intent['buy_quantity']:
            raise Exception("intent should not be partially executed")
        tolerance = 1e-6  # temporarily allowing some tolerance, only for testing purposes
        if sell_amt > 0 and buy_amt / sell_amt < (1 - tolerance) * intent['buy_quantity'] / intent['sell_quantity']:
            raise Exception("price is not within tolerance")

    return True


def validate_transfer_amounts(transfers: list):
    for transfer in transfers:
        if not transfer['agent'].is_holding(transfer['tkn_sell'], transfer['sell_quantity']):
            raise Exception(f"agent does not have enough holdings in token {transfer['tkn_sell']}")

    return True


def execute_solution(
        omnipool: OmnipoolState,
        transfers: list,
        deltas: dict,  # note that net_deltas can be easily reconstructed from transfers
        fee_match: float = 0.0,
        exit_to: str = None
):
    pool_agent = Agent()
    fee_agent = Agent()

    # transfer assets in from agents whose intents are being executed
    for transfer in transfers:
        if transfer['sell_quantity'] > 0:
            if transfer['tkn_sell'] not in pool_agent.holdings:
                pool_agent.holdings[transfer['tkn_sell']] = 0
            pool_agent.holdings[transfer['tkn_sell']] += transfer['sell_quantity']
            transfer['agent'].holdings[transfer['tkn_sell']] -= transfer['sell_quantity']
        elif transfer['sell_quantity'] < 0:
            raise Exception("sell quantity is negative")

    init_lrna = {tkn: omnipool.lrna[tkn] for tkn in deltas if tkn != 'LRNA'}
    for tkn_buy in deltas:
        if deltas[tkn_buy]["in"] < deltas[tkn_buy]["out"]:
            for tkn_sell in deltas:
                # try to sell tkn_buy for tkn_sell, if it is the direction we need to go.
                if tkn_buy != tkn_sell and deltas[tkn_sell]["in"] > deltas[tkn_sell]["out"]:
                    max_buy_amt = deltas[tkn_buy]["out"] - deltas[tkn_buy]["in"]
                    max_buy_amt = math.nextafter(max_buy_amt, math.inf) if max_buy_amt != 0 else 0
                    max_sell_amt = deltas[tkn_sell]["in"] - deltas[tkn_sell]["out"]
                    max_sell_amt = math.nextafter(max_sell_amt, -math.inf) if max_sell_amt != 0 else 0
                    test_state, test_agent = simulate_swap(omnipool, pool_agent, tkn_buy, tkn_sell, sell_quantity=max_sell_amt)
                    buy_given_max_sell = test_agent.holdings[tkn_buy] - (pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0)
                    if buy_given_max_sell > max_buy_amt:  # can't do max sell, do max buy instead
                        init_sell_holdings = pool_agent.holdings[tkn_sell]
                        omnipool.swap(pool_agent, tkn_buy, tkn_sell, buy_quantity=max_buy_amt)
                        deltas[tkn_buy]["out"] -= max_buy_amt
                        deltas[tkn_sell]["in"] -= init_sell_holdings - pool_agent.holdings[tkn_sell]
                    else:
                        init_buy_liquidity = pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0
                        omnipool.swap(pool_agent, tkn_buy, tkn_sell, sell_quantity=max_sell_amt)
                        deltas[tkn_sell]["in"] -= max_sell_amt
                        deltas[tkn_buy]["out"] -= pool_agent.holdings[tkn_buy] - init_buy_liquidity
        lrna_deltas = {tkn: omnipool.lrna[tkn] - init_lrna[tkn] for tkn in init_lrna}
        # transfer matched fees to fee agent
        matched_amt = min(deltas[tkn_buy]["in"], deltas[tkn_buy]["out"])
        fee_amt = matched_amt * fee_match
        if fee_amt > 0:
            pool_agent.holdings[tkn_buy] -= fee_amt
            fee_agent.holdings[tkn_buy] = fee_amt

    # transfer assets out to intent agents
    for transfer in transfers:
        if transfer['buy_quantity'] > 0:
            pool_agent.holdings[transfer['tkn_buy']] -= transfer['buy_quantity']
            if transfer['tkn_buy'] not in transfer['agent'].holdings:
                transfer['agent'].holdings[transfer['tkn_buy']] = 0
            transfer['agent'].holdings[transfer['tkn_buy']] += transfer['buy_quantity']
        elif transfer['buy_quantity'] < 0:
            raise Exception("buy quantity is negative")

    if exit_to is not None:  # have pool agent exit to a specific token
        tkns = [tkn for tkn in pool_agent.holdings if tkn != exit_to]
        for tkn in tkns:
            if pool_agent.holdings[tkn] > 0:
                omnipool.swap(pool_agent, exit_to, tkn, sell_quantity=pool_agent.holdings[tkn])
        for tkn in tkns:
            if pool_agent.holdings[tkn] < 0:
                omnipool.swap(pool_agent, tkn, exit_to, buy_quantity=-pool_agent.holdings[tkn])

    return pool_agent, fee_agent, lrna_deltas


def validate_remainder(pool_agent: Agent):
    for tkn, amt in pool_agent.holdings.items():
        if amt < -1e-10:
            # return False
            raise
    return True


def update_intents(intents: list, transfers: list):
    for i in range(len(intents)):
        intent, transfer = intents[i], transfers[i]
        intent['sell_quantity'] -= transfer['sell_quantity']
        intent['buy_quantity'] -= transfer['buy_quantity']
