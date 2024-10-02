from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
import copy
from mpmath import mp, mpf


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
        intent_deltas: list  # list of deltas for each intent
):

    validate_intents(intents, intent_deltas)
    transfers, deltas = calculate_transfers(intents, intent_deltas)
    validate_transfer_amounts(transfers)
    pool_agent, lrna_deltas = execute_solution(omnipool, transfers, deltas)
    if not validate_remainder(pool_agent):
        raise Exception("agent has negative holdings")

    update_intents(intents, transfers)

    return True


def validate_intents(intents: list, intent_deltas: list):
    for i in range(len(intents)):
        intent = intents[i]
        sell_amt = -intent_deltas[i][0]
        assert 0 <= sell_amt <= intent['sell_quantity'], "amt processed is not in range"
        # assert intent_deltas[i][1] / sell_amt >= intent['buy_quantity'] / intent['sell_quantity'], "price is not valid"  #TODO fix this with some tolerance

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
        if amt < -1e-10:
            # return False
            raise
    return True


def update_intents(intents: list, transfers: list):
    for i in range(len(intents)):
        intent, transfer = intents[i], transfers[i]
        intent['sell_quantity'] -= transfer['sell_quantity']
        intent['buy_quantity'] -= transfer['buy_quantity']
