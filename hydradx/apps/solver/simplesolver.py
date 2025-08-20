from matplotlib import pyplot as plt
import sys, os
import streamlit as st


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.exchange import Exchange
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule

class IntentSolution:
    def __init__(
            self,
            intents: list = None,
            intent_exec_amounts: list = None,
            prices: dict = None,
            amm_transactions: list = None,
            router: Exchange = None,
            score: float = None
    ):
        self.intents = intents if intents is not None else []
        self.intent_exec_amounts = intent_exec_amounts if intent_exec_amounts is not None else {}
        self.prices = prices if prices is not None else {}
        self.amm_transactions = amm_transactions if amm_transactions is not None else []
        self.router = router
        self.score = score


def validate_solution(solution: IntentSolution):
    return True


def try_add_intent_to_solution(solution: IntentSolution, intent: dict) -> IntentSolution:
    if len(solution.intents) == 0:  # try to execute new intent against amm/router
        test_router = solution.router.copy()
        test_agent = Agent(enforce_holdings=False)
        if 'buy_quantity' in intent:
            test_router.swap(test_agent, tkn_buy=intent['tkn_buy'], tkn_sell=intent['tkn_sell'], buy_quantity=intent['buy_quantity'])
        elif 'sell_quantity' in intent:
            test_router.swap(test_agent, tkn_buy=intent['tkn_buy'], tkn_sell=intent['tkn_sell'], sell_quantity=intent['sell_quantity'])
        else:
            raise ValueError("Intent must have either 'buy_quantity' or 'sell_quantity' defined.")
    else:
        raise ValueError("Adding intents to existing solution is not yet implemented.")


op_liquidity = {
    'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
    'DOT': {'liquidity': 1000000, 'LRNA': 1000000},
    'USDT': {'liquidity': 1000000, 'LRNA': 1000000},
}
op = OmnipoolState(tokens=op_liquidity)
sol = IntentSolution(router = op)

