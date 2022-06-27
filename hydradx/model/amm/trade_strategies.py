from .global_state import GlobalState, swap, add_liquidity, remove_liquidity, AMM
from .agents import Agent
from typing import Callable
import random


class TradeStrategy:
    def __init__(self, strategy_function: Callable):
        self.function = strategy_function

    def execute(self, state: GlobalState, agent: Agent) -> GlobalState:
        return self.function(state, agent_id=agent.unique_id)


class TradeStrategies:
    @staticmethod
    def random_swaps(pool: str, amount: dict[str: float], randomize_amount: bool = False):
        """
        amount should be a dict in the form of:
        {
            token_name: sell_quantity
        }
        """

        @TradeStrategy
        def strategy(state: GlobalState, agent_id: str):
            buy_asset = random.choice(list(amount.keys()))
            sell_asset = random.choice(list(amount.keys()))
            sell_quantity = (
                                    amount[sell_asset] * (random.random() if randomize_amount else 1)
                            ) or 1
            if buy_asset == sell_asset:
                return state
            else:
                return swap(
                    old_state=state,
                    pool_id=pool,
                    agent_id=agent_id,
                    tkn_sell=sell_asset,
                    tkn_buy=buy_asset,
                    sell_quantity=sell_quantity
                )

        return strategy
