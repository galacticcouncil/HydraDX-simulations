from .global_state import GlobalState, swap, add_liquidity, remove_liquidity, AMM
from typing import Callable
import random


class TradeStrategy:
    def __init__(self, strategy_function: Callable[[GlobalState, str], GlobalState], name: str):
        self.function = strategy_function
        self.name = strategy_function.__name__

    def execute(self, state: GlobalState, agent_id: str) -> GlobalState:
        return self.function(state, agent_id)


def random_swaps(
    pool: str,
    amount: dict[str: float],
    randomize_amount: bool = False
) -> TradeStrategy:
    """
    amount should be a dict in the form of:
    {
        token_name: sell_quantity
    }
    """

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

    return TradeStrategy(strategy, name=f'random swaps ({list(amount.key())})')


def invest_all(pool: str) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        agent = state.agents[agent_id]
        for asset in agent.holdings:

            state = add_liquidity(
                old_state=state,
                pool_id=pool,
                agent_id=agent_id,
                quantity=agent.holdings[asset],
                tkn_add=asset
            )

        # should only have to do this once
        state.agents[agent_id].trade_strategy = None
        return state

    return TradeStrategy(strategy, name=f'invest all ({pool})')
