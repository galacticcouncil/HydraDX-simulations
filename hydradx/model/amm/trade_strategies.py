import math

from .global_state import GlobalState, swap, add_liquidity
from .basilisk_amm import ConstantProductPoolState
from typing import Callable
import random


class TradeStrategy:
    def __init__(self, strategy_function: Callable[[GlobalState, str], GlobalState], name: str):
        self.function = strategy_function
        self.name = name

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

    return TradeStrategy(strategy, name=f'random swaps ({list(amount.keys())})')


def invest_all(pool_id: str) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        agent = state.agents[agent_id]
        for asset in agent.holdings:

            if asset in state.pools[pool_id].asset_list:
                state = add_liquidity(
                    old_state=state,
                    pool_id=pool_id,
                    agent_id=agent_id,
                    quantity=agent.holdings[asset],
                    tkn_add=asset
                )

        # should only have to do this once
        state.agents[agent_id].trade_strategy = None
        return state

    return TradeStrategy(strategy, name=f'invest all ({pool_id})')


def constant_product_arbitrage(pool_id: str) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        agent = state.agents[agent_id]
        pool = state.pools[pool_id]
        old_pool = pool.copy()
        if not(isinstance(pool, ConstantProductPoolState)):
            raise TypeError(f'{pool_id} is not compatible with constant product arbitrage.')

        x = pool.asset_list[0]
        y = pool.asset_list[1]
        p = state.price(x) / state.price(y)
        y2 = math.sqrt(pool.liquidity[x] * pool.liquidity[y] * p)

        buy_quantity = pool.liquidity[y] - y2
        state = swap(state, pool_id, agent_id, x, y, buy_quantity=buy_quantity)
        return state

    return TradeStrategy(strategy, name=f'invest all ({pool_id})')
