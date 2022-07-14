import math

from .global_state import GlobalState, swap, add_liquidity, external_market_trade
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

        pool = state.pools[pool_id]
        if not(isinstance(pool, ConstantProductPoolState)):
            raise TypeError(f'{pool_id} is not compatible with constant product arbitrage.')

        x = pool.asset_list[0]
        y = pool.asset_list[1]
        p = state.price(x) / state.price(y)
        y2 = math.sqrt(pool.liquidity[x] * pool.liquidity[y] * p)
        buy_quantity = pool.liquidity[y] - y2

        agent = state.agents[agent_id]
        if x not in agent.holdings:
            agent.holdings[x] = 0
        if y not in agent.holdings:
            agent.holdings[y] = 0

        # buy just enough of non-USD asset
        if buy_quantity > 0 and x != 'USD' or buy_quantity < 0 and y != 'USD':
            state = external_market_trade(
                state=state,
                agent_id=agent_id,
                tkn_buy=x if buy_quantity > 0 else 'USD',
                tkn_sell=y if buy_quantity < 0 else 'USD',
                sell_quantity=buy_quantity
            )

        # then swap
        new_state = swap(state, pool_id, agent_id, tkn_sell=x, tkn_buy=y, buy_quantity=buy_quantity)

        # immediately cash out everything for USD
        new_agent = new_state.agents[agent_id]
        for tkn, quantity in new_agent.holdings.items():
            if new_agent.holdings[tkn] > 0 and tkn != 'USD':
                new_state = external_market_trade(state, agent_id, tkn_buy='USD', tkn_sell=tkn, sell_quantity=quantity)

        return new_state

    return TradeStrategy(strategy, name=f'invest all ({pool_id})')
