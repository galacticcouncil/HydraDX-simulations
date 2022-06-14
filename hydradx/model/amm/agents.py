from ..amm import omnipool_amm as oamm
from ..amm import basilisk_amm as bamm
from typing import Callable
import random


def agent_dict(
        token_list: list = None,
        r_values: dict = {},
        s_values: dict = {},
        p_values: dict = {},
        q: float = 0,
        trade_strategy: Callable = None
) -> dict:

    if token_list is None:
        token_list = list(set(list(p_values.keys()) + list(r_values.keys()) + list(s_values.keys())))
    return {
        'p': {token: p_values[token] if token in p_values else 0 for token in token_list},
        'q': q,
        'r': {token: r_values[token] if token in r_values else 0 for token in token_list},
        's': {token: s_values[token] if token in s_values else 0 for token in token_list},
        'trade_strategy': trade_strategy
    }


class TradeStrategy:
    def __init__(self, strategy_function: Callable):
        self.function = strategy_function

    def execute(self, agents, agent_id, market) -> tuple:
        return self.function(agents, agent_id, market)


class TradeStrategies:
    @staticmethod
    def random_swaps(amount: dict[str: float], randomize_amount: bool = False):
        """
        amount should be a dict in the form of:
        {
            token_name: sell_quantity
        }
        """

        @TradeStrategy
        def strategy(agents: dict, agent_id: str, market):
            buy_asset = random.choice(list(amount.keys()))
            sell_asset = random.choice(list(amount.keys()))
            sell_quantity = (
                             amount[sell_asset] * (random.random() if randomize_amount else 1)
                             ) or 1
            if buy_asset == sell_asset:
                return market, agents
            elif isinstance(market, bamm.BasiliskPoolState):
                return bamm.swap(
                    old_state=market,
                    old_agents=agents,
                    trader_id=agent_id,
                    tkn_sell=sell_asset,
                    tkn_buy=buy_asset,
                    sell_quantity=sell_quantity
                )
            elif isinstance(market, oamm.OmnipoolState):
                return oamm.swap_assets_direct(
                    old_state=market,
                    old_agents=agents,
                    trader_id=agent_id,
                    tkn_sell=sell_asset,
                    tkn_buy=buy_asset,
                    delta_token=sell_quantity,
                    fee_assets=market.asset_fee,
                    fee_lrna=market.lrna_fee
                )
        return strategy
