import copy
import math
from typing import Callable
from .global_state import AMM
from .agents import Agent
from mpmath import mpf, mp
mp.dps = 50


class ConstantProductPoolState(AMM):
    unique_ids = {''}

    def __init__(self, tokens: dict[str: float], trade_fee: float = 0, fee_function: Callable = None, unique_id=''):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There should only be two.
        """
        super().__init__()
        self._base_fee = mpf(trade_fee)
        self.fee_function = fee_function
        self.liquidity = dict()
        self.asset_list: list[str] = []

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = mpf(quantity)

        self.shares = self.liquidity[self.asset_list[0]]

        self.unique_id = unique_id

    def thorchain_fee(self, sell_asset: str, buy_asset: str, trade_size: float) -> float:
        return trade_size * self.liquidity[buy_asset] / (trade_size + self.liquidity[sell_asset]) ** 2

    @staticmethod
    def custom_slip_fee(slip_factor: float) -> Callable:
        def fee_function(exchange, sell_asset: str, buy_asset: str, trade_size: float) -> float:
            return trade_size * slip_factor / exchange.liquidity[sell_asset]
        return fee_function

    def trade_fee(self, sell_asset: str, buy_asset: str, trade_size: float) -> float:
        fee = 0
        if self.fee_function:
            fee += self.fee_function(self, sell_asset, buy_asset, trade_size)
        fee += self._base_fee
        return fee

    @property
    def invariant(self):
        return math.prod(self.liquidity.values())

    def __repr__(self):
        return (
            f'Constant Product Pool\n'
            f'base trade fee: {self._base_fee}\n'
            f'tokens: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {token}\n'
                f'    quantity: {self.liquidity[token]}\n'
                f'    weight: {self.liquidity[token] / sum(self.liquidity.values())}\n'
            ) for token in self.asset_list]
        ) + '\n)'


def add_liquidity(
        old_state: ConstantProductPoolState,
        old_agent: Agent,
        quantity: float,
        tkn_add: str
) -> tuple[ConstantProductPoolState, Agent]:
    new_agent = old_agent.copy()
    new_state = old_state.copy()

    if new_state.unique_id not in new_agent.shares:
        new_agent.shares[new_state.unique_id] = 0

    for token in old_state.asset_list:
        delta_r = quantity * old_state.liquidity[token] / old_state.liquidity[tkn_add]
        new_agent.holdings[token] -= delta_r
        new_state.liquidity[token] += delta_r

        if new_agent.holdings[token] < 0:
            # fail
            return old_state.fail_transaction('Agent has insufficient funds.'), old_agent

    new_shares = (new_state.liquidity[tkn_add] / old_state.liquidity[tkn_add] - 1) * old_state.shares
    new_state.shares += new_shares

    new_agent.shares[new_state.unique_id] += new_shares
    return new_state, new_agent


def remove_liquidity(
        old_state: ConstantProductPoolState,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
) -> tuple[ConstantProductPoolState, Agent]:
    quantity = abs(quantity) / old_state.shares * old_state.liquidity[tkn_remove]
    new_state, new_agent = add_liquidity(
        old_state, old_agent, -quantity, tkn_remove
    )
    if min(new_state.liquidity.values()) < 0:
        return old_state.fail_transaction('Tried to remove more liquidity than exists in the pool.'), old_agent

    return new_state, new_agent


def swap(
        old_state: ConstantProductPoolState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0

) -> tuple[ConstantProductPoolState, Agent]:
    new_agent = old_agent.copy()
    new_state = old_state.copy()

    if not (tkn_buy in new_state.asset_list and tkn_sell in new_state.asset_list):
        return old_state.fail_transaction('Invalid token name.'), old_agent

    if sell_quantity != 0:
        # when amount to be paid in is specified, calculate payout
        buy_quantity = sell_quantity * old_state.liquidity[tkn_buy] / (old_state.liquidity[tkn_sell] + sell_quantity)
        trade_fee = new_state.trade_fee(tkn_sell, tkn_buy, abs(sell_quantity))
        buy_quantity *= 1 - trade_fee
        new_agent.holdings[tkn_buy] += buy_quantity
        new_agent.holdings[tkn_sell] -= sell_quantity
        new_state.liquidity[tkn_sell] += sell_quantity
        new_state.liquidity[tkn_buy] -= buy_quantity

    elif buy_quantity != 0:
        # calculate input price from a given payout
        sell_quantity = buy_quantity * old_state.liquidity[tkn_sell] / (old_state.liquidity[tkn_buy] - buy_quantity)
        trade_fee = new_state.trade_fee(tkn_sell, tkn_buy, abs(sell_quantity))
        sell_quantity *= 1 + trade_fee
        new_agent.holdings[tkn_sell] -= sell_quantity
        new_agent.holdings[tkn_buy] += buy_quantity
        new_state.liquidity[tkn_buy] -= buy_quantity
        new_state.liquidity[tkn_sell] += sell_quantity

    else:
        return old_state.fail_transaction('Must specify buy quantity or sell quantity.'), old_agent

    if new_state.liquidity[tkn_buy] <= 0 or new_state.liquidity[tkn_sell] <= 0:
        return old_state.fail_transaction('Not enough liquidity in the pool.'), old_agent

    if new_agent.holdings[tkn_sell] < 0 or new_agent.holdings[tkn_buy] < 0:
        return old_state.fail_transaction('Agent has insufficient holdings.'), old_agent

    return new_state, new_agent


ConstantProductPoolState.swap = staticmethod(swap)
ConstantProductPoolState.add_liquidity = staticmethod(add_liquidity)
ConstantProductPoolState.remove_liquidity = staticmethod(remove_liquidity)
