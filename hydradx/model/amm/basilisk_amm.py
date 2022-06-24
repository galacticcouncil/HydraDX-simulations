import copy
import math
from typing import Callable
from mpmath import mpf, mp
mp.dps = 50


class ConstantProductPoolState:
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
        self._base_fee = mpf(trade_fee)
        self.fee_function = fee_function
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.fail = ''

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = mpf(quantity)

        self.shares = self.liquidity[self.asset_list[0]]

        self.unique_id = unique_id

    def slip_fee(self, sell_asset: str, buy_asset: str, trade_size: float) -> float:
        return trade_size ** 2 * self.liquidity[buy_asset] / (trade_size + self.liquidity[sell_asset]) ** 2

    def trade_fee(self, sell_asset: str, buy_asset: str, trade_size: float) -> float:
        fee = 0
        if self.fee_function:
            fee += self.fee_function(self, sell_asset, buy_asset, trade_size)
        fee += self._base_fee * trade_size
        return fee

    def copy(self):
        new_self = copy.deepcopy(self)
        new_self.fail = ''
        return new_self

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
        old_agents: dict,
        lp_id: str,
        quantity: float,
        tkn_add: str
) -> tuple[ConstantProductPoolState, dict]:
    new_agents = copy.deepcopy(old_agents)
    new_state = old_state.copy()

    lp = new_agents[lp_id]
    if new_state.unique_id not in lp['s']:
        lp['s'][new_state.unique_id] = 0

    for token in old_state.asset_list:
        delta_r = quantity * old_state.liquidity[token] / old_state.liquidity[tkn_add]
        lp['r'][token] -= delta_r
        new_state.liquidity[token] += delta_r

        if lp['r'][token] < 0:
            # fail
            return fail(old_state, old_agents)

    new_shares = (new_state.liquidity[tkn_add] / old_state.liquidity[tkn_add] - 1) * old_state.shares
    new_state.shares += new_shares

    lp['s'][new_state.unique_id] += new_shares
    return new_state, new_agents


def remove_liquidity(
        old_state: ConstantProductPoolState,
        old_agents: dict,
        lp_id: str,
        quantity: float,
        tkn_remove: str
) -> tuple[ConstantProductPoolState, dict]:
    quantity = abs(quantity) / old_state.shares * old_state.liquidity[tkn_remove]
    new_state, new_agents = add_liquidity(
        old_state, old_agents, lp_id, -quantity, tkn_remove
    )
    if min(new_state.liquidity.values()) < 0:
        return fail(old_state, old_agents)

    return new_state, new_agents


def swap(
        old_state: ConstantProductPoolState,
        old_agents: dict,
        trader_id: str,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0

):
    new_agents = copy.deepcopy(old_agents)
    new_state = old_state.copy()
    trader = new_agents[trader_id]

    if not (tkn_buy in new_state.asset_list and tkn_sell in new_state.asset_list):
        return fail(old_state, old_agents, 'invalid token name')

    if sell_quantity != 0:
        # when amount to be paid in is specified, calculate payout
        buy_quantity = sell_quantity * old_state.liquidity[tkn_buy] / (old_state.liquidity[tkn_sell] + sell_quantity)
        trade_fee = new_state.trade_fee(tkn_sell, tkn_buy, abs(sell_quantity))
        trader['r'][tkn_buy] -= trade_fee - buy_quantity
        trader['r'][tkn_sell] -= sell_quantity
        new_state.liquidity[tkn_sell] += sell_quantity
        new_state.liquidity[tkn_buy] += trade_fee - buy_quantity

    elif buy_quantity != 0:
        # calculate input price from a given payout
        sell_quantity = buy_quantity * old_state.liquidity[tkn_sell] / (old_state.liquidity[tkn_buy] - buy_quantity)
        trade_fee = new_state.trade_fee(tkn_sell, tkn_buy, abs(buy_quantity))
        trader['r'][tkn_sell] -= trade_fee + sell_quantity
        trader['r'][tkn_buy] += buy_quantity
        new_state.liquidity[tkn_buy] -= buy_quantity
        new_state.liquidity[tkn_sell] += trade_fee + sell_quantity
        #

    else:
        return fail(old_state, old_agents)

    if new_state.liquidity[tkn_buy] <= 0 or new_state.liquidity[tkn_sell] <= 0:
        return fail(old_state, old_agents)

    if trader['r'][tkn_sell] < 0 or trader['r'][tkn_buy] < 0:
        return fail(old_state, old_agents)

    return new_state, new_agents


def fail(old_state: ConstantProductPoolState, old_agents: dict, error: str = 'fail') -> tuple[ConstantProductPoolState, dict]:
    failed_state = old_state.copy()
    failed_state.fail = error
    return failed_state, copy.deepcopy(old_agents)
