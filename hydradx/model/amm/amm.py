from .agents import Agent
import copy
from typing import Callable


class FeeMechanism:

    def __init__(self, fee_function: Callable, name: str):
        self.name = name
        self.fee_function = fee_function
        self.exchange = None
        self.tkn = None

    def assign(self, exchange, tkn=''):
        self.exchange = exchange
        self.tkn = tkn
        return self

    def compute(self, tkn: str = '', delta_tkn: float = 0) -> float:
        return self.fee_function(
            exchange=self.exchange,
            tkn=tkn or self.tkn,
            delta_tkn=delta_tkn
        )


class AMM:
    unique_id: str
    asset_list: list[str]
    liquidity: dict[str: float]
    update_function: Callable = None

    def __init__(self):
        self.oracles = None
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        copy_self.fail = ''
        return copy_self

    def update(self):
        pass

    @staticmethod
    def price(state, tkn: str, denomination: str = '') -> float:
        return 0

    def swap(
        self,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        return self

    def add_liquidity(
        self,
        agent: Agent,
        quantity: float,
        tkn_add: str
    ):
        return self

    def remove_liquidity(
        self,
        agent: Agent,
        quantity: float,
        tkn_remove: str
    ):
        return self

    def fail_transaction(self, error: str, agent: Agent):
        self.fail = error
        return self

    def set_fee(self, fee_name: str, fee_amount: dict or FeeMechanism or float):
        return setattr(self, fee_name, self._get_fee(fee_amount))

    def _get_fee(self, value: dict or FeeMechanism or float) -> dict:

        if isinstance(value, dict):
            if set(value.keys()) != set(self.asset_list):
                # I do not believe we were handling this case correctly
                # we can extend this when it is a priority
                raise ValueError(f'fee dict keys must match asset list: {self.asset_list}')
            return ({
                tkn: (
                    value[tkn].assign(self, tkn)
                    if isinstance(fee, FeeMechanism)
                    else basic_fee(fee).assign(self, tkn)
                )
                for tkn, fee in value.items()
            })
        elif isinstance(value, FeeMechanism):
            return {tkn: value.assign(self, tkn) for tkn in self.asset_list}
        else:
            return {tkn: basic_fee(value or 0).assign(self, tkn) for tkn in self.asset_list}


def basic_fee(f: float = 0) -> FeeMechanism:
    def fee_function(
            exchange: AMM, tkn: str, delta_tkn: float
    ) -> float:
        return f
    return FeeMechanism(fee_function, f"{f * 100}%")
