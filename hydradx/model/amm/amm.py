from .agents import Agent
import copy
from typing import Callable


class FeeMechanism:

    def __init__(self, fee_function: Callable, name: str):
        self.name = name
        self.fee_function = fee_function
        self.exchange = None

    def assign(self, exchange):
        self.exchange = exchange
        return self

    def compute(self, tkn: str, delta_tkn: float):
        return self.fee_function(
            exchange=self.exchange,
            tkn=tkn,
            delta_tkn=delta_tkn
        )


class AMM:
    unique_id: str
    asset_list: list[str]
    liquidity: dict[str: float]

    def __init__(self):
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        copy_self.fail = ''
        return copy_self

    def swap(
        self,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        return self.copy(), old_agent.copy()

    def add_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_add: str
    ):
        return self.copy(), old_agent.copy()

    def remove_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
    ):
        return self.copy(), old_agent.copy()

    def fail_transaction(self, error: str = 'fail'):
        self.fail = error
        return self

    @staticmethod
    def basic_fee(f: float) -> FeeMechanism:
        def fee_function(
                exchange: AMM, tkn: str, delta_tkn: float
        ) -> float:
            return f
        return FeeMechanism(fee_function, f"{f * 100}%")

    def __setattr__(self, key, value):
        if hasattr(self, key):
            if isinstance(self.__getattribute__(key), FeeMechanism):
                if not isinstance(value, FeeMechanism):
                    super().__setattr__(key, self.basic_fee(value))
                    return
                else:
                    super().__setattr__(key, value.assign(self))
                    return
        super().__setattr__(key, value)