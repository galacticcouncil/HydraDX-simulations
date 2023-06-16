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
        copy_self = copy.deepcopy(self)
        copy_self.exchange = exchange
        copy_self.tkn = tkn
        return copy_self

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

    def fail_transaction(self, error: str, agent: Agent):
        self.fail = error
        return self, agent

    def __setattr__(self, key, value):
        if hasattr(self, key):
            if isinstance(self.__getattribute__(key), FeeMechanism):
                if not isinstance(value, FeeMechanism):
                    super().__setattr__(key, basic_fee(value))
                    return
                else:
                    super().__setattr__(key, value.assign(self))
                    return
        super().__setattr__(key, value)


def basic_fee(f: float = 0) -> FeeMechanism:
    def fee_function(
            exchange: AMM, tkn: str, delta_tkn: float
    ) -> float:
        return f
    return FeeMechanism(fee_function, f"{f * 100}%")
