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

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None) -> float:
        """
        How much tkn_sell will 1 tkn_buy cost?
        """
        return 0

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None) -> float:
        """
        How much tkn_buy can be bought for 1 tkn_sell?
        """
        return 0

    def buy_limit(self, tkn_buy, tkn_sell):
        return 0

    def sell_limit(self, tkn_buy, tkn_sell):
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

    def value_assets(self, assets: dict[str: float], **kwargs) -> float:
        return 0

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

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        pass

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        pass

def basic_fee(f: float = 0) -> FeeMechanism:
    def fee_function(
            exchange: AMM, tkn: str, delta_tkn: float
    ) -> float:
        return f
    f_mech = FeeMechanism(fee_function, f"{f * 100}%")
    f_mech.fee = f
    return f_mech
