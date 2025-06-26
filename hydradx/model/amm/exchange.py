from .agents import Agent
import copy
from typing import Callable

class Exchange:
    unique_id: str
    asset_list: list[str]
    liquidity: dict[str: float]
    update_function: Callable = None
    time_step: int = 0

    def __init__(self):
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        copy_self.fail = ''
        return copy_self

    def update(self):
        pass

    def price(self, tkn: str, numeraire: str = '') -> float:
        return 0

    def buy_spot(self, tkn_buy: str, tkn_sell: str) -> float:
        """
        How much tkn_sell will 1 tkn_buy cost?
        """
        return 0

    def sell_spot(self, tkn_sell: str, tkn_buy: str) -> float:
        """
        How much tkn_buy can be bought for 1 tkn_sell?
        """
        return 0

    def buy_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    def sell_limit(self, tkn_buy, tkn_sell):
        return float('inf')

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

    def fail_transaction(self, error: str):
        self.fail = error
        return self

    def value_assets(self, assets: dict[str: float]) -> float:
        """
        Calculate the value of the assets in the exchange.
        """
        total_value = 0
        for tkn, quantity in assets.items():
            price = self.price(tkn)
            total_value += price * quantity
        return total_value

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        pass

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        pass

