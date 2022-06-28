import copy
from typing import Callable


class Agent:
    unique_id: str = ''

    def __init__(self,
                 holdings: dict[str: float] = None,
                 shares: dict[str: float] = None,
                 share_prices: dict[str: float] = None,
                 trade_strategy: Callable = None,
                 ):
        """
        holdings should be in the form of:
        {
            asset_name: quantity
        }
        shares should be in the form of:
        {
            pool_name: share quantity
        }
        share_prices should be in the same form as shares, and the keys should match.
        The values of share_prices reflect the price at which those shares were acquired.
        """
        self.holdings = holdings or {}
        self.shares = shares or {}
        self.share_prices = share_prices or {}
        self.trade_strategy = trade_strategy

        self.asset_list = list(set(list(self.holdings.keys()) + list(self.shares.keys())))

    def copy(self):
        return copy.deepcopy(self)
