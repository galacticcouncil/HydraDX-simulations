import copy


class Agent:
    unique_id: str = ''

    def __init__(self,
                 holdings: dict[str: float] = None,
                 share_prices: dict[str: float] = None,
                 trade_strategy: any = None,
                 ):
        """
        holdings should be in the form of:
        {
            asset_name: quantity
        }
        share_prices should be in the form of:
        {
            asset_name: price
        }
        The values of share_prices reflect the price at which those shares were acquired.
        """
        self.holdings = holdings or {}
        self.share_prices = share_prices or {}
        self.trade_strategy = trade_strategy
        self.asset_list = list(self.holdings.keys())

    def __repr__(self):
        return (
            f'Agent:\n' +
            f'name: {self.unique_id}\n'
            f'trade strategy: {self.trade_strategy.name if self.trade_strategy else "None"}\n' +
            f'holdings: (\n' +
            f')\n(\n'.join(
                [(
                    f'    {tkn}: {self.holdings[tkn]}\n' +
                    f'    price: {self.share_prices[tkn]}\n' if tkn in self.share_prices else ''
                ) for tkn in self.holdings]
            ) + ')\n')

    def copy(self):
        copy_self = copy.deepcopy(self)
        # copy_self.trade_strategy = copy.deepcopy(self.trade_strategy)
        # copy_self.trade_strategy.function = copy.deepcopy(self.trade_strategy.function)
        return copy_self
