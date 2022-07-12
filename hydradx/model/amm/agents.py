import copy


class Agent:
    unique_id: str = ''

    def __init__(self,
                 holdings: dict[str: float] = None,
                 shares: dict[any: float] = None,
                 share_prices: dict[str: float] = None,
                 trade_strategy: any = None,
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

        self.asset_list = list(set(list(self.holdings.keys()) + [share[1] for share in self.shares.keys()]))

    def __repr__(self):
        return (
            f'Agent:\n' +
            f'name: {self.unique_id}\n'
            f'trade strategy: {self.trade_strategy.name if self.trade_strategy else "None"}\n' +
            f'holdings: (\n' +
            f')\n(\n'.join(
                [(
                    f'    {token}: {self.holdings[token]}\n'
                ) for token in self.holdings]
            ) + ')\n' +
            f'shares: (\n' +
            f')\n(\n'.join(
                [(
                    f'    {pool}: {self.shares[pool]}\n'
                    f'    price: {self.share_prices[pool]}\n'
                ) for pool in self.shares]
            ) + ')\n')

    def copy(self):
        return copy.deepcopy(self)
