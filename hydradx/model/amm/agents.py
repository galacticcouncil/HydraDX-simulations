import copy


class Agent:
    unique_id: str = ''

    def __init__(self,
                 holdings: dict[str: float] = None,
                 share_prices: dict[str: float] = None,
                 delta_r: dict[str: float] = None,
                 trade_strategy: any = None,
                 unique_id: str = 'agent',
                 nfts: dict[str: any] = None,
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
        self.initial_holdings = copy.copy(self.holdings)
        self.share_prices = share_prices or {}
        self.delta_r = delta_r or {}
        self.trade_strategy = trade_strategy
        self.asset_list = list(self.holdings.keys())
        self.unique_id = unique_id
        self.nfts = nfts or {}

    def __repr__(self):
        precision = 10
        holdings = {tkn: round(self.holdings[tkn], precision) for tkn in self.holdings}
        share_prices = {tkn: round(self.share_prices[tkn], precision) for tkn in self.share_prices}
        return (
                f'Agent: {self.unique_id}\n'
                f'********************************\n'
                f'trade strategy: {self.trade_strategy.name if self.trade_strategy else "None"}\n' +
                f'holdings: (\n\n' +
                f'\n'.join(
                    [(
                            f'    *{tkn}*: {holdings[tkn]}\n' +
                            (f'    price: {share_prices[tkn]}\n' if tkn in share_prices else '')
                    ) for tkn in self.holdings]
                ) + ')\n')

    def copy(self):
        copy_self = Agent(
            holdings={k: v for k, v in self.holdings.items()},
            share_prices={k: v for k, v in self.share_prices.items()},
            delta_r={k: v for k, v in self.delta_r.items()},
            trade_strategy=self.trade_strategy,
            unique_id=self.unique_id,
            nfts={id: copy.deepcopy(nft) for id, nft in self.nfts.items()}
        )
        copy_self.initial_holdings = {k: v for k, v in self.initial_holdings.items()}
        copy_self.asset_list = [tkn for tkn in self.asset_list]
        return copy_self
        # return copy.deepcopy(self)

    def is_holding(self, tkn, amt=None):
        if amt is None:
            return tkn in self.holdings and self.holdings[tkn] > 0
        elif amt == 0:
            return True
        else:
            return tkn in self.holdings and self.holdings[tkn] >= amt


class AgentArchiveState:
    def __init__(self, agent: Agent):
        self.unique_id = agent.unique_id
        self.holdings = {k: v for k, v in agent.holdings.items()}
        self.share_prices = {k: v for k, v in agent.share_prices.items()}
        self.asset_list = [tkn for tkn in agent.asset_list]
