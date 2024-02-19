import copy

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState


class OmnipoolRouter:
    """
    Handles routing between Omnipool and Stableswap subpools that have LP assets in Omnipool
    """

    def __init__(self, exchanges: dict):
        self.exchanges = exchanges
        self.omnipool_id = None
        for exchange_id in exchanges:
            if isinstance(exchanges[exchange_id], OmnipoolState):
                if self.omnipool_id is not None:
                    raise ValueError('Multiple Omnipools in exchange list')
                else:
                    self.omnipool_id = exchange_id
        if self.omnipool_id is None:
            raise ValueError('No Omnipool in exchange list')

    def copy(self):
        copy_self = copy.deepcopy(self)
        return copy_self

    def price(self, tkn: str, denomination: str, tkn_pool_id: str, denom_pool_id: str) -> float:
        omnipool = self.exchanges[self.omnipool_id]

        tkn_subpool_share_price = 1
        denom_subpool_share_price = 1
        tkn_adj = tkn
        denom_adj = denomination
        if tkn_pool_id != self.omnipool_id:
            tkn_subpool_share_price = self.exchanges[tkn_pool_id].share_price(tkn)
            tkn_adj = tkn_pool_id
        if denom_pool_id != self.omnipool_id:
            denom_subpool_share_price = self.exchanges[denom_pool_id].share_price(denomination)
            denom_adj = denom_pool_id

        return denom_subpool_share_price * omnipool.price(omnipool, tkn_adj, denom_adj) / tkn_subpool_share_price

    def buy_limit(self, tkn_buy, tkn_sell=None):
        return sum([exchange.buy_limit(tkn_buy, tkn_sell) for exchange in self.exchanges])

    def sell_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    def swap(
        self,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0,
        buy_pool_id: str = None,
        sell_pool_id: str = None
    ):
        if not buy_quantity and not sell_quantity:
            return self

        if buy_pool_id is None:
            buy_pool_id = self.omnipool_id
        if sell_pool_id is None:
            sell_pool_id = self.omnipool_id

        if buy_pool_id == sell_pool_id:  # just call the swap function of the pool
            self.exchanges[buy_pool_id].swap(agent, tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=buy_quantity, sell_quantity=sell_quantity)
        elif buy_quantity and not sell_quantity:
            # need to calculate sell_quantity from buy_quantity
            # calculate LP tokens required to buy buy_quantity of tkn_buy
            if buy_pool_id != self.omnipool_id:
                buy_quantity = self.exchanges[buy_pool_id].calculate_withdrawal_shares(tkn_buy, buy_quantity)
            # calculate LP tokens of sell_pool_id required to buy sufficient buy_pool_id
            sell_quantity = self.exchanges[sell_pool_id].calculate_sell_from_buy(buy_pool_id, tkn_sell, buy_quantity)
            # calculate quantity of tkn_sell required to buy sufficient sell_pool_id
            if sell_pool_id != self.omnipool_id:
                sell_quantity = self.exchanges[sell_pool_id].calculate_buy_shares(sell_quantity, tkn_sell)
            self.swap(agent, tkn_sell, tkn_buy, 0, sell_quantity, buy_pool_id, sell_pool_id)
        elif sell_quantity and not buy_quantity:
            # add liquidity to sell_pool
            if sell_pool_id != self.omnipool_id:
                init_amt = agent.holdings[sell_pool_id] if sell_pool_id in agent.holdings else 0
                self.exchanges[sell_pool_id].add_liquidity(agent, sell_quantity, tkn_sell)
                sell_amt_1 = agent.holdings[sell_pool_id] - init_amt
                tkn_sell_1 = sell_pool_id
            else:
                sell_amt_1 = sell_quantity
                tkn_sell_1 = tkn_sell
            # swap LP shares
            if buy_pool_id == self.omnipool_id:
                self.exchanges[self.omnipool_id].swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell_1, sell_quantity=sell_amt_1)
            else:
                init_amt = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
                self.exchanges[self.omnipool_id].swap(agent, tkn_buy=buy_pool_id, tkn_sell=tkn_sell_1,
                                                      sell_quantity=sell_amt_1)
                share_amt = agent.holdings[buy_pool_id] - init_amt
                # remove liquidity from buy_pool
                self.exchanges[buy_pool_id].remove_liquidity(agent, share_amt, tkn_buy)
        else:
            raise ValueError('One of buy_quantity or sell_quantity must be zero')
