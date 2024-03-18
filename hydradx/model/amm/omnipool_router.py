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

    def price_route(self, tkn: str, denomination: str, tkn_pool_id: str, denom_pool_id: str) -> float:
        omnipool = self.exchanges[self.omnipool_id]

        if tkn_pool_id == denom_pool_id:
            if tkn_pool_id == self.omnipool_id:  # This is necessary because Omnipool has wrong price signature
                return omnipool.price(omnipool, tkn, denomination)
            else:
                return self.exchanges[tkn_pool_id].price(tkn, denomination)

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

    def swap_route(
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
            init_buy_pool_shares = 0
            if buy_pool_id != self.omnipool_id:
                if buy_pool_id == tkn_sell:
                    # we are just selling LP token for tkn_sell, i.e. withdrawing asset
                    self.exchanges[buy_pool_id].withdraw_asset(agent, buy_quantity, tkn_buy)
                    return self
                buy_quantity = self.exchanges[buy_pool_id].calculate_withdrawal_shares(tkn_buy, buy_quantity)
                omnipool_tkn_buy = buy_pool_id
                init_buy_pool_shares = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
            else:
                omnipool_tkn_buy = tkn_buy

            # calculate LP tokens of sell_pool_id required to buy sufficient buy_pool_id
            if sell_pool_id == self.omnipool_id:
                self.exchanges[self.omnipool_id].swap(agent, tkn_buy=omnipool_tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_quantity)
            elif sell_pool_id == tkn_buy:
                # we are just buying LP token for tkn_sell, i.e. buying shares
                self.exchanges[sell_pool_id].buy_shares(agent, buy_quantity, tkn_sell)
                return self
            else:
                # calculate quantity of tkn_sell required to buy sufficient sell_pool_id
                sell_quantity1 = self.exchanges[self.omnipool_id].calculate_sell_from_buy(buy_pool_id, sell_pool_id,
                                                                         buy_quantity)
                # buy shares of sell_pool_id, i.e., add liquidity
                shares_owned = agent.holdings[sell_pool_id] if sell_pool_id in agent.holdings else 0
                self.exchanges[sell_pool_id].buy_shares(agent, sell_quantity1, tkn_sell)
                shares_bought = agent.holdings[sell_pool_id] - shares_owned

                # sell shares of sell_pool_id
                self.exchanges[self.omnipool_id].swap(agent, tkn_buy=omnipool_tkn_buy, tkn_sell=sell_pool_id, sell_quantity=shares_bought)

            if buy_pool_id != self.omnipool_id:
                # withdraw liquidity from buy_pool_id if necessary
                buy_pool_shares_bought = agent.holdings[buy_pool_id] - init_buy_pool_shares
                self.exchanges[buy_pool_id].remove_liquidity(agent, buy_pool_shares_bought, tkn_buy)

        elif sell_quantity and not buy_quantity:
            # add liquidity to sell_pool
            if sell_pool_id != self.omnipool_id:
                if sell_pool_id == tkn_buy:
                    self.exchanges[sell_pool_id].add_liquidity(agent, sell_quantity, tkn_sell)
                    return self
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
            elif buy_pool_id == tkn_sell:
                self.exchanges[buy_pool_id].remove_liquidity(agent, sell_amt_1, tkn_buy)
                return self
            else:
                init_amt = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
                self.exchanges[self.omnipool_id].swap(agent, tkn_buy=buy_pool_id, tkn_sell=tkn_sell_1,
                                                      sell_quantity=sell_amt_1)
                share_amt = agent.holdings[buy_pool_id] - init_amt
                # remove liquidity from buy_pool
                self.exchanges[buy_pool_id].remove_liquidity(agent, share_amt, tkn_buy)
        else:
            raise ValueError('One of buy_quantity or sell_quantity must be zero')
        return self  # should never be reached

    def find_routes(self, tkn_buy, tkn_sell):
        '''Finds all possible routes to swap between tkn_buy and tkn_sell'''
        tkn_buy_pools = [pool_id for pool_id in self.exchanges if tkn_buy in self.exchanges[pool_id].asset_list]
        tkn_sell_pools = [pool_id for pool_id in self.exchanges if tkn_sell in self.exchanges[pool_id].asset_list]

        if len(tkn_buy_pools) == 0:
            raise ValueError(f'No pool with {tkn_buy} in asset list')
        if len(tkn_sell_pools) == 0:
            raise ValueError(f'No pool with {tkn_sell} in asset list')

        return [(tkn_sell_pool, tkn_buy_pool) for tkn_buy_pool in tkn_buy_pools for tkn_sell_pool in tkn_sell_pools]

    def find_best_route(self, tkn_buy, tkn_sell):
        '''Finds route to swap between tkn_buy and tkn_sell with lowest spot price'''
        routes = self.find_routes(tkn_buy, tkn_sell)
        return sorted(routes, key=lambda x: self.price_route(tkn_buy, tkn_sell, x[1], x[0]))[0]

    def swap(self, agent, tkn_buy, tkn_sell, buy_quantity=0, sell_quantity=0):
        '''Does swap along whatever route has best spot price'''
        if not buy_quantity and not sell_quantity:
            return self
        route = self.find_best_route(tkn_buy, tkn_sell)
        return self.swap_route(agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity, route[1], route[0])

    def simulate_swap_route(self,
        agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0,
        buy_pool_id: str = None,
        sell_pool_id: str = None):
        '''Does swap along specified route, returning new router and agent'''
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap_route(new_agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity, buy_pool_id, sell_pool_id)
        return new_state, new_agent

    def simulate_swap(self, agent, tkn_buy, tkn_sell, buy_quantity=0, sell_quantity=0):
        '''Does swap along whatever route has best spot price'''
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap(new_agent, tkn_buy, tkn_sell, buy_quantity, sell_quantity)
        return new_state, new_agent
