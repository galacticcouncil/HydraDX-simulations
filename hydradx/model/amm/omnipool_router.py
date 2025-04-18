import copy

from .agents import Agent
from .exchange import Exchange
from .omnipool_amm import OmnipoolState
from .stableswap_amm import StableSwapPoolState

class OmnipoolRouter(Exchange):
    """
    Handles routing between Omnipool and Stableswap subpools that have LP assets in Omnipool
    """

    def __init__(self, exchanges: dict or list):
        super().__init__()
        self.exchanges = exchanges if type(exchanges) == dict else {ex.unique_id: ex for ex in exchanges}
        self.omnipool_id = None
        for exchange_id in self.exchanges:
            if isinstance(self.exchanges[exchange_id], OmnipoolState):
                if self.omnipool_id is not None:
                    raise ValueError('Multiple Omnipools in exchange list')
                else:
                    self.omnipool_id = exchange_id
        if self.omnipool_id is None:
            raise ValueError('No Omnipool in exchange list')
        self.omnipool: OmnipoolState = self.exchanges[self.omnipool_id]
        self.asset_list = list(set([tkn for exchange in self.exchanges.values() for tkn in exchange.asset_list]))
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        return copy_self

    def price_route(self, tkn: str, denomination: str, tkn_pool_id: str, denom_pool_id: str) -> float:
        if tkn_pool_id == denom_pool_id:
            if tkn_pool_id == self.omnipool_id:  # This is necessary because Omnipool has wrong price signature
                return self.omnipool.price(tkn, denomination)
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

        return denom_subpool_share_price * self.omnipool.price(tkn_adj, denom_adj) / tkn_subpool_share_price

    def buy_limit(self, tkn_buy, tkn_sell=None):
        return sum([exchange.buy_limit(tkn_buy, tkn_sell) for exchange in self.exchanges.values()])

    def sell_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    @property
    def liquidity(self):
        return {
            tkn: sum([exchange.liquidity[tkn] if tkn in exchange.liquidity else 0 for exchange in self.exchanges.values()])
            for tkn in [tkn for exchange in self.exchanges.values() for tkn in exchange.asset_list]
        }

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        sell_pool, buy_pool = self.find_best_route(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
        if fee == 0:
            # this handles custom fees only in the case that fee is explicitly set to 0
            # otherwise, each pool's native fees will be used. Values other than 0 are ignored.
            return self.price_route(tkn=tkn_buy, denomination=tkn_sell, tkn_pool_id=buy_pool, denom_pool_id=sell_pool)
        if buy_pool == sell_pool:
            return self.exchanges[buy_pool].buy_spot(tkn_buy, tkn_sell)
        elif sell_pool == self.omnipool_id != buy_pool:
            price = self.omnipool.buy_spot(tkn_buy=buy_pool, tkn_sell=tkn_sell)
            price /= self.exchanges[buy_pool].withdraw_asset_spot(tkn_remove=tkn_buy)
            return price
        elif buy_pool == self.omnipool_id != sell_pool:
            price = self.exchanges[sell_pool].buy_shares_spot(tkn_add=tkn_sell)
            price *= self.omnipool.buy_spot(tkn_buy=tkn_buy, tkn_sell=sell_pool)
            return price
        elif buy_pool != self.omnipool_id != sell_pool:
            price1 = self.exchanges[sell_pool].buy_shares_spot(tkn_add=tkn_sell)
            price2 = self.omnipool.buy_spot(tkn_buy=buy_pool, tkn_sell=sell_pool)
            price3 = 1 / self.exchanges[buy_pool].withdraw_asset_spot(tkn_remove=tkn_buy)
            return price1 * price2 * price3

        return  self.price_route(tkn_buy, tkn_sell, buy_pool, sell_pool)

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        sell_pool, buy_pool = self.find_best_route(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
        if fee == 0:
            # this handles custom fees only in the case that fee is explicitly set to 0
            # otherwise, each pool's native fees will be used. Values other than 0 are ignored.
            return self.price_route(tkn=tkn_sell, denomination=tkn_buy, tkn_pool_id=sell_pool, denom_pool_id=buy_pool)
        if sell_pool == buy_pool:
            return self.exchanges[sell_pool].sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
        elif sell_pool == self.omnipool_id != buy_pool:
            # we will sell enough shares of sell_pool to buy sell_quantity tkn_sell
            price = self.exchanges[sell_pool].sell_spot(tkn_sell=tkn_sell, tkn_buy=buy_pool)
            price *= self.exchanges[buy_pool].remove_liquidity_spot(tkn_remove=tkn_buy)
            return price
        elif buy_pool == self.omnipool_id != sell_pool:
            price = 1 / self.exchanges[sell_pool].add_liquidity_spot(tkn_sell)
            price *= self.exchanges[buy_pool].sell_spot(tkn_sell=sell_pool, tkn_buy=tkn_buy)
            return price
        elif buy_pool != self.omnipool_id != sell_pool:
            price1 = 1 / self.exchanges[sell_pool].add_liquidity_spot(tkn_sell)
            price2 = self.omnipool.sell_spot(tkn_sell=sell_pool, tkn_buy=buy_pool)
            price3 = self.exchanges[buy_pool].remove_liquidity_spot(tkn_buy)
            return price1 * price2 * price3

        return self.price_route(tkn_sell, tkn_buy, sell_pool, buy_pool)

    def calculate_buy_from_sell(self, tkn_sell: str, tkn_buy: str, sell_quantity: float):
        sell_pool, buy_pool = self.find_best_route(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
        if sell_pool == buy_pool:
            return self.exchanges[sell_pool].calculate_buy_from_sell(
                tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity
            )
        elif sell_pool == self.omnipool_id != buy_pool:
            shares_bought = self.exchanges[sell_pool].calculate_buy_from_sell(
                tkn_buy=buy_pool, tkn_sell=tkn_sell, sell_quantity=sell_quantity
            )
            return self.exchanges[buy_pool].calculate_remove_liquidity(shares_bought, tkn_buy)
        elif buy_pool == self.omnipool_id != sell_pool:
            shares_bought = self.exchanges[sell_pool].calculate_add_liquidity(
                tkn_add=tkn_sell, quantity=sell_quantity
            )
            return self.exchanges[buy_pool].calculate_buy_from_sell(
                tkn_sell=sell_pool, tkn_buy=tkn_buy, sell_quantity=shares_bought
            )
        elif buy_pool != self.omnipool_id != sell_pool:
            shares_bought = self.calculate_buy_from_sell(
                tkn_buy=buy_pool, tkn_sell=tkn_sell, sell_quantity=sell_quantity
            )
            return self.exchanges[buy_pool].calculate_remove_liquidity(shares_bought, tkn_buy)
        else:
            test_router = self.copy()
            test_agent = Agent(holdings={tkn_sell: sell_quantity})
            test_router.swap(
                agent=test_agent,
                tkn_buy=tkn_buy,
                tkn_sell=tkn_sell,
                sell_quantity=sell_quantity
            )
            return test_agent.holdings[tkn_buy]

    def fail_transaction(self, fail_message: str):
        self.fail = fail_message
        return self

    def swap_route(
        self,
        agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0,
        buy_pool_id: str = None,
        sell_pool_id: str = None
    ):
        if not buy_quantity and not sell_quantity:
            return self
        omnipool = self.omnipool

        if buy_pool_id is None:
            buy_pool_id = self.omnipool_id
        if sell_pool_id is None:
            sell_pool_id = self.omnipool_id

        if buy_pool_id == sell_pool_id:  # just call the swap function of the pool
            self.exchanges[buy_pool_id].swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_quantity, sell_quantity=sell_quantity)
            return self
        elif buy_quantity and buy_pool_id == tkn_sell:
            # we are selling LP token for tkn_sell, i.e. withdrawing asset
            self.exchanges[buy_pool_id].withdraw_asset(agent, buy_quantity, tkn_buy)
            return self
        elif buy_quantity and sell_pool_id == tkn_buy:
            # we are just buying LP token for tkn_sell, i.e. buying shares
            self.exchanges[sell_pool_id].buy_shares(agent, buy_quantity, tkn_sell)
            return self
        elif sell_quantity and sell_pool_id == tkn_buy:
            # add liquidity to sell_pool
            self.exchanges[sell_pool_id].add_liquidity(agent, sell_quantity, tkn_sell)
            return self
        elif sell_quantity and buy_pool_id == tkn_sell:
            # remove liquidity from buy_pool
            self.exchanges[buy_pool_id].remove_liquidity(agent, sell_quantity, tkn_buy)
            return self
        elif tkn_sell == 'LRNA':
            # this would be in the case that you are buying a stableswap asset with LRNA
            stable_pool: StableSwapPoolState = self.exchanges[buy_pool_id]
            if buy_quantity:
                # buy a specific quantity of a stableswap asset using LRNA
                shares_needed = stable_pool.calculate_withdrawal_shares(tkn_remove=tkn_buy, quantity=buy_quantity)
                omnipool.swap(agent, tkn_sell='LRNA', buy_quantity=shares_needed, tkn_buy=buy_pool_id)
                if omnipool.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(omnipool.fail)
                stable_pool.withdraw_asset(agent, buy_quantity, tkn_buy)
                return self
            elif sell_quantity:
                # sell a specific quantity of LRNA for a stableswap asset
                agent_shares = agent.holdings[buy_pool_id]
                omnipool.swap(
                    agent=agent,
                    tkn_buy=buy_pool_id, tkn_sell='LRNA',
                    sell_quantity=sell_quantity
                )
                if omnipool.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(omnipool.fail)
                delta_shares = agent.holdings[stable_pool.unique_id] - agent_shares
                stable_pool.remove_liquidity(agent, delta_shares, tkn_buy)
                return self

        elif buy_pool_id == self.omnipool_id != sell_pool_id:
            stable_pool: StableSwapPoolState = self.exchanges[sell_pool_id]
            if sell_quantity:
                # sell a stableswap asset for an omnipool asset
                agent_shares = agent.holdings[stable_pool.unique_id] if stable_pool.unique_id in agent.holdings else 0
                stable_pool.add_liquidity(agent, sell_quantity, tkn_sell)
                if stable_pool.fail:
                    # the transaction failed.
                    return self.fail_transaction(stable_pool.fail)
                delta_shares = agent.holdings[stable_pool.unique_id] - agent_shares
                omnipool.swap(
                    agent=agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=stable_pool.unique_id,
                    sell_quantity=delta_shares
                )
                return self
            elif buy_quantity:
                # buy an omnipool asset with a stableswap asset
                sell_shares = omnipool.calculate_sell_from_buy(tkn_buy, stable_pool.unique_id, buy_quantity)
                if sell_shares < 0:
                    return self.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.")
                elif sell_shares > stable_pool.shares:
                    return self.fail_transaction("Not enough shares in the stableswap pool.")
                stable_pool.buy_shares(agent, sell_shares, tkn_sell)
                if stable_pool.fail:
                    return self.fail_transaction(stable_pool.fail)
                omnipool.swap(agent, tkn_buy, stable_pool.unique_id, buy_quantity)
                return self

        elif sell_pool_id == self.omnipool_id != buy_pool_id:
            stable_pool: StableSwapPoolState = self.exchanges[buy_pool_id]
            if buy_quantity:
                # buy a stableswap asset with an omnipool asset
                shares_traded = stable_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)

                # buy shares in the subpool
                omnipool.swap(agent, tkn_buy=stable_pool.unique_id, tkn_sell=tkn_sell, buy_quantity=shares_traded)
                if omnipool.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(omnipool.fail)
                # withdraw the shares for the desired token
                stable_pool.withdraw_asset(agent, quantity=buy_quantity, tkn_remove=tkn_buy)
                if stable_pool.fail:
                    return self.fail_transaction(stable_pool.fail)
                return self
            elif sell_quantity:
                # sell an omnipool asset for a stableswap asset
                agent_shares = agent.holdings[stable_pool.unique_id] if stable_pool.unique_id in agent.holdings else 0
                omnipool.swap(
                    agent=agent,
                    tkn_buy=stable_pool.unique_id,
                    tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )
                delta_shares = agent.holdings[stable_pool.unique_id] - agent_shares
                if omnipool.fail:
                    return self.fail_transaction(omnipool.fail)
                stable_pool.remove_liquidity(
                    agent=agent, shares_removed=delta_shares, tkn_remove=tkn_buy
                )
                return self
        elif sell_pool_id != self.omnipool_id != buy_pool_id:
            # trade between two stableswap pools
            pool_buy: StableSwapPoolState = self.exchanges[buy_pool_id]
            pool_sell: StableSwapPoolState = self.exchanges[sell_pool_id]
            if buy_quantity:
                # buy enough shares of tkn_sell to afford buy_quantity worth of tkn_buy
                shares_bought = pool_buy.calculate_withdrawal_shares(tkn_buy, buy_quantity)
                if shares_bought > pool_buy.liquidity[tkn_buy]:
                    return self.fail_transaction(f'Not enough liquidity in {pool_buy.unique_id}: {tkn_buy}.')
                shares_sold = omnipool.calculate_sell_from_buy(
                    tkn_buy=pool_buy.unique_id,
                    tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                pool_sell.buy_shares(
                    agent=agent, quantity=shares_sold,
                    tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail)
                omnipool.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                if omnipool.fail:
                    return self.fail_transaction(omnipool.fail)
                pool_buy.withdraw_asset(
                    agent=agent, quantity=buy_quantity,
                    tkn_remove=tkn_buy, fail_on_overdraw=False
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail)

                # if all three parts succeeded, then we're good!
                return self
            elif sell_quantity:
                agent_sell_holdings = agent.holdings[sell_pool_id] if sell_pool_id in agent.holdings else 0
                pool_sell.add_liquidity(
                    agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail)
                delta_sell_holdings = agent.holdings[sell_pool_id] - agent_sell_holdings
                agent_buy_holdings = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
                omnipool.swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    sell_quantity=delta_sell_holdings
                )
                if omnipool.fail:
                    return self.fail_transaction(omnipool.fail)
                delta_buy_holdings = agent.holdings[buy_pool_id] - agent_buy_holdings
                pool_buy.remove_liquidity(
                    agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail)
                return self

    def find_routes(self, tkn_buy, tkn_sell) -> list[tuple[str, str]]:
        """
        Finds all possible routes to swap between tkn_buy and tkn_sell
        The tuples in the list are in the order of (sell_pool_id, buy_pool_id)
        """
        tkn_buy_pools = [pool_id for pool_id in self.exchanges if tkn_buy in self.exchanges[pool_id].asset_list]
        tkn_sell_pools = [pool_id for pool_id in self.exchanges if tkn_sell in self.exchanges[pool_id].asset_list]

        if len(tkn_buy_pools) == 0:
            raise ValueError(f'No pool with {tkn_buy} in asset list')
        if len(tkn_sell_pools) == 0:
            raise ValueError(f'No pool with {tkn_sell} in asset list')

        return [(tkn_sell_pool, tkn_buy_pool) for tkn_buy_pool in tkn_buy_pools for tkn_sell_pool in tkn_sell_pools]

    def find_best_route(self, tkn_buy, tkn_sell) -> tuple[str, str]:
        """
        Finds route to swap between tkn_buy and tkn_sell with the lowest spot price
        Returns tuple in the order of (sell_pool_id, buy_pool_id)
        """
        routes = self.find_routes(tkn_buy, tkn_sell)
        return sorted(routes, key=lambda x: self.price_route(tkn_buy, tkn_sell, x[1], x[0]))[0]

    def swap(self, agent, tkn_buy, tkn_sell, buy_quantity: float = 0, sell_quantity: float = 0):
        """Does swap along whatever route has best spot price"""
        if not buy_quantity and not sell_quantity:
            return self
        route = self.find_best_route(tkn_buy, tkn_sell)
        return self.swap_route(
            agent,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            buy_pool_id=route[1],
            sell_pool_id=route[0]
        )

    def simulate_swap_route(self,
        agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0,
        buy_pool_id: str = None,
        sell_pool_id: str = None):
        """Does swap along specified route, returning new router and agent"""
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap_route(
            new_agent,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            buy_pool_id=buy_pool_id,
            sell_pool_id=sell_pool_id
        )
        return new_state, new_agent

    def simulate_swap(self, agent, tkn_buy, tkn_sell, buy_quantity=0, sell_quantity=0):
        """Does swap along whatever route has best spot price"""
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap(
            agent=new_agent,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity
        )
        return new_state, new_agent
