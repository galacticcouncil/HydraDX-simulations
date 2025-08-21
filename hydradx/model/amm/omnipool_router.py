import copy

from .agents import Agent
from .exchange import Exchange
from .omnipool_amm import OmnipoolState
from .stableswap_amm import StableSwapPoolState
from typing import Literal

class Trade:
    def __init__(self, exchange: Exchange, tkn_sell: str, tkn_buy: str, trade_type: Literal['buy', 'sell']):
        self.exchange = exchange
        self.tkn_sell = tkn_sell
        self.tkn_buy = tkn_buy
        self.trade_type = trade_type  # 'buy' or 'sell'
        if trade_type not in ['buy', 'sell']:
            raise ValueError("trade_type must be 'buy' or 'sell'")

    def price(self) -> float:
        exchange = self.exchange
        tkn_buy = self.tkn_buy
        tkn_sell = self.tkn_sell
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                return exchange.withdraw_asset_spot(tkn_remove=tkn_sell)
            else:
                return exchange.remove_liquidity_spot(tkn_remove=tkn_buy)
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                return exchange.buy_shares_spot(tkn_add=tkn_sell)
            else:
                return exchange.add_liquidity_spot(tkn_add=tkn_sell)
        else:
            if self.trade_type == "buy":
                return exchange.buy_spot(tkn_buy, tkn_sell)
            else:
                return exchange.sell_spot(tkn_sell, tkn_buy)


    def execute(self, agent: Agent, quantity: float):
        exchange = self.exchange
        tkn_buy = self.tkn_buy
        tkn_sell = self.tkn_sell
        start_quantity = agent.get_holdings(tkn_buy) if self.trade_type == "buy" else agent.get_holdings(tkn_sell)
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                exchange.withdraw_asset(
                    agent=agent,
                    quantity=quantity,
                    tkn_remove=tkn_sell
                )
            else:
                exchange.remove_liquidity(
                    agent=agent,
                    shares_removed=quantity,
                    tkn_remove=tkn_buy
                )
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                exchange.buy_shares(
                    agent=agent,
                    quantity=quantity,
                    tkn_add=tkn_sell
                )
            else:
                exchange.add_liquidity(
                    agent=agent,
                    quantity=quantity,
                    tkn_add=tkn_sell
                )
        else:
            if self.trade_type == "buy":
                exchange.swap(
                    agent=agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=tkn_sell,
                    buy_quantity=quantity
                )
            else:
                exchange.swap(
                    agent=agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=tkn_sell,
                    sell_quantity=quantity
                )
        return (agent.get_holdings(tkn_buy) if self.trade_type == "buy" else agent.get_holdings(tkn_sell)) - start_quantity

    def calculate(self, quantity):
        exchange = self.exchange
        tkn_buy = self.tkn_buy
        tkn_sell = self.tkn_sell
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                return exchange.calculate_withdrawal_asset(tkn_remove=tkn_sell, quantity=quantity)
            else:
                return exchange.calculate_remove_liquidity(tkn_remove=tkn_buy, shares_removed=quantity)
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if self.trade_type == "buy":
                return exchange.calculate_buy_shares(tkn_add=tkn_sell, quantity=quantity)
            else:
                return exchange.calculate_add_liquidity(tkn_add=tkn_sell, quantity=quantity)
        else:
            if self.trade_type == "buy":
                return exchange.calculate_buy_from_sell(tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=quantity)
            else:
                return exchange.calculate_sell_from_buy(tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=quantity)


def price_route(
        route: list[Trade]
) -> float:
    price = 1.0
    for trade in route:
        price *= trade.price()
    return price


class OmnipoolRouter(Exchange):
    """
    Handles routing between Omnipool and Stableswap subpools that have LP assets in Omnipool
    """

    def __init__(self, exchanges: dict or list, unique_id: str = 'omnipool_router'):
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
        self.unique_id = unique_id

    def copy(self):
        copy_self = copy.deepcopy(self)
        return copy_self

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
        return price_route(self.find_best_route(tkn_buy, tkn_sell, trade_type='buy'))

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        return price_route(self.find_best_route(tkn_buy, tkn_sell, trade_type='sell'))

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
        tkn_intermediate: str = None,
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
        sell_pool = self.exchanges[sell_pool_id]
        buy_pool = self.exchanges[buy_pool_id]

        if buy_pool == sell_pool:  # just call the swap function of the pool
            buy_pool.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_quantity, sell_quantity=sell_quantity)
            return self
        elif buy_quantity and buy_pool_id == tkn_sell:
            # we are selling LP token for tkn_sell, i.e. withdrawing asset
            buy_pool.withdraw_asset(agent, buy_quantity, tkn_buy)
            return self
        elif buy_quantity and sell_pool_id == tkn_buy:
            # we are just buying LP token for tkn_sell, i.e. buying shares
            sell_pool.buy_shares(agent, buy_quantity, tkn_sell)
            return self
        elif sell_quantity and sell_pool_id == tkn_buy:
            # add liquidity to sell_pool
            sell_pool.add_liquidity(agent, sell_quantity, tkn_sell)
            return self
        elif sell_quantity and buy_pool_id == tkn_sell:
            # remove liquidity from buy_pool
            buy_pool.remove_liquidity(agent, sell_quantity, tkn_buy)
            return self
        elif tkn_intermediate:
            agent_holdings = agent.get_holdings(tkn_intermediate)
            if sell_quantity:
                sell_pool.swap(
                    agent=agent,
                    tkn_sell=tkn_sell,
                    tkn_buy=tkn_intermediate,
                    sell_quantity=sell_quantity
                )
                if sell_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                buy_quantity = agent.get_holdings(tkn_intermediate) - agent_holdings
                buy_pool.swap(
                    agent=agent,
                    tkn_buy=tkn_buy, tkn_sell=tkn_intermediate,
                    sell_quantity=buy_quantity
                )
            elif buy_quantity:
                buy_pool.swap(
                    agent=agent,
                    tkn_buy=tkn_buy, tkn_sell=tkn_intermediate,
                    buy_quantity=buy_quantity
                )
                if buy_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                sell_quantity=agent.get_holdings(tkn_intermediate) - agent_holdings
                sell_pool.swap(
                    agent=agent,
                    tkn_buy=tkn_intermediate, tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )

        elif tkn_sell == 'LRNA':
            if isinstance(sell_pool, OmnipoolState) and isinstance(buy_pool, StableSwapPoolState):
                if buy_quantity:
                    # buy a specific quantity of a stableswap asset using LRNA
                    shares_needed = buy_pool.calculate_withdrawal_shares(tkn_remove=tkn_buy, quantity=buy_quantity)
                    sell_pool.swap(agent, tkn_sell='LRNA', buy_quantity=shares_needed, tkn_buy=buy_pool_id)
                    if sell_pool.fail:
                        # if the swap failed, the transaction failed.
                        return self.fail_transaction(sell_pool.fail)
                    buy_pool.withdraw_asset(agent, buy_quantity, tkn_buy)
                    return self
                elif sell_quantity:
                    # sell a specific quantity of LRNA for a stableswap asset
                    agent_shares = agent.holdings[buy_pool_id]
                    sell_pool.swap(
                        agent=agent,
                        tkn_buy=buy_pool_id, tkn_sell='LRNA',
                        sell_quantity=sell_quantity
                    )
                    if sell_pool.fail:
                        # if the swap failed, the transaction failed.
                        return self.fail_transaction(sell_pool.fail)
                    delta_shares = agent.holdings[buy_pool_id] - agent_shares
                    buy_pool.remove_liquidity(agent, delta_shares, tkn_buy)
                    return self

        elif buy_pool == self.omnipool and isinstance(sell_pool, StableSwapPoolState):
            if sell_quantity:
                # sell a stableswap asset for an omnipool asset
                agent_shares = agent.holdings[sell_pool_id] if sell_pool_id in agent.holdings else 0
                sell_pool.add_liquidity(agent, sell_quantity, tkn_sell)
                if sell_pool.fail:
                    # the transaction failed.
                    return self.fail_transaction(sell_pool.fail)
                delta_shares = agent.holdings[sell_pool_id] - agent_shares
                buy_pool.swap(
                    agent=agent,
                    tkn_buy=tkn_buy, tkn_sell=sell_pool_id,
                    sell_quantity=delta_shares
                )
                return self
            elif buy_quantity:
                # buy an omnipool asset with a stableswap asset
                sell_shares = buy_pool.calculate_sell_from_buy(tkn_buy, sell_pool_id, buy_quantity)
                if sell_shares < 0:
                    return self.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.")
                elif sell_shares > sell_pool.shares:
                    return self.fail_transaction("Not enough shares in the stableswap pool.")
                sell_pool.buy_shares(agent, sell_shares, tkn_sell)
                if sell_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                buy_pool.swap(agent, tkn_buy, sell_pool_id, buy_quantity)
                return self

        elif sell_pool == self.omnipool and isinstance(buy_pool, StableSwapPoolState):
            if buy_quantity:
                # buy a stableswap asset with an omnipool asset
                shares_traded = buy_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)

                # buy shares in the subpool
                sell_pool.swap(agent, tkn_buy=buy_pool_id, tkn_sell=tkn_sell, buy_quantity=shares_traded)
                if sell_pool.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(sell_pool.fail)
                # withdraw the shares for the desired token
                buy_pool.withdraw_asset(agent, quantity=buy_quantity, tkn_remove=tkn_buy)
                if buy_pool.fail:
                    return self.fail_transaction(buy_pool.fail)
                return self
            elif sell_quantity:
                # sell an omnipool asset for a stableswap asset
                agent_shares = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
                sell_pool.swap(
                    agent=agent,
                    tkn_buy=buy_pool_id,
                    tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )
                delta_shares = agent.holdings[buy_pool_id] - agent_shares
                if sell_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                buy_pool.remove_liquidity(
                    agent=agent, shares_removed=delta_shares, tkn_remove=tkn_buy
                )
                return self

        elif sell_pool != self.omnipool != buy_pool:
            # trade between two stableswap pools
            omnipool: OmnipoolState = self.omnipool
            if buy_quantity:
                # buy enough shares of tkn_sell to afford buy_quantity worth of tkn_buy
                shares_bought = buy_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)
                if shares_bought > buy_pool.liquidity[tkn_buy]:
                    return self.fail_transaction(f'Not enough liquidity in {buy_pool_id}: {tkn_buy}.')
                shares_sold = omnipool.calculate_sell_from_buy(
                    tkn_buy=buy_pool_id,
                    tkn_sell=sell_pool_id,
                    buy_quantity=shares_bought
                )
                sell_pool.buy_shares(
                    agent=agent, quantity=shares_sold,
                    tkn_add=tkn_sell
                )
                if sell_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                omnipool.swap(
                    agent=agent,
                    tkn_buy=buy_pool_id, tkn_sell=sell_pool_id,
                    buy_quantity=shares_bought
                )
                if omnipool.fail:
                    return self.fail_transaction(omnipool.fail)
                buy_pool.withdraw_asset(
                    agent=agent, quantity=buy_quantity,
                    tkn_remove=tkn_buy, fail_on_overdraw=False
                )
                if buy_pool.fail:
                    return self.fail_transaction(buy_pool.fail)

                # if all three parts succeeded, then we're good!
                return self
            elif sell_quantity:
                agent_sell_holdings = agent.holdings[sell_pool_id] if sell_pool_id in agent.holdings else 0
                sell_pool.add_liquidity(
                    agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
                )
                if sell_pool.fail:
                    return self.fail_transaction(sell_pool.fail)
                delta_sell_holdings = agent.holdings[sell_pool_id] - agent_sell_holdings
                agent_buy_holdings = agent.holdings[buy_pool_id] if buy_pool_id in agent.holdings else 0
                omnipool.swap(
                    agent=agent,
                    tkn_buy=buy_pool.unique_id, tkn_sell=sell_pool_id,
                    sell_quantity=delta_sell_holdings
                )
                if omnipool.fail:
                    return self.fail_transaction(omnipool.fail)
                delta_buy_holdings = agent.holdings[buy_pool_id] - agent_buy_holdings
                buy_pool.remove_liquidity(
                    agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
                )
                if buy_pool.fail:
                    return self.fail_transaction(buy_pool.fail)
                return self
        raise ValueError(
            f"Invalid swap route: sell_pool_id={sell_pool_id}, buy_pool_id={buy_pool_id}, tkn_buy={tkn_buy}, tkn_sell={tkn_sell}"
        )

    def find_routes(self, tkn_buy: str, tkn_sell: str, trade_type: Literal['buy', 'sell']='sell') -> list[list[Trade]]:
        """
        Finds all possible routes to swap between tkn_buy and tkn_sell
        The tuples in the list are in the order of (sell_pool_id, buy_pool_id)
        """
        tkn_buy_pools: list[Exchange] = [pool for pool in self.exchanges.values() if tkn_buy in pool.asset_list or pool.unique_id == tkn_buy]
        tkn_sell_pools: list[Exchange] = [pool for pool in self.exchanges.values() if tkn_sell in pool.asset_list or pool.unique_id == tkn_sell]

        if len(tkn_buy_pools) == 0:
            raise ValueError(f'No pool with {tkn_buy} in asset list')
        if len(tkn_sell_pools) == 0:
            raise ValueError(f'No pool with {tkn_sell} in asset list')

        routes: list[list[Trade]] = []
        for pool in list(set(tkn_buy_pools) & set(tkn_sell_pools)):
            routes.append([Trade(
                exchange=pool,
                tkn_sell=tkn_sell,
                tkn_buy=tkn_buy,
                trade_type=trade_type
            )])
        for pool1, pool2 in [(tkn_sell_pool, tkn_buy_pool) for tkn_buy_pool in tkn_buy_pools for tkn_sell_pool in tkn_sell_pools]:
            if pool1.unique_id in pool2.asset_list:
                routes.append([Trade(
                    exchange=pool2,
                    tkn_sell=tkn_sell,
                    tkn_buy=pool1.unique_id,
                    trade_type=trade_type
                ), Trade(
                    exchange=pool1,
                    tkn_sell=pool1.unique_id,
                    tkn_buy=tkn_buy,
                    trade_type=trade_type
                )])
            elif pool2.unique_id in pool1.asset_list:
                routes.append([Trade(
                    exchange=pool1,
                    tkn_sell=tkn_sell,
                    tkn_buy=pool2.unique_id,
                    trade_type=trade_type
                ), Trade(
                    exchange=pool2,
                    tkn_sell=pool2.unique_id,
                    tkn_buy=tkn_buy,
                    trade_type=trade_type
                )])
            else:
                intermediaries = set(pool1.asset_list) & set(pool2.asset_list) - {tkn_buy, tkn_sell}
                if len(intermediaries) == 0:
                    continue
                best_tkn = min(intermediaries, key=lambda x: pool1.price(tkn_sell, x) / pool2.price(tkn_buy, x))
                routes.append([Trade(
                    exchange=pool1,
                    tkn_sell=tkn_sell,
                    tkn_buy=best_tkn,
                    trade_type=trade_type
                ), Trade(
                    exchange=pool2,
                    tkn_sell=best_tkn,
                    tkn_buy=tkn_buy,
                    trade_type=trade_type
                )])
            if trade_type == 'buy':
                routes[-1].reverse()

        return routes

    def find_best_route(self, tkn_buy, tkn_sell, trade_type: Literal['buy', 'sell']='sell') -> list[Trade]:
        """
        Finds route to swap between tkn_buy and tkn_sell with the lowest spot price
        Returns tuple in the order of (sell_pool_id, buy_pool_id)
        """
        routes = self.find_routes(tkn_buy, tkn_sell, trade_type)
        return min(routes, key=lambda x: price_route(x))

    def swap(self, agent, tkn_buy, tkn_sell, buy_quantity: float = 0, sell_quantity: float = 0):
        """Does swap along whatever route has best spot price"""
        if not buy_quantity and not sell_quantity:
            return self
        route = self.find_best_route(tkn_buy, tkn_sell, trade_type='buy' if buy_quantity else 'sell')
        quantity = buy_quantity if buy_quantity else sell_quantity
        for swap in route:
            quantity = swap.execute(agent, quantity)
        return self


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
