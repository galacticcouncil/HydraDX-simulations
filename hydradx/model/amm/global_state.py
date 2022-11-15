from .agents import Agent
import copy
import random
from .amm import AMM
from typing import Callable


class GlobalState:
    def __init__(self,
                 agents: dict[str: Agent],
                 pools: dict[str: AMM],
                 external_market: dict[str: float] = None,
                 evolve_function: Callable = None
                 ):
        if external_market is None:
            self.external_market = {}
        else:
            self.external_market = external_market
        # get a list of all assets contained in any member of the state
        self.asset_list = list(set(
            [asset for pool in pools.values() for asset in pool.liquidity.keys()]
            + [asset for agent in agents.values() for asset in agent.asset_list]
            + list(self.external_market.keys())
        ))
        self.agents = agents
        for agent_name in self.agents:
            self.agents[agent_name].unique_id = agent_name
        self.pools = pools
        for pool_name in self.pools:
            self.pools[pool_name].unique_id = pool_name
        if 'USD' not in self.external_market:
            self.external_market['USD'] = 1  # default denomination
        for agent in self.agents.values():
            for asset in self.asset_list:
                if asset not in agent.holdings:
                    agent.holdings[asset] = 0
        self._evolve_function = evolve_function
        self.evolve_function = evolve_function.__name__ if evolve_function else 'None'
        self.time_step = 0

    def price(self, asset: str):
        if asset in self.external_market:
            return self.external_market[asset]
        else:
            return 0

    def total_wealth(self):
        return sum([
            sum([
                agent.holdings[tkn] * self.price(tkn) for tkn in agent.holdings
            ])
            for agent in self.agents.values()
        ]) + sum([
            sum([
                pool.liquidity[tkn] * self.price(tkn) for tkn in pool.asset_list
            ])
            for pool in self.pools.values()
        ])

    def total_asset(self, tkn):
        return (
            sum([pool.liquidity[tkn] if tkn in pool.liquidity else 0 for pool in self.pools.values()])
            + sum([agent.holdings[tkn] if tkn in agent.holdings else 0 for agent in self.agents.values()])
        )

    def total_assets(self):
        return {tkn: self.total_asset(tkn) for tkn in self.asset_list}

    def copy(self):
        copy_state = GlobalState(
            agents={agent_id: self.agents[agent_id].copy() for agent_id in self.agents},
            pools={pool_id: self.pools[pool_id].copy() for pool_id in self.pools},
            external_market=copy.copy(self.external_market),
            evolve_function=copy.copy(self._evolve_function)
        )
        copy_state.time_step = self.time_step
        return copy_state

    def evolve(self):
        self.time_step += 1
        if self._evolve_function:
            return self._evolve_function(self)

    def execute_swap(
            self,
            pool_id: str,
            agent_id: str,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        self.pools[pool_id], self.agents[agent_id] = self.pools[pool_id].swap(
            old_state=self.pools[pool_id],
            old_agent=self.agents[agent_id],
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity
        )
        return self

    def __repr__(self):
        newline = "\n"
        return (
            f'global state {newline}'
            f'pools: {newline}{newline.join([repr(pool) for pool in self.pools.values()])}'
            f'{newline}'
            f'agents: {newline}{newline.join([repr(agent) for agent in self.agents.values()])}'
            f'{newline}'
            f'evolution function: {self.evolve_function}'
            f'{newline}'
        )


def fluctuate_prices(volatility: dict[str: float], trend: dict[str: float] = None):
    """
        Volatility is in the form of
        {tkn: percentage for tkn in asset_list}
        and is the maximum percentage by which the asset price will vary per step.

        Trend is also in the form
        {tkn: percentage for tkn in asset_list}
        and is the percentage by the which the price will predictably move, every step.

        Approximately:
        price[tkn] += random.random() * volatility[tkn] / 100 + trend[tkn] / 100
    """
    trend = trend or {}

    def transform(state: GlobalState) -> GlobalState:
        new_state = state  # .copy()
        for asset in new_state.external_market:
            change = volatility[asset] if asset in volatility else 0
            bias = trend[asset] / 100 if asset in trend else 0
            # not exactly the same as above, because adding 1% and then subtracting 1% does not get us back to 100%
            # instead, we need to subtract (100/101)% to avoid a long-term downward trend
            new_state.external_market[asset] *= (
                (1 + random.random() * change / 100 if random.choice([True, False])
                 else 1 - random.random() * (1 - 100 / (100 + change)))
                # 1 / (1 + change / 100)
                # + random.random() * (1 - 1 / (1 + change / 100) + change / 100)
                + bias / 100
            )
        return new_state

    return transform


def oscillate_prices(volatility: dict[str: float], trend: dict[str: float] = None, period: int = 1) -> Callable:
    # steadily oscillate, no unpredictable motion
    class UpDown:
        def __init__(self, magnitude, wavelength, bias):
            self.bias = bias
            self.inertia = 0
            self.wavelength = wavelength
            self.direction = magnitude

    trend = trend or {}
    updown: dict[str: UpDown] = {}
    for token in volatility:
        updown[token] = UpDown(
            wavelength=period,
            magnitude=volatility[token],
            bias=trend[token] if token in trend else 0
        )

    def transform(state: GlobalState) -> GlobalState:
        for tkn in updown:
            if abs(updown[tkn].inertia) >= updown[tkn].wavelength:
                # reverse trend
                updown[tkn].direction = (updown[tkn].direction + 1) * -1 + 1
                updown[tkn].inertia = 0
            state.external_market[tkn] += (
                updown[tkn].direction / 100 / updown[tkn].wavelength
                + updown[tkn].bias / 100 / updown[tkn].wavelength
            )
            updown[tkn].inertia += 1
        return state

    return transform


def swap(
    old_state: GlobalState,
    pool_id: str,
    agent_id: str,
    tkn_sell: str,
    tkn_buy: str,
    buy_quantity: float = 0,
    sell_quantity: float = 0
) -> GlobalState:
    """
    copy state, execute swap, return swapped state
    """
    return old_state.copy().execute_swap(
        pool_id=pool_id,
        agent_id=agent_id,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity
    )


def add_liquidity(
    old_state: GlobalState,
    pool_id: str,
    agent_id: str,
    quantity: float,
    tkn_add: str
) -> GlobalState:
    """
    copy state, execute add liquidity
    """
    new_state = old_state.copy()
    # add liquidity to sub_pools through main pool
    if pool_id not in new_state.pools:
        for pool in new_state.pools.values():
            if hasattr(pool, 'sub_pools') and pool_id in pool.sub_pools:
                pool_id = pool.unique_id

    new_state.pools[pool_id], new_state.agents[agent_id] = new_state.pools[pool_id].add_liquidity(
        old_state=new_state.pools[pool_id],
        old_agent=new_state.agents[agent_id],
        quantity=quantity,
        tkn_add=tkn_add
    )
    return new_state


def remove_liquidity(
    old_state: GlobalState,
    pool_id: str,
    agent_id: str,
    quantity: float,
    tkn_remove: str
) -> GlobalState:

    new_state = old_state.copy()
    new_state.pools[pool_id], new_state.agents[agent_id] = new_state.pools[pool_id].remove_liquidity(
        old_state=new_state.pools[pool_id],
        old_agent=new_state.agents[agent_id],
        quantity=quantity,
        tkn_remove=tkn_remove
    )
    return new_state


def withdraw_all_liquidity(state: GlobalState, agent_id: str) -> GlobalState:
    agent = state.agents[agent_id]
    new_state = state
    for key in agent.holdings.keys():
        # shares.keys might just be the pool name, or it might be a tuple (pool, token)
        if isinstance(key, tuple):
            pool_id = key[0]
            tkn = key[1]
        else:
            pool_id = key
            tkn = key
        if pool_id in state.pools:
            new_state = remove_liquidity(new_state, pool_id, agent.unique_id, agent.holdings[key], tkn_remove=tkn)

    return new_state


def external_market_trade(
    old_state: GlobalState,
    agent_id: str,
    tkn_buy: str,
    tkn_sell: str,
    buy_quantity: float = 0,
    sell_quantity: float = 0
) -> GlobalState:

    # do a trade at spot price on the external market
    # this should maybe only work in USD, cause we're probably talking about coinbase or something
    new_state = old_state.copy()
    agent = new_state.agents[agent_id]
    if buy_quantity:
        sell_quantity = buy_quantity * new_state.price(tkn_buy) / new_state.price(tkn_sell)
    elif sell_quantity:
        buy_quantity = sell_quantity * new_state.price(tkn_sell) / new_state.price(tkn_buy)
    else:
        # raise TypeError('Must include either buy_quantity or sell_quantity.')
        return old_state

    if tkn_buy not in agent.holdings:
        agent.holdings[tkn_buy] = 0

    if agent.holdings[tkn_sell] - sell_quantity < 0:
        # insufficient funds, reduce quantity to match
        sell_quantity = agent.holdings[tkn_sell]
    elif agent.holdings[tkn_buy] + buy_quantity < 0:
        # also insufficient funds
        buy_quantity = -agent.holdings[tkn_buy]

    # there could probably be a fee or something here, but for now you can sell infinite quantities for free
    agent.holdings[tkn_buy] += buy_quantity
    agent.holdings[tkn_sell] -= sell_quantity

    return new_state


def migrate(
    old_state: GlobalState,
    pool_id: str,
    sub_pool_id: str,
    tkn_migrate: str
) -> GlobalState:
    if not hasattr(old_state.pools[pool_id], 'execute_migration'):
        raise AttributeError(f"Pool {pool_id} does not implement migrations.")
    new_state = old_state.copy()
    new_state.pools[pool_id] = new_state.pools[pool_id].execute_migration(
        tkn_migrate=tkn_migrate,
        sub_pool_id=sub_pool_id
    )
    return new_state
