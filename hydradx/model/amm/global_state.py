from .agents import Agent
import copy
import random
from .amm import AMM
from typing import Callable


class GlobalState:
    def __init__(self,
                 agents: dict[str: Agent],
                 pools: dict[str: AMM],
                 external_market: dict[str: float] = {},
                 evolve_function: Callable = None
                 ):
        # get a list of all assets contained in any member of the state
        self.asset_list = list(set(
            [asset for pool in pools.values() for asset in pool.liquidity.keys()]
            + [asset for agent in agents.values() for asset in agent.asset_list]
            + list(external_market.keys())
        ))
        self.agents = agents
        for agent_name in self.agents:
            self.agents[agent_name].unique_id = agent_name
        self.pools = pools
        for pool_name in self.pools:
            self.pools[pool_name].unique_id = pool_name
        self.external_market = external_market
        self._evolve_function = evolve_function
        self.evolve_function = evolve_function.__name__ if evolve_function else 'None'

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
        self_copy = copy.deepcopy(self)
        return self_copy

    def evolve(self):
        if self._evolve_function:
            return self._evolve_function(self)

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


def fluctuate_prices(percent: float, bias: float = 0):

    def transform(state: GlobalState) -> GlobalState:
        new_state = state  # .copy()
        for asset in new_state.external_market:
            new_state.external_market[asset] *= (
                    1 / (1 + percent / 100)
                    + random.random() * (1 - 1 / (1 + percent / 100) + percent / 100)
                    + bias / 100
            )
        return new_state

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
    new_state = old_state  # .copy()
    new_state.pools[pool_id], new_state.agents[agent_id] = new_state.pools[pool_id].swap(
        old_state=new_state.pools[pool_id],
        old_agent=new_state.agents[agent_id],
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity
    )
    return new_state


def add_liquidity(
    old_state: GlobalState,
    pool_id: str,
    agent_id: str,
    quantity: float,
    tkn_add: str
) -> GlobalState:
    new_state = old_state.copy()
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
    for key in agent.shares.keys():
        # shares.keys might just be the pool name, or it might be a tuple (pool, token)
        if isinstance(key, tuple):
            pool_id = key[0]
            tkn = key[1]
        else:
            pool_id = key
            tkn = key
        new_state = remove_liquidity(new_state, pool_id, agent.unique_id, agent.shares[key], tkn_remove=tkn)

    return new_state
