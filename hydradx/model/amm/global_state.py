from .agents import Agent
import copy
import random


class AMM:
    def __init__(self):
        self.fail = ''

    def copy(self):
        copy_self = copy.deepcopy(self)
        copy_self.fail = ''
        return copy_self

    def swap(
        self,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
    ):
        return self.copy(), old_agent.copy()

    def add_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_add: str
    ):
        return self.copy(), old_agent.copy()

    def remove_liquidity(
        self,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
    ):
        return self.copy(), old_agent.copy()

    def fail(self, agent: Agent, error: str = 'fail'):
        failed_state = self.copy()
        failed_state.fail = error
        return failed_state, agent.copy()


class GlobalState:
    def __init__(self, agents: dict[str: Agent], pools: dict[str: AMM], external_market: dict[str: float]):
        # get a list of all assets contained in any member of the state
        self.asset_list = list(set(
            [asset for pool in pools for asset in pool.liquidity.keys()]
            + [asset for agent in agents for asset in agent.asset_list]
            + list(external_market.keys())
        ))
        self.agents = agents
        for agent_name in self.agents:
            self.agents[agent_name].unique_id = agent_name
        self.pools = pools
        for pool_name in self.pools:
            self.pools[pool_name].unique_id = pool_name
        self.external_market = external_market

    def price(self, asset):
        return self.external_market[asset] if asset in self.external_market else 0

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy


def fluctuate_prices(state: GlobalState, percent: float, bias: float):
    new_state = state.copy()
    for asset in new_state.external_market:
        new_state.external_market[asset] *= (
                1 / (1 + percent / 100)
                + random.random() * (1 / (1 + percent / 100) + percent / 100)
                + bias
        )
    return new_state


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
    new_state = old_state  # .copy()
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
    new_state = old_state  # .copy()
    new_state.pools[pool_id], new_state.agents[agent_id] = new_state.pools[pool_id].add_liquidity(
        old_state=new_state.pools[pool_id],
        old_agent=new_state.agents[agent_id],
        quantity=quantity,
        tkn_remove=tkn_remove
    )
    return new_state


def withdraw_all_liquidity(state: GlobalState, agent: Agent) -> GlobalState:
    # I am not currently having these functions make or return a copy of the state, because I found that makes it slow
    for i in agent.shares.keys():
        remove_liquidity(state, i, agent.unique_id, agent.shares[i], tkn_remove=i)

    return state


def value_assets(state: GlobalState, agent: Agent) -> float:
    return sum([
        agent.holdings[i] * state.price(i)
        for i in agent.holdings.keys()
    ])


def cash_out(state: GlobalState, agent: Agent) -> float:
    withdraw_all_liquidity(state, agent)
    return value_assets(state, agent)
