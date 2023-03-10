import copy
import random
from typing import Callable
from .omnipool_amm import OmnipoolState, calculate_remove_liquidity

from .agents import Agent, AgentArchiveState
from .amm import AMM, FeeMechanism
from .omnipool_amm import OmnipoolState, calculate_remove_liquidity, OmnipoolArchiveState


class GlobalState:
    def __init__(self,
                 agents: dict[str: Agent],
                 pools: dict[str: AMM],
                 external_market: dict[str: float] = None,
                 evolve_function: Callable = None,
                 save_data: dict = None,
                 archive_all: bool = True,
                 initial_block: int = 0
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
        self.datastreams = save_data
        self.save_data = {
            tag: save_data[tag].assemble(self)
            for tag in save_data
        } if save_data else {}
        self.time_step = 0
        self.archive_all = archive_all
        self.block_number = initial_block

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
            evolve_function=copy.copy(self._evolve_function),
            save_data=self.datastreams,
            archive_all=self.archive_all,
            initial_block=self.block_number
        )
        copy_state.time_step = self.time_step
        return copy_state

    def archive(self):
        if self.archive_all and not self.save_data:
            return self.copy()
        elif self.save_data:
            return {
                datastream: self.save_data[datastream](self)
                for datastream in self.save_data
            }
        else:
            return ArchiveState(self)

    def evolve(self):
        self.time_step += 1
        self.block_number += 1
        for pool in self.pools.values():
            pool.update()
        if self._evolve_function:
            return self._evolve_function(self)

    def execute_swap(
            self,
            pool_id: str,
            agent_id: str,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0,
            **kwargs
    ):
        self.pools[pool_id].execute_swap(
            state=self.pools[pool_id],
            agent=self.agents[agent_id],
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            **kwargs  # pass any additional arguments to the pool
        )
        return self

    # functions for calculating extra parameters we may want to track

    def market_prices(self, shares: dict) -> dict:
        """
        return the market price of each asset in state.asset_list, as well as the price of each asset in shares
        """
        prices = {tkn: self.price(tkn) for tkn in self.asset_list}
        for share_id in shares:
            # if shares are for a specific asset in a specific pool, get prices according to that pool
            if isinstance(share_id, tuple):
                pool_id = share_id[0]
                tkn_id = share_id[1]
                prices[share_id] = self.pools[pool_id].usd_price(self.pools[pool_id], tkn_id)

        return prices

    def cash_out(self, agent: Agent) -> float:
        """
        return the value of the agent's holdings if they withdraw all liquidity
        and then sell at current spot prices
        """
        if 'LRNA' not in agent.holdings:
            agent.holdings['LRNA'] = 0
        withdraw_holdings = {tkn: agent.holdings[tkn] for tkn in list(agent.holdings.keys())}

        for key in agent.holdings.keys():
            # shares.keys might just be the pool name, or it might be a tuple (pool, token)
            if isinstance(key, tuple):
                pool_id = key[0]
                tkn = key[1]
            else:
                pool_id = key
                tkn = key
            if pool_id in self.pools:
                if isinstance(self.pools[pool_id], OmnipoolState):
                    # optimized for omnipool, no copy operations
                    delta_qa, delta_r, delta_q, \
                        delta_s, delta_b, delta_l = calculate_remove_liquidity(
                        self.pools[pool_id],
                        agent,
                        agent.holdings[key],
                        tkn_remove=tkn
                    )
                    withdraw_holdings[key] = 0
                    withdraw_holdings['LRNA'] += delta_qa
                    withdraw_holdings[tkn] -= delta_r
                else:
                    # much less efficient, but works for any pool
                    new_state = remove_liquidity(self, pool_id, agent.unique_id, agent.holdings[key], tkn_remove=tkn)
                    new_agent = new_state.agents[agent.unique_id]
                    withdraw_holdings = {
                        tkn: withdraw_holdings[tkn] + new_agent.holdings[tkn] - agent.holdings[tkn]
                        for tkn in agent.holdings
                    }

        prices = self.market_prices(withdraw_holdings)
        return value_assets(prices, withdraw_holdings)

    def pool_val(self, pool: AMM):
        """ get the total value of all liquidity in the pool. """
        total = 0
        for asset in pool.asset_list:
            total += pool.liquidity[asset] * self.price(asset)
        return total

    def impermanent_loss(self, agent: Agent) -> float:
        return self.cash_out(agent) / self.deposit_val(agent) - 1

    def deposit_val(self, agent: Agent) -> float:
        return value_assets(
            self.market_prices(agent.holdings),
            agent.initial_holdings
        )

    def withdraw_val(self, agent: Agent) -> float:
        return self.cash_out(agent)

    def __repr__(self):
        newline = "\n"
        indent = '    '
        return (
                f'global state {newline}'
                f'pools: {newline + newline + indent}' +
                ((newline + indent).join([
                    (newline + indent).join(pool_desc.split('\n'))
                    for pool_desc in [repr(pool) for pool in self.pools.values()]
                ])) +
                newline + newline +
                f'agents: {newline + newline}    ' +
                ((newline + indent).join([
                    (newline + indent).join(agent_desc.split('\n'))
                    for agent_desc in [repr(agent) for agent in self.agents.values()]
                ])) + newline +
                f'market prices: {newline + newline}    ' +
                ((newline + indent).join([
                    f'{indent}{tkn}: ${price}' for tkn, price in self.external_market.items()
                ])) +
                f'{newline}{newline}'
                f'evolution function: {self.evolve_function}'
                f'{newline}'
        )


class ArchiveState:
    def __init__(self, state: GlobalState):
        self.time_step = state.time_step
        self.external_market = {k: v for k, v in state.external_market.items()}
        self.pools = {k: v.archive() for (k, v) in state.pools.items()}
        self.agents = {k: AgentArchiveState(v) for (k, v) in state.agents.items()}


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])


GlobalState.value_assets = staticmethod(value_assets)


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


def historical_prices(price_list: list[dict[str: float]]) -> Callable:
    def transform(state: GlobalState) -> GlobalState:
        for tkn in price_list[state.time_step]:
            state.external_market[tkn] = price_list[state.time_step][tkn]
        return state

    return transform


def swap(
        old_state: GlobalState,
        pool_id: str,
        agent_id: str,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0,
        **kwargs
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
        sell_quantity=sell_quantity,
        **kwargs  # pass through any extra arguments
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

    new_state.pools[pool_id].execute_add_liquidity(
        state=new_state.pools[pool_id],
        agent=new_state.agents[agent_id],
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
    # this should maybe only work in USD, because we're probably talking about coinbase or something
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
    if not hasattr(old_state.pools[pool_id], 'execute_migrate_asset'):
        raise AttributeError(f"Pool {pool_id} does not implement migrations.")
    new_state = old_state.copy()
    new_state.pools[pool_id] = new_state.pools[pool_id].execute_migrate_asset(
        tkn_migrate=tkn_migrate,
        sub_pool_id=sub_pool_id
    )
    return new_state


def migrate_lp(
        old_state: GlobalState,
        pool_id: str,
        agent_id: str,
        sub_pool_id: str,
        tkn_migrate: str
) -> GlobalState:
    if not hasattr(old_state.pools[pool_id], 'execute_migrate_lp'):
        raise AttributeError(f"Pool {pool_id} does not implement migrations.")
    new_state = old_state.copy()
    new_state.pools[pool_id], agent = new_state.pools[pool_id].execute_migrate_lp(
        agent=new_state.agents[agent_id],
        tkn_migrate=tkn_migrate,
        sub_pool_id=sub_pool_id
    )
    return new_state


def create_sub_pool(
        old_state: GlobalState,
        pool_id: str,
        sub_pool_id: str,
        tkns_migrate: list[str],
        amplification: float,
        trade_fee: FeeMechanism or float
):
    new_state = old_state.copy()
    new_pool = new_state.pools[pool_id]
    new_pool.execute_create_sub_pool(
        tkns_migrate=tkns_migrate,
        sub_pool_id=sub_pool_id,
        amplification=amplification,
        trade_fee=trade_fee
    )
    return new_state


GlobalState.create_sub_pool = create_sub_pool
GlobalState.migrate_lp = migrate_lp
GlobalState.swap = swap
GlobalState.add_liquidity = add_liquidity
GlobalState.remove_liquidity = remove_liquidity
