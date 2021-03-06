from .amm.global_state import GlobalState, withdraw_all_liquidity, AMM
from .amm.agents import Agent
from .amm import global_state


# functions for calculating extra parameters we may want to track
def value_assets(prices: dict, agent: Agent) -> float:
    return sum([
        agent.holdings[i] * prices[i]
        for i in agent.holdings.keys()
    ])


def market_prices(state: GlobalState, shares: dict) -> dict:
    prices = {tkn: state.price(tkn) for tkn in state.asset_list}
    for share_id in shares:
        # if shares are for a specific asset in a specific pool, get prices according to that pool
        if isinstance(share_id, tuple):
            pool_id = share_id[0]
            tkn_id = share_id[1]
            prices[tkn_id] = state.pools[pool_id].price(tkn_id)

    return prices


def cash_out(state: GlobalState, agent: Agent) -> float:
    new_agent = withdraw_all_liquidity(state, agent.unique_id).agents[agent.unique_id]
    prices = market_prices(state, agent.shares)
    return value_assets(prices, new_agent)


def pool_val(state: GlobalState, pool: AMM):
    """ get the total value of all liquidity in the pool. """
    total = 0
    for asset in pool.asset_list:
        total += pool.liquidity[asset] * state.price(asset)
    return total


def postprocessing(events, optional_params=()):
    """
    Definition:
    Compute more abstract metrics from the simulation

    Optional parameters:
    'withdraw_val': tracks the actual value of each agent's assets if they were withdrawn from the pool at each step
    'deposit_val': tracks the theoretical value of each agent's original assets at each step's current spot prices,
        if they had been held outside the pool from the beginning
    'holdings_val': the total value of the agent's outside holdings
    'pool_val': tracks the value of all assets held in the pool
    'impermanent_loss': computes loss for LPs due to price movements in either direction
    """
    tokens = events[0]['state'].asset_list
    # save initial state
    initial_state: GlobalState = events[0]['state']
    initial_agents = initial_state.agents
    withdraw_state: GlobalState = initial_state.copy()

    optional_params = set(optional_params)
    if 'impermanent_loss' in optional_params:
        optional_params.add('deposit_val')
        optional_params.add('withdraw_val')

    agent_params = {
        'deposit_val',
        'withdraw_val',
        'holdings_val',
        'impermanent_loss'
    }
    exchange_params = {
        'pool_val',
        'usd_price'
    }
    unrecognized_params = optional_params.difference(agent_params | exchange_params)
    if unrecognized_params:
        raise ValueError(f'Unrecognized parameter {unrecognized_params}')

    print(f'processing {optional_params}')

    # a little pre-processing
    if 'deposit_val' in optional_params:
        # move the agents' liquidity deposits back into holdings, as something to compare against later
        for agent_id in initial_state.agents:
            # do it this convoluted way because we're pretending each agent withdrew their assets alone,
            # isolated from any effects of the other agents withdrawing *their* assets
            withdraw_state.agents[agent_id] = withdraw_all_liquidity(initial_state.copy(), agent_id).agents[agent_id]

    for step in events:
        state: GlobalState = step['state']

        for pool in state.pools.values():
            if 'pool_val' in optional_params:
                pool.pool_val = pool_val(state, pool)
            if 'usd_price' in optional_params:
                pool.usd_price = {tkn: pool.price(tkn) for tkn in pool.asset_list}

        # agents
        for agent in state.agents.values():
            if 'deposit_val' in optional_params:
                # what are this agent's original holdings theoretically worth at current spot prices?
                agent.deposit_val = value_assets(
                    market_prices(state, agent.shares),
                    withdraw_state.agents[agent.unique_id]
                )
            if 'withdraw_val' in optional_params:
                # what are this agent's holdings worth if sold?
                agent.withdraw_val = cash_out(state, agent)
            if 'holdings_val' in optional_params:
                agent.holdings_val = sum([quantity * state.price(asset) for asset, quantity in agent.holdings.items()])
            if 'impermanent_loss' in optional_params:
                agent.impermanent_loss = agent.withdraw_val / agent.deposit_val - 1

    return events
