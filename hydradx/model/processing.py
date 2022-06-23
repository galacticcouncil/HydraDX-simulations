import pandas as pd

from .amm import amm
from .amm import omnipool_amm as oamm
from .amm import basilisk_amm as bamm


def postprocessing(events, optional_params=()):
    """
    Definition:
    Refine and extract metrics from the simulation

    Optional parameters:
    'withdraw_val': tracks the actual value of each agent's assets if they were withdrawn from the pool at each step
    'deposit_val': tracks the theoretical value of each agent's original assets at each step's current spot prices,
        if they had been held outside the pool from the beginning
    'holdings_val': the total value of the agent's outside holdings
    'pool_val': tracks the value of all assets held in the pool
    """
    tokens = events[0]['state']['amm'].asset_list
    # save initial state
    initial_state = events[0]['state']['amm']
    initial_agents = events[0]['state']['agents']
    withdraw_agents: dict[dict] = {}
    if 'deposit_val' in optional_params:
        # move the agents' liquidity deposits back into
        for agent_id in initial_agents:
            _, withdraw_agents[agent_id] = amm.withdraw_all_liquidity(initial_state, initial_agents[agent_id])

    agent_d = {'simulation': [], 'subset': [], 'run': [], 'substep': [], 'timestep': [], 'agent_id': []}
    state_d = {}
    if isinstance(initial_state, oamm.OmnipoolState):
        keys = 'PQRST'
        state_d = {key: [] for key in [item for sublist in [
            [f'{key}-{token}' for key in keys]
            for token in tokens
        ] for item in sublist]}
        # result: {'P-USD': [], 'Q-USD': [], 'R-USD': [], 'S-USD': [], 'P-HDX': [], 'Q-HDX': []...}
        state_d.update({'L': []})
    elif isinstance(initial_state, bamm.ConstantProductPoolState):
        state_d = {f'R-{tkn}': [] for tkn in initial_state.asset_list}
        state_d.update({'S': []})

    optional_params = set(optional_params)
    agent_params = {
        'deposit_val',
        'withdraw_val',
        'holdings_val'
    }
    exchange_params = {
        'pool_val'
    }
    unrecognized_params = optional_params.difference(agent_params | exchange_params)
    if unrecognized_params:
        raise ValueError(f'Unrecognized parameter {unrecognized_params}')

    # add optional params to the dictionaries
    for key in optional_params & agent_params:
        agent_d[key] = []
    for key in optional_params & exchange_params:
        state_d[key] = []

    # the mysterious list-flattening comprehension
    agent_assets = list(set([x for agent in initial_agents.values() for x in agent['r'].keys()]))

    # build the DFs
    for step in events:
        state = step['state']
        market = state['amm']

        # amm market
        if isinstance(market, oamm.OmnipoolState):
            for token in market.asset_list:
                state_d[f'P-{token}'].append(float(market.lrna[token] / market.liquidity[token]))
                state_d[f'Q-{token}'].append(float(market.lrna[token]))
                state_d[f'R-{token}'].append(float(market.liquidity[token]))
                state_d[f'S-{token}'].append(float(market.shares[token]))
                state_d[f'T-{token}'].append(float(market.tvl[token]))
            state_d['L'].append(float(market.lrna_imbalance))
        elif isinstance(market, bamm.ConstantProductPoolState):
            for token in market.asset_list:
                state_d[f'R-{token}'].append(market.liquidity[token])
            state_d['S'].append(market.shares)

        if 'pool_val' in state_d:
            state_d['pool_val'].append(pool_val(market))

        # external market
        for key, value in state['external'].items():
            expand_state_var(key, value, state_d)

        # agents
        for agent_id in state['agents']:
            agent_state = state['agents'][agent_id]
            agent_d['agent_id'].append(agent_id)
            for tkn in agent_assets:
                for key in 'prs':
                    if tkn not in agent_state[key].keys():
                        agent_state[key][tkn] = 0

            for key, value in agent_state.items():
                expand_state_var(key, value, agent_d)

            # add simulation columns
            for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
                agent_d[key].append(step[key])

            if 'deposit_val' in agent_d:
                # what are this agent's original holdings theoretically worth at current spot prices?
                agent_d['deposit_val'].append(amm.value_assets(market, withdraw_agents[agent_id]))
            if 'withdraw_val' in agent_d:
                # what are this agent's holdings worth if sold?
                agent_d['withdraw_val'].append(amm.cash_out(market, agent_state))
            if 'holdings_val' in agent_d:
                # crude. Todo: update this to take asset price into account.
                agent_d['holdings_val'].append(sum(agent_state['r'].values()))

        # other cadCAD stuff
        for k in step:
            # expand AMM structure
            if k == 'state':
                pass
            else:
                expand_state_var(k, step[k], state_d)

    # print({key: len(agent_d[key]) for key in agent_d})
    # print({key: len(state_d[key]) if isinstance(state_d[key], list) else 0 for key in state_d})
    df = pd.DataFrame(state_d)
    agent_df = pd.DataFrame(agent_d)

    # subset to last substep
    df = df[df['substep'] == df.substep.max()]
    agent_df = agent_df[agent_df['substep'] == agent_df.substep.max()]

    return df, agent_df


def expand_state_var(k, var, d) -> None:
    if isinstance(var, dict):
        for i in var.keys():
            expand_state_var(f"{k}-{i}", var[i], d)
    else:
        if k not in d:
            d[k] = []
        d[k].append(var)


def pool_val(state: oamm.OmnipoolState):
    return state.lrna_total + sum([
        state.liquidity[i] * state.protocol_shares[i] / state.shares[i] * state.price(i)
        for i in state.asset_list
    ])
