import copy

import pandas as pd

# from amm.amm import WorldState
from .amm import omnipool_amm
from .amm.amm import Agent


def postprocessing(events, params_to_include: list[str] = ()):
    """
    Definition:
    Refine and extract metrics from the simulation

    optional parameters:
     * 'pool_holdings_value': adds a ['val_pool'] key to each agent in the dataframe, equal to the total value of all that agent's liquidity provisions if sold and then denominated in LRNA.
    """
    token_count = len(events[0]['WorldState'].exchange.pool_list)
    # n = len(events[0]['AMM']['R'])
    agent_d = {'simulation': [], 'subset': [], 'run': [], 'substep': [], 'timestep': [], 'q': [], 'agent_label': []}

    exchange_d = {
        "timestep": [],
        "L": [],
        'simulation': [],
        'substep': [],
        'subset': [],
        'run': []
    }
    exchange_d.update({f'R-{i}': [] for i in range(token_count)})
    exchange_d.update({f'Q-{i}': [] for i in range(token_count)})
    exchange_d.update({f'B-{i}': [] for i in range(token_count)})
    exchange_d.update({f'S-{i}': [] for i in range(token_count)})

    agent_d.update({f'p-{i}': [] for i in range(token_count)})
    agent_d.update({f'r-{i}': [] for i in range(token_count)})
    agent_d.update({f's-{i}': [] for i in range(token_count)})

    # build the DFs
    for (n, step) in enumerate(events):
        omnipool: omnipool_amm.OmniPool = step['WorldState'].exchange
        agents: dict[str, omnipool_amm.OmnipoolAgent] = step['WorldState'].agents
        for i in range(token_count):
            exchange_d[f'R-{i}'].append(omnipool.pool_list[i].assetQuantity)
            exchange_d[f'Q-{i}'].append(omnipool.pool_list[i].lrnaQuantity)
            exchange_d[f'B-{i}'].append(omnipool.pool_list[i].sharesOwnedByProtocol)
            exchange_d[f'S-{i}'].append(omnipool.pool_list[i].shares)
        exchange_d['L'].append(omnipool.L)
        for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
            exchange_d[key].append(step[key])

        for (a, agent_name) in enumerate(agents):
            for i in range(token_count):
                agent_d[f'p-{i}'].append(
                    agents[agent_name].position(omnipool.pool_list[i].shareToken).price
                    if agents[agent_name].position(omnipool.pool_list[i].shareToken) else 0
                )
                agent_d[f's-{i}'].append(agents[agent_name].holdings(omnipool.pool_list[i].shareToken) or 0)
                agent_d[f'r-{i}'].append(agents[agent_name].holdings(omnipool.asset(i).name) or 0)
            agent_d['agent_label'].append(agent_name)
            agent_d['q'].append(agents[agent_name].q)
            for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
                agent_d[key].append(step[key])

            # optional parameters
            if 'pool_holdings_value' in params_to_include:
                market_copy = copy.deepcopy(omnipool)
                agent_d['val_pool'] = (
                    agents[agent_name]
                    .erase_external_holdings()
                    .remove_all_liquidity(market_copy)
                    .value_holdings(omnipool)
                )

    df = pd.DataFrame(exchange_d)
    agent_df = pd.DataFrame(agent_d)

    # subset to last substep
    df = df[df['substep'] == df.substep.max()]
    agent_df = agent_df[agent_df['substep'] == agent_df.substep.max()]

    return df, agent_df


def expand_state_var(k, var, d) -> None:
    if isinstance(var, list):
        for i in range(len(var)):
            expand_state_var(k + "-" + str(i), var[i], d)
    else:
        if k not in d:
            d[k] = []
        d[k].append(var)


def get_state_from_row(row) -> dict:
    state = {
        'token_list': [None] * row['n'],
        'Q': [0] * row['n'],
        'R': [0] * row['n'],
        'A': [0] * row['n'],
        'S': [0] * row['n'],
        'B': [0] * row['n'],
        'L': row['L']
    }

    if 'H' in row:
        state['H'] = row['H']
    if 'T' in row:
        state['T'] = row['T']

    for i in range(row['n']):
        state['R'][i] = row['R-' + str(i)]
        state['S'][i] = row['S-' + str(i)]
        state['B'][i] = row['B-' + str(i)]
        state['Q'][i] = row['Q-' + str(i)]
        state['A'][i] = row['A-' + str(i)]
        state['token_list'][i] = row['token_list-' + str(i)]

    return state


def get_agent_from_row(row) -> dict:
    agent_d = {
        'r': [0] * row['n'],
        's': [0] * row['n'],
        'p': [0] * row['n'],
        'q': row['q']
    }

    for i in range(row['n']):
        agent_d['r'][i] = row['r-' + str(i)]
        agent_d['s'][i] = row['s-' + str(i)]
        agent_d['p'][i] = row['p-' + str(i)]

    return agent_d


def val_pool(agent: Agent, market: Market):
    state = get_state_from_row(row)
    agent_d = get_agent_from_row(row)
    return amm.value_holdings(state, agent_d, row['agent_label'])


def val_hold(row, orig_agent_d):
    state = get_state_from_row(row)
    agent = orig_agent_d[row['agent_label']]
    value = amm.value_assets(state, agent)
    return value


def get_withdraw_agent_d(initial_values: dict, agent_d: dict) -> dict:
    # Calculate withdrawal based on initial state
    withdraw_agent_d = {}
    initial_state = iu.complete_initial_values(initial_values, agent_d)
    agents_init_d = amm.convert_agents(initial_state, initial_values['token_list'], agent_d)
    for agent_id in agents_init_d:
        new_state, new_agents = amm.withdraw_all_liquidity(initial_state, agents_init_d[agent_id], agent_id)
        withdraw_agent_d[agent_id] = new_agents[agent_id]
    return withdraw_agent_d


def pool_val(row):
    state = get_state_from_row(row)
    value = sum(state['Q'])
    for i in range(len(state['R'])):
        value += state['R'][i] * state['B'][i] / state['S'][i] * amm.price_i(state, i)
    return value
