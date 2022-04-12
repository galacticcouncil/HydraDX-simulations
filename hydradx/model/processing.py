import pandas as pd

from . import init_utils as iu
from .amm import amm


def postprocessing(events, count=True, count_tkn='R', count_k='n'):
    '''
    Definition:
    Refine and extract metrics from the simulation

    Parameters:
    df: simulation dataframe
    '''
    d = {}
    n = len(events[0]['AMM']['R'])
    agent_d = {'simulation': [], 'subset': [], 'run': [], 'substep': [], 'timestep': []}

    # build the DFs
    for step in events:
        for k in step:
            # expand AMM structure
            if k == 'AMM':
                for k in step['AMM']:
                    expand_state_var(k, step['AMM'][k], d)
                if count and count_tkn in step['AMM']:
                    d[count_k] = len(step['AMM'][count_tkn])

            elif k == 'external':
                for k in step['external']:
                    expand_state_var(k, step['external'][k], d)

            elif k == 'uni_agents':
                for agent_k in step['uni_agents']:
                    agent_state = step['uni_agents'][agent_k]

                    if 'agent_label' not in agent_d:
                        agent_d['agent_label'] = list()
                    agent_d['agent_label'].append(agent_k)

                    for k in agent_state:
                        expand_state_var(k, agent_state[k], agent_d)

                    # add simulation columns
                    for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
                        agent_d[key].append(step[key])

            else:
                expand_state_var(k, step[k], d)

    df = pd.DataFrame(d)
    agent_df = pd.DataFrame(agent_d)

    # subset to last substep
    df = df[df['substep'] == df.substep.max()]
    agent_df = agent_df[agent_df['substep'] == agent_df.substep.max()]

    #     # Clean substeps
    #     first_ind = (df.substep == 0) & (df.timestep == 0)
    #     last_ind = df.substep == max(df.substep)
    #     inds_to_drop = (first_ind | last_ind)
    #     df = df.loc[inds_to_drop].drop(columns=['substep'])

    #     # Attribute parameters to each row
    #     df = df.assign(**configs[0].sim_config['M'])
    #     for i, (_, n_df) in enumerate(df.groupby(['simulation', 'subset', 'run'])):
    #         df.loc[n_df.index] = n_df.assign(**configs[i].sim_config['M'])
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


def val_pool(row):
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
