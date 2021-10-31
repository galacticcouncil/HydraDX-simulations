import pandas as pd


def postprocessing(events):
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


'''
def get_state_from_row(row, cfmm_type) -> dict:
    state = {
        'token_list': [None] * row['n'],
        'Q': [0] * row['n']
        'R': [0] * row['n'],
        'S': [0] * row['n'],
        'B': [0] * row['n']
    }

    for i in range(row['n']):
        state['R'][i] = row['R-' + str(i)]
        state['S'][i] = row['S-' + str(i)]
        state['B'][i] = row['B-' + str(i)]
        state['Q'][i] = row['B-' + str(i)]
        state['token_list'][i] = row['token_list-' + str(i)]

    return state


def get_agent_from_row(row) -> dict:
    agent_d = {
        'r': [0] * row['n'],
        's': [0] * row['n'],
        'h': row['h'],
        'q': row['q']
    }

    for i in range(row['n']):
        agent_d['r'][i] = row['r-' + str(i)]
        agent_d['s'][i] = row['s-' + str(i)]

    return agent_d


def val_pool(row, cfmm_type):
    state = get_state_from_row(row, cfmm_type)
    agent_d = get_agent_from_row(row)
    return amm.value_holdings(state, agent_d, row['agent_label'], cfmm_type)


def val_hold(row, orig_agent_d, cfmm_type):
    state = get_state_from_row(row, cfmm_type)
    agent = orig_agent_d[row['agent_label']]
    value = amm.value_assets(state, agent, state['P'])
    return value


def get_withdraw_agent_d(initial_values: dict, agent_d: dict, cfmm_type) -> dict:
    # Calculate withdrawal based on initial state
    withdraw_agent_d = {}
    initial_state = complete_initial_values(initial_values, agent_d, cfmm_type)
    agents_init_d = amm.convert_agents(initial_state, agent_d)
    for agent_id in agents_init_d:
        new_state, new_agents = amm.withdraw_all_liquidity(initial_state, agents_init_d[agent_id], agent_id, cfmm_type)
        withdraw_agent_d[agent_id] = new_agents[agent_id]
    return withdraw_agent_d


def pool_val(row, cfmm_type):
    state = get_state_from_row(row, cfmm_type)
    value = state['Q']*state['D']/state['H']
    for i in range(state['n']):
        value += state['R'][i] * state['B'][i]/state['S'][i] * state['P'][i]
    return value
'''
