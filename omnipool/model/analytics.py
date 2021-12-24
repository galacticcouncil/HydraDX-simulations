import pandas as pd


def add_var_to_dict(state, df_row, var) -> None:
    if var in df_row:
        state[var] = df_row[var]
    else:
        ls = []
        i = 0
        while var + '-' + str(i) in df_row:
            ls.append(df_row[var + '-' + str(i)])
            i += 1
        if ls:
            state[var] = ls


def get_state_from_df_row(df_row) -> dict:
    state = {}
    state_vars = ['Q', 'R', 'S', 'W', 'C', 'Y', 'P']
    for k in state_vars:
        add_var_to_dict(state, df_row, k)
    return state


def get_agent_from_df_row(df_row) -> dict:
    agent = {}
    agent_vars = ['h', 'r', 's', 'p', 'b']
    for k in agent_vars:
        add_var_to_dict(agent, df_row, k)
    return agent


# need to have merged CFMM state into agent DF
def calc_lp_val_pool(row, params_list):
    # print(row)
    state = get_state_from_df_row(row)
    agent = get_agent_from_df_row(row)
    transaction = {'s_burn': agent['s']}
    cfmm = params_list[row['simulation']]['cfmm'][0]
    new_agent = cfmm.remove_liquidity_agent(state, transaction, agent)
    val = new_agent['h']
    for i in range(len(agent['r'])):
        val += state['P'][i] * new_agent['r'][i]
    return val


def get_lp_val_df(rdf, agent_df, params_list):
    merged_df = pd.merge(agent_df, rdf, how='inner', on=['simulation', 'run', 'timestep', 'subset', 'substep'])
    # print(merged_df[merged_df['simulation'] == 5][['Q', 'R0', 'S0', 'W0', 'C0', 'Y', 'P0',
    #      'R1', 'S1', 'W1', 'C1', 'P1', 'h', 'r0', 's0', 'p0', 'r1', 's1', 'p1']])
    merged_df["value"] = merged_df.apply(lambda x: calc_lp_val_pool(x, params_list), axis=1)
    return merged_df[['simulation', 'run', 'timestep', 'subset', 'substep', 'value', 'agent_label']]


def calc_impermanent_loss(rdf, agent_df):  # TODO
    # agent_df should be trimmed down to a single simulation & agent
    pass
