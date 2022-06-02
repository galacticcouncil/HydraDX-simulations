import pandas as pd

from .amm import amm
from .amm import omnipool_amm as oamm


def postprocessing(events, count=True, count_tkn='R', count_k='n', optional_params=()):
    """
    Definition:
    Refine and extract metrics from the simulation

    Parameters:
    df: simulation dataframe
    """
    tokens = events[0]['AMM'].asset_list
    keys = 'PQRST'
    state_d = {key: [] for key in [item for sublist in [
        [f'{key}-{token}' for key in keys]
        for token in tokens
    ] for item in sublist]}
    # result: {'P-USD': [], 'Q-USD': [], 'R-USD': [], 'S-USD': [], 'P-HDX': [], 'Q-HDX': []...}
    state_d.update({'L': []})

    agent_d = {'simulation': [], 'subset': [], 'run': [], 'substep': [], 'timestep': [], 'agent_id': []}

    optional_params = set(optional_params)
    agent_params = {
        'hold_val',
        'withdraw_val',
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

    # build the DFs
    for step in events:
        step_state: oamm.OmnipoolState = step['AMM']
        for token in step_state.asset_list:
            state_d[f'P-{token}'].append(step_state.liquidity[token] / step_state.lrna[token])
            state_d[f'Q-{token}'].append(step_state.lrna[token])
            state_d[f'R-{token}'].append(step_state.liquidity[token])
            state_d[f'S-{token}'].append(step_state.shares[token])
            state_d[f'T-{token}'].append(step_state.tvl[token])
        state_d['L'].append(step_state.lrna_imbalance)
        if count:
            state_d[count_k] = len(step_state.asset_list)
        if 'pool_val' in state_d:
            state_d['pool_val'].append(pool_val(step_state))

        for k in step:
            # expand AMM structure
            if k == 'AMM':
                pass
            elif k == 'external':
                for k in step['external']:
                    expand_state_var(k, step['external'][k], state_d)

            elif k == 'uni_agents':
                for agent_id in step['uni_agents']:
                    agent_state = step['uni_agents'][agent_id]
                    agent_d['agent_id'].append(agent_id)

                    for k in agent_state:
                        expand_state_var(k, agent_state[k], agent_d)

                    # add simulation columns
                    for key in ['simulation', 'subset', 'run', 'substep', 'timestep']:
                        agent_d[key].append(step[key])

                    if 'hold_val' in agent_d:
                        agent_d['hold_val'].append(amm.value_assets(step_state, agent_state))
                    if 'withdraw_val' in agent_d:
                        agent_d['withdraw_val'].append(amm.cash_out(step_state, agent_state))

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
        state.liquidity[i] * state.protocol_shares[i] / state.shares[i] * oamm.price_i(state, i)
        for i in state.asset_list
    ])
