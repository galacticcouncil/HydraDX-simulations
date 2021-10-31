import string

from . import actions
from .amm import amm


def complete_initial_values(values: dict, cfmm_type: string) -> dict:
    state = amm.initialize_state(values)
    state['token_list'] = values['token_list']
    return state


def get_configuration(config_d: dict) -> tuple:
    # amm = amm_selector.get_amm(config_d['cfmm_type'])

    initial_values = complete_initial_values(config_d['initial_values'], config_d['cfmm_type'])
    timesteps = sum([x[1] for x in config_d['action_ls']])
    action_list = actions.get_action_list(config_d['action_ls'], config_d['prob_dict'])
    params = {'cfmm_type': [config_d['cfmm_type']],
              'action_list': [action_list],
              'action_dict': [config_d['action_dict']],
              'timesteps': [timesteps]}
    converted_agent_d = amm.convert_agents(initial_values, config_d['agent_d'])
    state = {'external': {}, 'AMM': initial_values, 'uni_agents': converted_agent_d}
    config_dict = {
        'N': 1,  # number of monte carlo runs
        'T': range(timesteps),  # number of timesteps - 147439 is the length of uniswap_events
        'M': params,  # simulation parameters
    }
    return config_dict, state


'''


def init_uniswap_lp(d_init, n) -> dict:
    d = {
        'r': [0]*n,
        's': [[0]*n for i in range(n)],
        'p': [0] * n,
        'h': 0
    }
    for k in d_init:
        x = k.split('-')
        if x[0] != 's':
            raise Exception
        i = int(x[1])
        j = int(x[2])
        d['s'][i][j] = d_init[k]
        d['s'][j][i] = d_init[k]
    return d


def init_lp(d_init: dict, n: int, opt:int=0) -> dict:
    d = {
        'r': [0]*n,
        's': [0]*n,
        'h': 0
    }
    if opt == 0:
        d['b'] = [0] * n
    else:  # TODO remove
        d['p'] = [0] * n
    for k in d_init:
        x = k.split('-')
        if x[0] != 's':
            raise Exception
        i = int(x[1])
        d['s'][i] = d_init[k]
    return d


#Uniswap
def init_trader(d_init, n) -> dict:
    d = {
        's': [[0] * n for i in range(n)],
        'p': [0] * n
    }
    d['r'] = copy.deepcopy(d_init['r'])
    d['h'] = d_init['h']
    return d


def init_balancer_trader(d_init:dict, n:int, opt:int=0) -> dict:
    d = {
        's': [0] * n,
    }
    if opt == 0:
        d['b'] = [0] * n
    else:  # TODO remove
        d['p'] = [0] * n

    d['r'] = copy.deepcopy(d_init['r'])
    d['h'] = d_init['h']
    return d

'''
