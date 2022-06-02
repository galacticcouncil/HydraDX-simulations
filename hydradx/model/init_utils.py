from . import actions
from .amm import amm
from .amm import omnipool_amm as oamm
import ipdb


def get_configuration(config_d: dict) -> tuple:

    initial_state: oamm.OmnipoolState = config_d['initial_state']
    # add liquidity held by agents to the pool
    for agent in config_d['agent_d'].values():
        for asset in agent:
            if asset[:4] == 'omni':
                poolName = asset[4:]
                add_liquidity = agent[asset]
                initial_state.shares[poolName] += ((initial_state.shares[poolName] / initial_state.liquidity[poolName])
                                                   * add_liquidity)
                initial_state.lrna[poolName] += add_liquidity * oamm.price_i(initial_state, poolName)
                initial_state.liquidity[poolName] += add_liquidity
    converted_agent_d = amm.convert_agents(initial_state, config_d['agent_d'])
    timesteps = sum([x[1] for x in config_d['action_ls']])
    action_list = actions.get_action_list(config_d['action_ls'], config_d['prob_dict'])
    params = {'action_list': [action_list],
              'action_dict': [config_d['action_dict']],
              'timesteps': [timesteps]}
    state = {'external': {}, 'AMM': initial_state, 'uni_agents': converted_agent_d}
    config_dict = {
        'N': 1,  # number of monte carlo runs
        'T': range(timesteps),  # number of timesteps - 147439 is the length of uniswap_events
        'M': params,  # simulation parameters
    }
    return config_dict, state
