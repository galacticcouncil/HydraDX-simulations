from . import actions
from .amm import amm
from .amm import omnipool_amm as oamm
import ipdb


def get_configuration(config_d: dict) -> tuple:
    """
    convert from notebook setup language to internal data format
    """
    initial_state: oamm.OmnipoolState = config_d['initial_state']
    # add liquidity held by agents to the pool
    initial_agents = config_d['agent_d']
    converted_agents = {agent_id: {} for agent_id in initial_agents}
    for agent_id, assets in initial_agents.items():
        converted_agents[agent_id]['s'] = {token: 0 for token in initial_state.asset_list}
        converted_agents[agent_id]['r'] = {token: 0 for token in initial_state.asset_list}
        converted_agents[agent_id]['p'] = {token: 0 for token in initial_state.asset_list}
        converted_agents[agent_id]['q'] = 0
        for token in assets:
            if token[:4] == 'omni':
                i = token[4:]
                add_liquidity = assets[token]
                new_shares = ((initial_state.shares[i] / initial_state.liquidity[i]) * add_liquidity)
                initial_state.shares[i] += new_shares
                initial_state.lrna[i] += add_liquidity * oamm.price_i(initial_state, i)
                initial_state.liquidity[i] += add_liquidity
                initial_state.tvl[i] = initial_state.lrna[i] / initial_state.price(initial_state.stablecoin)
                converted_agents[agent_id]['s'][i] = new_shares
                converted_agents[agent_id]['p'][i] = initial_state.price(i)
            elif token == 'LRNA':
                converted_agents[agent_id]['q'] = assets[token]
            else:
                converted_agents[agent_id]['r'][token] = assets[token]
    # converted_agent_d = amm.convert_agents(initial_state, config_d['agent_d'])
    timesteps = sum([x[1] for x in config_d['action_ls']])
    action_list = actions.get_action_list(config_d['action_ls'], config_d['prob_dict'])
    params = {'action_list': [action_list],
              'action_dict': [config_d['action_dict']],
              'timesteps': [timesteps]}
    state = {'external': {}, 'AMM': initial_state, 'uni_agents': converted_agents}
    config_dict = {
        'N': 1,  # number of monte carlo runs
        'T': range(timesteps),  # number of timesteps - 147439 is the length of uniswap_events
        'M': params,  # simulation parameters
    }
    return config_dict, state
