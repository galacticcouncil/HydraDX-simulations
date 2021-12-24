import copy

import ipdb

from .amm import amm

# Behaviors
def actionDecoder(params, step, history, prev_state) -> dict:
    t = len(history) - 1
    # print(t)
    action_list = params['action_list']
    action_dict = params['action_dict']

    if isinstance(action_list[t], tuple) or isinstance(action_list[t], list):
        policy_input = []
        for action in action_list[t]:
            if action in action_dict:
                policy_input.append(action_dict[action])
    elif action_list[t] not in action_dict:
        return {}
    else:
        policy_input = [action_dict[action_list[t]]]
    # We translate path dependent actions like market arb here
    '''
    if policy_input['action_id'] == 'ArbMarket':
        prev_market = prev_state['external']['market']
        amm = params['cfmm']
        trades = amm.trade_to_prices(prev_market, prev_state['AMM'])
        policy_input['arb_trades'] = trades
        print(trades)
    '''

    return {'policy_input': policy_input}


'''
def settle_to_asset(agent, prices_market, i_settle):
    for i in range(len(prices_market)):
        if i != i_settle:
            # convert profits/losses to i_settle
            if i_settle >= 0:
                agent['r'][i_settle] += agent['r'][i]*prices_market[i]/prices_market[i_settle]
            else:
                agent['h'] += agent['r'][i]*prices_market[i]
            agent['r'][i] = 0
    return agent
'''


def externalHub(params, substep, state_history, prev_state, policy_input):
    '''
    if 'market' in prev_state['external']:
        prev_market = prev_state['external']['market']
        sigma = params['sigma']
        mu = params['mu']
        n = len(prev_market)
        if params['fix-first']:
            W = list(np.random.randn(n - 1))
            #market = [1] + [prev_market[i+1] * math.exp(-sigma[i]**2/2 + sigma[i]*W[i]) for i in range(n - 1)]
            market = [1] + [prev_market[i + 1] * math.exp(mu[i] + sigma[i] * W[i]) for i in range(n - 1)]
        else:
            W = list(np.random.randn(n))
            #market = [prev_market[i] * math.exp(-sigma[i]**2/2 + sigma[i]*W[i]) for i in range(n)]
            market = [prev_market[i] * math.exp(mu[i] + sigma[i] * W[i]) for i in range(n)]
        state = copy.deepcopy(prev_state['external'])
        state['market'] = market
        return ('external', state)
    '''
    return ('external', prev_state['external'])


# Mechanisms
def mechanismHub_AMM(params, substep, state_history, prev_state, policy_input):
    policy_inputs = policy_input['policy_input']
    new_state = copy.deepcopy(prev_state['AMM'])
    new_agents = copy.deepcopy(prev_state['uni_agents'])
    if 'batch' in params and params['batch']:
        new_state, new_agents = amm.process_transactions(new_state, new_agents, policy_inputs)
    else:
        for action in policy_inputs:
            if 'action_id' in action:
                # amm = amm_selector.get_amm(params['cfmm_type'])
                action_id = action['action_id']
                if action_id == 'Trade':
                    new_state, new_agents = amm.swap(new_state, new_agents, action)
        '''
        elif action == 'ArbMarket':
            next_state = prev_state['AMM']
            for trade in policy_input['arb_trades']:
                next_state = params['cfmm'].trade(next_state, trade)
            return ('AMM', next_state)
        elif action == 'AddLiquidity':
            return ('AMM', params['cfmm'].add_liquidity(prev_state['AMM'], policy_input))
        elif action == 'RemoveLiquidity':
            return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], policy_input))
        elif action == 'RemoveLiquidityPercent':
            agent_id = policy_input['agent_id']
            agents = prev_state['uni_agents']
            agent = agents[agent_id]
            new_policy_input = copy.deepcopy(policy_input)
            new_policy_input['s_burn'] = [policy_input['r_percent'][i] * agent['s'][i] for i in range(len(policy_input['r_percent']))]
            return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], new_policy_input))
        '''
    return ('AMM', new_state)


def agenthub(params, substep, state_history, prev_state, policy_input):
    policy_inputs = policy_input['policy_input']
    new_state = copy.deepcopy(prev_state['AMM'])
    new_agents = copy.deepcopy(prev_state['uni_agents'])
    for action in policy_inputs:
        if 'action_id' in action and 'agent_id' in action:
            # amm = amm_selector.get_amm(params['cfmm_type'])
            action_id = action['action_id']
            agent_id = action['agent_id']
            if action_id == 'Trade':
                # agents[agent_id] = params['cfmm'].trade_agent(prev_state['AMM'], policy_input, agents[agent_id])
                new_state, new_agents = amm.swap(new_state, new_agents, action)
                return ('uni_agents', new_agents)
        '''
        elif action == 'ArbMarket':     # Note that we track changes of sequential changes to the AMM here too
            next_state = prev_state['AMM']
            prev_market = prev_state['external']['market']
            for trade in policy_input['arb_trades']:
                trade['agent_id'] = agent_id
                agents = params['cfmm'].trade_agents(next_state, trade, agents)
                next_state = params['cfmm'].trade(next_state, trade)
                agents[agent_id] = settle_to_asset(agents[agent_id], prev_market, policy_input['settle_asset'])
    
        elif action == 'AddLiquidity':
            agents[agent_id] = params['cfmm'].add_liquidity_agent(prev_state['AMM'], policy_input, agents[agent_id])
        elif action == 'RemoveLiquidity':
            agents[agent_id] = params['cfmm'].remove_liquidity_agent(prev_state['AMM'], policy_input, agents[agent_id])
        elif action == 'RemoveLiquidityPercent':
            agent = agents[agent_id]
            new_policy_input = copy.deepcopy(policy_input)
            new_policy_input['s_burn'] = [policy_input['r_percent'][i] * agent['s'][i] for i in range(len(policy_input['r_percent']))]
            return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], new_policy_input))
        '''
    return ('uni_agents', new_agents)


def unified_hub(params, substep, state_history, prev_state, policy_input):
    policy_inputs = policy_input['policy_input']

    new_state = copy.deepcopy(prev_state['global_state']['AMM'])
    new_agents = copy.deepcopy(prev_state['global_state']['uni_agents'])
    if 'batch' in params and params['batch']:
        new_state, new_agents = amm.process_transactions(new_state, new_agents, policy_inputs)
    else:
        for action in policy_inputs:
            if 'action_id' in action:
                # amm = amm_selector.get_amm(params['cfmm_type'])
                action_id = action['action_id']
                if action_id == 'Trade':
                    new_state, new_agents = amm.swap(new_state, new_agents, action)
    return ('global_state', {'AMM': new_state, 'uni_agents': new_agents})

'''
def posthub(params, substep, state_history, prev_state, policy_input):
    if 'T' in prev_state['AMM'] and prev_state['AMM']['T'] is not None:
        return ('AMM', amm.adjust_supply(prev_state['AMM']))
    return ('AMM', prev_state['AMM'])
'''