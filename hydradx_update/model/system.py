from .amm import amm


# Behaviors
def actionDecoder(params, step, history, prev_state) -> dict:
    t = len(history) - 1
    # print(t)
    action_list = params['action_list']
    action_dict = params['action_dict']
    if action_list[t] not in action_dict:
        return {}
    policy_input = action_dict[action_list[t]]
    # We translate path dependent actions like market arb here
    '''
    if policy_input['action_id'] == 'ArbMarket':
        prev_market = prev_state['external']['market']
        amm = params['cfmm']
        trades = amm.trade_to_prices(prev_market, prev_state['AMM'])
        policy_input['arb_trades'] = trades
        print(trades)
    '''
    return policy_input


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
    if 'action_id' not in policy_input:
        return ('AMM', prev_state['AMM'])
    # amm = amm_selector.get_amm(params['cfmm_type'])
    action = policy_input['action_id']
    agents = prev_state['uni_agents']
    if action == 'Trade':
        new_state, _ = amm.swap(prev_state['AMM'], agents, policy_input)
        return ('AMM', new_state)
    elif action == 'AddLiquidity':
        print("Adding Liquidity")
        new_state, _ = amm.add_liquidity(prev_state['AMM'], agents, policy_input)
        return ('AMM', new_state)
        #return ('AMM', params['cfmm'].add_liquidity(prev_state['AMM'], policy_input)) #KP-Q: What is going on here?
    elif action == 'RemoveLiquidity':
        print("Removing Liquidity")
        new_state, _ = amm.remove_liquidity(prev_state['AMM'], agents, policy_input)
        return ('AMM', new_state)
    #    return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], policy_input)) #KP-Q: What is going on here?
    '''
    elif action == 'ArbMarket':
        next_state = prev_state['AMM']
        for trade in policy_input['arb_trades']:
            next_state = params['cfmm'].trade(next_state, trade)
        return ('AMM', next_state)
    
    elif action == 'RemoveLiquidityPercent':
        agent_id = policy_input['agent_id']
        agents = prev_state['uni_agents']
        agent = agents[agent_id]
        new_policy_input = copy.deepcopy(policy_input)
        new_policy_input['s_burn'] = [policy_input['r_percent'][i] * agent['s'][i] for i in range(len(policy_input['r_percent']))]
        return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], new_policy_input))
    '''
    return ('AMM', prev_state['AMM'])


def agenthub(params, substep, state_history, prev_state, policy_input):
    if 'action_id' not in policy_input or 'agent_id' not in policy_input:
        return ('uni_agents', prev_state['uni_agents'])
    # amm = amm_selector.get_amm(params['cfmm_type'])
    action = policy_input['action_id']
    agent_id = policy_input['agent_id']
    agents = prev_state['uni_agents']
    if action == 'Trade':
        # agents[agent_id] = params['cfmm'].trade_agent(prev_state['AMM'], policy_input, agents[agent_id])
        _, new_agents = amm.swap(prev_state['AMM'], agents, policy_input)
        return ('uni_agents', new_agents)
    elif action == 'AddLiquidity':
        print("Agent update for AddLiquidity")
        _, new_agents = amm.add_liquidity(prev_state['AMM'], agents, policy_input)
        #KP-Q: params key 'cfmm' not defined and function add_liquidity_agent where defined?
        #agents[agent_id] = params['cfmm'].add_liquidity_agent(prev_state['AMM'], policy_input, agents[agent_id])
        return ('uni_agents', new_agents)
    elif action == 'RemoveLiquidity':
        print("Agent update for RemoveLiquidity")
        _, new_agents = amm.remove_liquidity(prev_state['AMM'], agents, policy_input)
        #agents[agent_id] = params['cfmm'].remove_liquidity_agent(prev_state['AMM'], policy_input, agents[agent_id])
        #KP-Q: params key 'cfmm' not defined and function add_liquidity_agent where defined?
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


    elif action == 'RemoveLiquidityPercent':
        agent = agents[agent_id]
        new_policy_input = copy.deepcopy(policy_input)
        new_policy_input['s_burn'] = [policy_input['r_percent'][i] * agent['s'][i] for i in range(len(policy_input['r_percent']))]
        return ('AMM', params['cfmm'].remove_liquidity(prev_state['AMM'], new_policy_input))
    '''
    return ('uni_agents', agents)
