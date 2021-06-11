import numpy as np
import pandas as pd
import copy


def H_agent_add_liq(params, substep, state_history, prev_state, policy_input):
    """
This function updates system and Hydra agent local states when liquidity is added in one asset.
If symmetric liquidity add is enabled additional calculations are made.

    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']

    delta_R = policy_input['ri_deposit']
    # R = pool.get_reserve(asset_id)
    # S = pool.get_share(asset_id) 
    P = pool.get_price(asset_id) 
    # if policy_input['ri_deposit'] == 0:
    #     token_amount = 0
    # else:
    #     token_amount = int(policy_input['ri_deposit'])
    BTR = Sq / Q
    delta_Q = delta_R * P
    delta_Sq = delta_Q * BTR
    
    # JS May 19: weight adjustment for price invariance, a neq 1
    a = params['a']
    delta_W = delta_Q * (Wq / Q)**a
    #delta_W = delta_Q * Wq / Q

    delta_S = delta_W * Sq / Wq

    agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_R
    agents.at[agent_id, 'r_' + asset_id + '_in'] = H_chosen_agent['r_' + asset_id + '_in'].values + delta_R
    agents.at[agent_id, 's_' + asset_id] = H_chosen_agent['s_' + asset_id].values + delta_S
        

    return ('hydra_agents', agents)  


def H_agent_remove_liq(params, substep, state_history, prev_state, policy_input):
    """
This function updates system and Hydra agent states when liquidity is removed in one asset.
If symmetric liquidity add is enabled additional calculations are made.
    
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_S = policy_input['UNI_burn']
    # Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']
    Wi = pool.get_weight(asset_id)

    
    R = pool.get_reserve(asset_id)
    # S = pool.get_share(asset_id) 
    # P = pool.get_price(asset_id) 
    # if policy_input['ri_deposit'] == 0:
    #     token_amount = 0
    # else:
    #     token_amount = int(policy_input['ri_deposit'])
    
    # Before weight to share conversion
    # delta_R = delta_S / S * R
    
    # JS May 19: reserve adjustment for price invariance, a neq 1
    Q = prev_state['Q']
    P = pool.get_price(asset_id)
    delta_R = (delta_S / Sq) * (Q / P)
    # delta_R = (Wq / Wi) * (delta_S / Sq) * R
 

    agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values + delta_R
    agents.at[agent_id, 'r_' + asset_id + '_in'] = H_chosen_agent['r_' + asset_id + '_in'].values - delta_R
    agents.at[agent_id, 's_' + asset_id] = H_chosen_agent['s_' + asset_id].values - delta_S
        
    return ('hydra_agents', agents)

def getInputPrice(input_amount, input_reserve, output_reserve, params):
    fee_numerator = params['fee_numerator']
    fee_denominator = params['fee_denominator']
    input_amount_with_fee = input_amount * fee_numerator
    numerator = input_amount_with_fee * output_reserve
    denominator = (input_reserve * fee_denominator) + input_amount_with_fee
    return int(numerator // denominator)

def H_agent_q_to_r_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when a 'q to r' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    print('H_chosen_agent', H_chosen_agent)
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    print('delta_Q', delta_Q)

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id) 
    Si = pool.get_share(asset_id) 
    Pi = pool.get_price(asset_id) 

    a = params['a']

    if delta_Q == 0:
        return ('hydra_agents', agents)
    else:

        delta_Ri = Ri * ((Q / (Q + delta_Q))**(Wq / Wi) - 1) # blue box force negative because sebtracted from pool
        if params['a'] != 1:
            first = 1 /(1 - a)
            second = (Wq**a) / Wi
            third = Q**(1 - a) - (Q + delta_Q)**(1 - a)
            delta_Ri =  Ri * (np.exp(first*second*third) - 1)  # making negative because subtracted

        
        # print('delta_Ri = ',delta_Ri)
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values + delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values - delta_Q
        # print('H_agent r out = ', H_chosen_agent['r_' + asset_id + '_out'].values[0])
        return ('hydra_agents', agents)

def H_agent_q_to_r_trade_discrete(params, substep, state_history, prev_state, policy_input):
    """
This function updates Hydra agent states when a 'q to r' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id) 
    Si = pool.get_share(asset_id) 
    Pi = pool.get_price(asset_id) 
    

    if delta_Q == 0:
        return ('hydra_agents', agents)
    else:
        delta_Ri =- Ri * ((Q / (Q + delta_Q))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool

             
        # H_chosen_agent['h'] = H_chosen_agent['h'] - delta_Q
        # H_chosen_agent['r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'] + delta_Ri

        # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])
        # agents.update(H_chosen_agent_df)
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values + delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values - delta_Q
        
        return ('hydra_agents', agents)

def H_agent_q_to_r_trade(params, substep, state_history, prev_state, policy_input):
    """
This function updates Hydra agent states when a 'q to r' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    R = pool.get_reserve(asset_id)
    W = pool.get_weight(asset_id) 
    S = pool.get_share(asset_id) 
    P = pool.get_price(asset_id) 
    Wq = prev_state['Wq']

    if delta_Q == 0:
        return ('hydra_agents', agents)
    else:
        W_ratio = Wq / W
        inner_term = Q /(Q+ delta_Q*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_R = R * (1 - power)
             
        # H_chosen_agent['h'] = H_chosen_agent['h'] - delta_Q
        # H_chosen_agent['r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'] + delta_R
        # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])
        # agents.update(H_chosen_agent_df)

        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values + delta_R
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values - delta_Q
        
        return ('hydra_agents', agents)

def H_agent_r_to_q_trade_discrete(params, substep, state_history, prev_state, policy_input):
    """
This function updates Hydra agent states when a 'r to q' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']
    
    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    if delta_Ri == 0:
        return ('hydra_agents', agents)

    else:
        # Make negative from this perspective agent gains delta q
        delta_Q =- Q * ((Ri / (Ri + delta_Ri))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool
        # print('delta_Q', delta_Q)
        # H_chosen_agent['h'] = H_chosen_agent['h'] + delta_Q
        # H_chosen_agent['r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'] - delta_Ri
        # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])
        # agents.update(H_chosen_agent_df)

        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values + delta_Q
        return ('hydra_agents', agents) 

def H_agent_r_to_q_trade(params, substep, state_history, prev_state, policy_input):
    """
This function updates Hydra agent states when a 'r to q' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]

    delta_R = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    Q = prev_state['Q']
    # Sq = prev_state['Sq']
    R = pool.get_reserve(asset_id)
    W = pool.get_weight(asset_id) 
    # S = pool.get_share(asset_id) 
    # P = pool.get_price(asset_id) 
    Wq = prev_state['Wq']
    if delta_R == 0:
        return ('hydra_agents', agents)

    else:
        W_ratio = W / Wq
        inner_term = R /(R+ delta_R*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Q = Q * (1 - power)
                       
        # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])
        # agents.update(H_chosen_agent_df)
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_R
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values + delta_Q
 
        return ('hydra_agents', agents)

def H_agent_r_to_r_swap(params, substep, state_history, prev_state, policy_input):
    """
This function updates Hydra agent states when a 'r to r' swap is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
 
    if delta_Ri == 0:
        return ('hydra_agents', agents)

    else:
        W_ratio = Wi / Wk
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Rk = Rk * (1 - power)
                       
        # H_chosen_agent['r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'] - delta_Ri
        # H_chosen_agent['r_' + purchased_asset_id + '_out'] = H_chosen_agent['r_' + purchased_asset_id + '_out'] + delta_Rk

        # agents.update(H_chosen_agent_df)
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'r_' + purchased_asset_id + '_out'] = H_chosen_agent['r_' + purchased_asset_id + '_out'].values + delta_Rk
        
        return ('hydra_agents', agents)        



def H_agent_r_to_r_swap_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when a 'r to r' swap is performed
    Deepcopy fixes double resolution error
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    # agents =  prev_state['hydra_agents']
    H_chosen_agent = agents[agents['m']==agent_id]
    # H_chosen_agent = copy.deepcopy(agents[agents['m']==agent_id])
    # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 
 
    if delta_Ri == 0:
        return ('hydra_agents', agents)

    else:
        delta_Si = - (delta_Ri / Ri) * Si**2

        Si_ratio = delta_Si / Si
        delta_Sk = Sk * ((1/(1+Si_ratio))-1)
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool

        # new_value =                
        # H_chosen_agent_df['r_' + asset_id + '_out'] = H_chosen_agent_df['r_' + asset_id + '_out'] - delta_Ri
        # H_chosen_agent_df['r_' + purchased_asset_id + '_out'] = H_chosen_agent_df['r_' + purchased_asset_id + '_out'] + delta_Rk
        # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])
        # agents.update(H_chosen_agent_df)
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'r_' + purchased_asset_id + '_out'] = H_chosen_agent['r_' + purchased_asset_id + '_out'].values + delta_Rk
        
        return ('hydra_agents', agents)


def H_agent_r_to_r_swap_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when a 'r to r' swap is performed
    Deepcopy fixes double resolution error
    """
    print(' R to R swap called ')
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    # agents =  prev_state['hydra_agents']
    H_chosen_agent = agents[agents['m']==agent_id]
    # H_chosen_agent = copy.deepcopy(agents[agents['m']==agent_id])
    # H_chosen_agent_df = pd.DataFrame(H_chosen_agent, index=[int(H_chosen_agent['m'])])

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 
 
    if delta_Ri == 0:
        return ('hydra_agents', agents)

    else:
        delta_Wi = - (delta_Ri / Ri) * Wi

        Wi_ratio = delta_Wi / Wi
        delta_Wk = - Wi_ratio * Wk

        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Wi / Wk) - 1) # blue box force negative because sebtracted from pool
 
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'r_' + purchased_asset_id + '_out'] = H_chosen_agent['r_' + purchased_asset_id + '_out'].values + delta_Rk
        
        return ('hydra_agents', agents)

def H_agent_r_to_q_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when a 'q to r' trade is performed
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id) 
    Si = pool.get_share(asset_id) 
    Pi = pool.get_price(asset_id) 

    a = params['a']

    if delta_Ri == 0:
        return ('hydra_agents', agents)
    else:
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Wi / Wq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Wi * (1 -a) / Wq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q

        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values + delta_Q
        
        return ('hydra_agents', agents)