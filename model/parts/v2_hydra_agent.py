import numpy as np
import pandas as pd
import copy 


def H_agent_add_liq(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent local states when liquidity is added in one asset.
    Amended 9 July 2021 to V2 Spec

    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']


    delta_R = policy_input['ri_deposit']
    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id) 
    delta_S = S * ( delta_R / R )

    agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_R
    agents.at[agent_id, 'r_' + asset_id + '_in'] = H_chosen_agent['r_' + asset_id + '_in'].values + delta_R
    agents.at[agent_id, 's_' + asset_id] = H_chosen_agent['s_' + asset_id].values + delta_S
        

    return ('hydra_agents', agents)  


def H_agent_remove_liq(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent local states when liquidity is removed in one asset.
    Amended 9 July 2021 to V2 Spec
    
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_S = policy_input['UNI_burn']
    
    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id) 

    delta_R = R * ( delta_S / S)
 
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

def H_agent_q_to_r(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when the pool asset Q is traded into the pool in return for a risk asset Ri
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
    Y = prev_state['Y']

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)

    a = params['a']
    if delta_Q == 0 or delta_Q <0:
        return ('hydra_agents', agents)
    else:

        delta_Ri = ( (1/Ci) * ((Q*Y) / (Q + delta_Q))**(-a) - (Y**(-a) / Ci) + Ri**(-a) )**(1/a) - Ri

        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values + delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values - delta_Q

        return ('hydra_agents', agents)


def H_agent_r_to_r_swap(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when one risk asset is traded for another risk asset
    Deepcopy fixes double resolution error
    """
    print(' R to R swap called ')
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]


    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Ck = pool.get_coefficient(purchased_asset_id)

    a = params['a']
 
    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Ri**(-a) + (Ck/Ci)*Rk**(-a) - (Ri + delta_Ri)**(-a)

    if delta_Ri == 0 or threshold < 0:
        return ('hydra_agents', agents)

    else:
        # The swap out value delta_Rk is always negative
        delta_Rk = ( (Ci/Ck)*Ri**(-a) - (Ci/Ck)*(Ri + delta_Ri)**(-a) + Rk**(-a) )**(-1/a) - Rk

        # Make the delta_Rk value positive for readability in agent balance update below
        delta_Rk = - delta_Rk

        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'r_' + purchased_asset_id + '_out'] = H_chosen_agent['r_' + purchased_asset_id + '_out'].values + delta_Rk
        
        return ('hydra_agents', agents)

def H_agent_r_to_q(params, substep, state_history, prev_state, policy_input):
    """
    This function updates Hydra agent states when a risk asset Ri is traded into the pool in return for the pool asset Q  
    """
    agent_id = policy_input['agent_id']
    agents =  copy.deepcopy(prev_state['hydra_agents'])
    H_chosen_agent = agents[agents['m']==agent_id]
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    Q = prev_state['Q']
    Y = prev_state['Y']

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)

    a = params['a']

    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Ri + delta_Ri

    if delta_Ri == 0 or threshold < 0:
        return ('hydra_agents', agents)
    else:
        delta_Q = Q * Y * (Y**(-a) - Ci * Ri**(-a) + Ci * (Ri + delta_Ri)**(-a))**(1/a) - Q
    
        agents.at[agent_id, 'r_' + asset_id + '_out'] = H_chosen_agent['r_' + asset_id + '_out'].values - delta_Ri
        agents.at[agent_id, 'h'] = H_chosen_agent['h'].values + delta_Q
        
        return ('hydra_agents', agents)