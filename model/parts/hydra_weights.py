import numpy as np

def q_to_r_Wq(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    delta_Q = int(policy_input['q_sold']) #amount of Q being sold by the user
    Q = prev_state['Q']
    Wq = prev_state['Wq']

    if delta_Q == 0:
        return ('Wq', Wq)
    else:
        delta_Wq = delta_Q / Q * Wq  # making negative
    
        return ('Wq', Wq + delta_Wq) 

def r_to_q_Wq(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']

    Q = prev_state['Q']
    Wq = prev_state['Wq']
    
    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    a = params['a']

    if delta_Ri == 0:
        return ('Wq', Wq)

    else:
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Wi / Wq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Wi * (1 -a) / Wq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q
        delta_Wq = delta_Q / Q * Wq
        
        return ('Wq', Wq + delta_Wq)

def addLiquidity_Wq(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns shares Wq of a risk asset after a liquidity add.
    Wq = Wq + delta_Wq
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    # asset = prev_state['asset']
    Q = prev_state['Q']
    Wq = prev_state['Wq']

    delta_R = policy_input['ri_deposit']
    pool = prev_state['pool']
    P = pool.get_price(asset_id) 

    BTR = Wq / Q
    delta_Q = delta_R * P
    delta_Wq = delta_Q * BTR
    # print('Wq ADD LIQUIDITY Timestep === ', prev_state['timestep'],' delta W = ', delta_Wq)

    return ('Wq', Wq + delta_Wq)

def removeLiquidity_Wq(params, substep, state_history, prev_state, policy_input):
    """
    This function returns shares Wq after a liquidity removal in a specific risk asset.
    Wq = Wq - delta_Wq
    The delta_Wq is taken prom the policy_input as the amount 'UNI_burn'
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_S = int(policy_input['UNI_burn'])
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']
    delta_Wq = Wq / Sq * delta_S
    
    return ('Wq', Wq - delta_Wq)

def r_to_r_swap_Wq(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = int(policy_input['ri_sold']) #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Q = prev_state['Q']
    Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('Wq', Wq)

    else:
        delta_Wi = - (delta_Ri / Ri) * Wi

        Wi_ratio = delta_Wi / Wi
        delta_Wk = - Wi_ratio * Wk

        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Wi / Wk) - 1) # blue box force negative because sebtracted from pool
 
        delta_Wq = delta_Wi + delta_Wk

        delta_Q = delta_Wq / Wq * Q

        return ('Wq', Wq + delta_Wq)