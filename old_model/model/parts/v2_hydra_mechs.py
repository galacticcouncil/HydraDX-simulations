import numpy as np

def r_to_r_pool(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between 
    two risk assets where delta_Ri is the amount being sold.
    As per the mechanism of June 28, 2021, a weight update is informative only--thus this mechanism 
    returns the pool variable after delta_Ri and delta_Rk have been added to/removed from the pool's asset balances.
    """

    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
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
        return ('pool', pool)
    else:
        # The swap out value delta_Rk is always negative
        delta_Rk = ( (Ci/Ck)*Ri**(-a) - (Ci/Ck)*(Ri + delta_Ri)**(-a) + Rk**(-a) )**(-1/a) - Rk

        # Make the delta_Rk value positive for pool balance update below
        delta_Rk = - delta_Rk

        print(' R added to pool of ',asset_id,' = ', delta_Ri)
        print(' R removed from pool of ',purchased_asset_id,' = ', delta_Rk)

        pool.r_to_q_pool(asset_id, delta_Ri) # adds Ri to pool
        pool.q_to_r_pool(purchased_asset_id, delta_Rk) # SUBTRACTS removes Rk from pool

        return ('pool', pool)


def r_to_r_swap_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between two risk assets
    Under the mechanism defined June 28, 2021 there is **no change** in the quantity Q
    """
    Q = prev_state['Q']
    return('Q', Q)

def r_to_r_swap_H(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity H after a trade between two risk assets
    Under the mechanism defined June 28, 2021 there is **no change** in the quantity H
    """

    H = prev_state['H']
    return ('H', H)



def q_to_r_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold into the pool
    """
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    Q = prev_state['Q']
    
    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Q + delta_Q

    if delta_Q == 0 or threshold < 0:
        return ('Q', Q)
    else:
        return ('Q', Q + delta_Q)

def q_to_r_H(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns H after a trade where delta_Q is the amount being sold into the pool
    """
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    H = prev_state['H']
    Q = prev_state['Q']

    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Q + delta_Q

    if delta_Q == 0 or threshold < 0:
        return ('H', H)
    else:
        return ('H', H + delta_Q)


def r_to_q_pool(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a weight update follwing a trade 
    between a risk asset and the base asset where delta_R is the amount being sold.
    As per the mechanism of June 28, 2021, a weight update is informative only--thus this mechanism 
    returns the pool variable after delta_Ri is added to the pool's Ri balance.
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold']
    pool = prev_state['pool']

    if delta_Ri > 0:
        pool.r_to_q_pool(asset_id, delta_Ri) # adds delta_Ri to pool
    return ('pool', pool)

def r_to_q_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool base asset after a risk asset is traded for the base asset, 
    where delta_Ri is the risk asset amount being sold
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Ri = policy_input['ri_sold'] #amount of Ri being sold by the user

    pool = prev_state['pool']
    Q = prev_state['Q']
    Y = prev_state['Y']
    pool = prev_state['pool']
    
   

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)

    a = params['a']

    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Ri + delta_Ri

    if delta_Ri == 0 or threshold < 0:
        return ('Q', Q)
    else:
        delta_Q = Q * Y * (Y**(-a) - Ci * Ri**(-a) + Ci * (Ri + delta_Ri)**(-a))**(1/a) - Q
        return('Q', Q + delta_Q)

def r_to_q_H(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the total base asset after a risk asset is traded for the base asset, 
    where delta_Ri is the risk asset amount being sold
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Ri = policy_input['ri_sold'] #amount of Ri being sold by the user

    pool = prev_state['pool']
    Q = prev_state['Q']
    Y = prev_state['Y']
    pool = prev_state['pool']
    

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)
    H = prev_state['H']

    a = params['a']

    # JS July 9, 2021: compute threshold for reserve availability
    threshold = Ri + delta_Ri

    if delta_Ri == 0 or threshold < 0:
        return ('H', H)
    else:
        delta_Q = Q * Y * (Y**(-a) - Ci * Ri**(-a) + Ci * (Ri + delta_Ri)**(-a))**(1/a) - Q
        return ('H', H + delta_Q)

