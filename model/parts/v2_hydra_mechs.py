import numpy as np

def r_to_r_pool_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('pool', pool)
    else:
        delta_Wi = - (delta_Ri / (Ri + delta_Ri)) * Wi

        Wi = Wi + delta_Wi
        Wi_ratio = delta_Wi / Wi
        delta_Wk = - Wi_ratio * Wk
        Wk = Wk + delta_Wk
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Wi / Wk) - 1) # blue box force negative because sebtracted from pool
        print(' R added to pool of ',asset_id,' = ', delta_Ri)
        print(' R removed from pool of ',purchased_asset_id,' = ', delta_Rk)

        pool.r_to_q_pool(asset_id, delta_Ri) # adds Ri to pool
        pool.q_to_r_pool(purchased_asset_id, delta_Rk) # SUBTRACTS removes Rk from pool

        pool.swap_weight_pool(asset_id, delta_Wi) # adds share
        pool.swap_weight_pool(purchased_asset_id, delta_Wk) # adds share

        return ('pool', pool)


def r_to_r_swap_Qh_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
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
        return ('Q', Q)

    else:
        delta_Wi = - (delta_Ri / (Ri + delta_Ri)) * Wi

        Wi = Wi + delta_Wi
        Wi_ratio = delta_Wi / Wi
        delta_Wk = - Wi_ratio * Wk
        Wk = Wk + delta_Wk
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Wi / Wk) - 1) # blue box force negative because sebtracted from pool
 
        delta_Wq = delta_Wi + delta_Wk

        delta_Q = delta_Wq / Wq * Q

        return ('Q', Q + delta_Q)

def r_to_r_swap_H_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Q = prev_state['Q']
    H = prev_state['H']

    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('H', H)

    else:
        delta_Wi = - (delta_Ri / (Ri + delta_Ri)) * Wi

        Wi_ratio = delta_Wi / Wi
        delta_Wk = - Wi_ratio * Wk

        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Wi / Wk) - 1) # blue box force negative because sebtracted from pool
 
        delta_Sq = delta_Wi + delta_Wk

        delta_Q = delta_Sq / Sq * Q

        return ('H', H + delta_Q)

def r_to_r_swap_Sq_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('Sq', Sq)

    else:
        # FROM REORDERING
        # delta_Wi = - (delta_Ri / (Ri + delta_Ri)) * Wi
        delta_Si = - (delta_Ri / (Ri + delta_Ri)) * Si

        Si_ratio = delta_Si / Si
        delta_Sk = - Si_ratio * Sk

        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool
 
        delta_Sq = delta_Si + delta_Sk

        delta_Q = delta_Sq / Sq * Q

        return ('Sq', Sq + delta_Sq)

def q_to_r_pool_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)
    # print(params['a'])
    a = params['a']
    # print(a)
    if delta_Q == 0:
        return ('pool', pool)
    else:
        delta_Ri = - Ri * ((Q / (Q + delta_Q))**(Wq / Wi) - 1)  # making negative because subtracted
        if params['a'] != 1:
            first = 1 /(1 - a)
            second = (Wq**a) / Wi
            third = Q**(1 - a) - (Q + delta_Q)**(1 - a)
            delta_Ri = - Ri * (np.exp(first*second*third) - 1)  # making negative because subtracted

        # print("**********")
        # print(f"POOL RESERVE ONE REMOVING ASSET {asset_id} in amount {delta_Ri} from reserve {Ri}")
        # print("**********")
        
        delta_Wi = (delta_Q / Q) * Wq # check sign

        pool.q_to_r_pool(asset_id, delta_Ri) # subtracts Ri from pool
        pool.swap_weight_pool(asset_id, delta_Wi) # adds share

        return ('pool', pool)

def q_to_r_Qh_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    Q = prev_state['Q']
    
    # print("**********")
    # print(f"Q ADDED INTO POOL: {delta_Q} to existing amount {Q}")
    # print("**********")

    if delta_Q == 0:
        return ('Q', Q)
    else:
        return ('Q', Q + delta_Q)

def q_to_r_H_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    H = prev_state['H']

    if delta_Q == 0:
        return ('H', H)
    else:
        return ('H', H + delta_Q)

def q_to_r_Sq_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    if delta_Q == 0:
        return ('Sq', Sq)
    else:
        delta_Sq = delta_Q / Q * Sq  # making negative
    
        return ('Sq', Sq + delta_Sq) 

def r_to_q_pool_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between a risk asset and the base asset where delta_R is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    a = params['a']

    if delta_Ri == 0:
        return ('pool', pool)

    else:
        pool.r_to_q_pool(asset_id, delta_Ri) # adds delta_Ri to pool

        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Wi / Wq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Wi * (1 -a) / Wq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q
        
        delta_Wq = delta_Q / Q * Wq  # making negative

        delta_Wi = delta_Wq
        pool.swap_weight_pool(asset_id, delta_Wi)

        return ('pool', pool)

def r_to_q_Qh_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between a risk asset and the base asset where delta_R is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    a = params['a']

    if delta_Ri == 0:
        return ('Q', Q)

    else:
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Wi / Wq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Wi * (1 -a) / Wq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q
      
        return ('Q', Q + delta_Q)

def r_to_q_H_reserve_one(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between a risk asset and the base asset where delta_R is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)
    H = prev_state['H']

    a = params['a']

    if delta_Ri == 0:
        return ('H', H)


    else:
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Wi / Wq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Wi * (1 -a) / Wq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q
      
        return ('H', H + delta_Q)

def r_to_q_Sq_reserve_one(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']

    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']

    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Wq = prev_state['Wq']
    
    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    a = params['a']

    if delta_Ri == 0:
        return ('Sq', Sq)

    else:
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Si / Sq) - 1) 
        if params['a'] != 1:
            first = Q**(1 - a)
            second = Si * (1 -a) / Sq**a
            third = np.log(1 + delta_Ri / Ri)
            delta_Q = (first - second * third)**(1/(1 - a)) - Q
        delta_Sq = delta_Q / Q * Sq
        
        return ('Sq', Sq + delta_Sq)

