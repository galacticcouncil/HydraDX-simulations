import numpy as np

def addLiquidity_Sq(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns shares Sq of the pool after a liquidity add.
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    Sq = prev_state['Sq']
    delta_R = policy_input['ri_deposit']
    pool = prev_state['pool']
    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id) 
   
    delta_S = S * (delta_R / R )
    
    return ('Sq', Sq + delta_S)

def addLiquidity_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns quantity Q after a deposit in a risk asset
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_R = policy_input['ri_deposit']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    P = pool.get_price(asset_id) 
   
    Ri_plus = Ri + delta_R
    Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('Q', Q_plus)


def addLiquidity_pool(params, substep, state_history, prev_state, policy_input):
    """
    Updates pool values after an add liquidity event
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']      
    delta_R = policy_input['ri_deposit']
    a = params['a']

    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id)
    C = pool.get_coefficient(asset_id)

    a = params['a']
    delta_S = S * ( delta_R / R )
    delta_C = C * ( ((R + delta_R) / R) ** (a+1) - 1 )

    pool.add_liquidity_pool(asset_id, delta_R, delta_S, delta_C)    
    return ('pool', pool)

def removeLiquidity_Sq(params, substep, state_history, prev_state, policy_input):
    """
    This function returns shares Sq after a liquidity removal in a specific risk asset.
    The delta_Sq is taken prom the policy_input as the amount 'UNI_burn'
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_S = policy_input['UNI_burn']
    Sq = prev_state['Sq']
    
    return ('Sq', Sq - delta_S)

def resolve_addLiquidity_H(params, substep, state_history, prev_state, policy_input):
    """
    This function returns the toal amount of H in the system after a deposit in a specific risk asset. Works the same as Q.
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_R = policy_input['ri_deposit']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    P = pool.get_price(asset_id) 
   
    Ri_plus = Ri + delta_R
    Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('H', Q_plus)


def resolve_remove_Liquidity_H(params, substep, state_history, prev_state, policy_input):
    """
    This function returns the toal amount of H in the system after a removal in a specific risk asset. Works the same as Q.
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_S = policy_input['UNI_burn']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    P = pool.get_price(asset_id) 
    Si = pool.get_share(asset_id) 
    
    delta_R = (delta_S / Si) * (Q / P)   
    Ri_plus = Ri - delta_R
    Ci_plus = Ci * ((Ri - delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri - delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('H', (Q_plus))

def removeLiquidity_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns the amount Q after a liquidity removal in a specific risk asset; spec 6-28-21
    as delta R is assumed to be positive, the signs are reversed
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_S = policy_input['UNI_burn']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    P = pool.get_price(asset_id) 
    Si = pool.get_share(asset_id) 

    delta_R = (delta_S / Si) * (Q / P)   
    Ri_plus = Ri - delta_R
    Ci_plus = Ci * ((Ri - delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri - delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('Q', Q_plus)

def removeLiquidity_pool(params, substep, state_history, prev_state, policy_input):
    """
    Updates pool values after a remove liquidity event
    Amended 9 July, 2021 to V2 Spec
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_S = policy_input['UNI_burn']
    
    pool = prev_state['pool']
   
    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id)
    C = pool.get_coefficient(asset_id)

    a = params['a']
    Sq = prev_state['Sq']

    delta_R = R * ( delta_S / S )
    delta_C = C * ( ((R - delta_R) / R) ** (a+1) - 1 )
    
    # JS July 9, 2021: Note the minus sign added to delta_C in function call below: 
    # delta_C < 0, but remove_liquidity_pool expects a positive number
    pool.remove_liquidity_pool(asset_id, delta_R, delta_S, -delta_C, a)

    return ('pool', pool)


def q_to_r_pool(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade where 
    delta_Q is the amount being sold.
    """

    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Q  = policy_input['q_sold'] #amount of Q being sold by the user

    pool = prev_state['pool']
    Q    = prev_state['Q']
    Y    = prev_state['Y']

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)

    a = params['a']

    if delta_Q == 0 or delta_Q < 0:
        return ('pool', pool)
    else:
        delta_Ri = ( (1/Ci) * ((Q*Y) / (Q + delta_Q))**(-a) - (Y**(-a) / Ci) + Ri**(-a) )**(1/a) - Ri
    
        pool.q_to_r_pool(asset_id, delta_Ri)

        return ('pool', pool)

def addLiquidity_Y(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns Y after a liquidity add; according to spec 6-28-21    
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_R = policy_input['ri_deposit']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    P = pool.get_price(asset_id) 
    Ri_plus = Ri + delta_R
    Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))
    

    return ('Y', Y_plus)

def removeLiquidity_Y(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns Y after a liquidity remove; according to spec 6-28-21    
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_S = policy_input['UNI_burn']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    Si = pool.get_share(asset_id)

    P = pool.get_price(asset_id) 
    delta_R = (delta_S / Si) * (Q / P)
    
    Ri_plus = Ri - delta_R
    print('Ri_plus = ', Ri_plus)
    Ci_plus = Ci * ((Ri - delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri - delta_R) ** (-a))) ** (- (1 / a))
    

    return ('Y', Y_plus)


