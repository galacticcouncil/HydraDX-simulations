import numpy as np

def addLiquidity_Sq(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns shares Sq of the pool after a liquidity add.
    Sq = Sq + delta_Sq
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    # asset = prev_state['asset']
    # Q = prev_state['Q']
    Sq = prev_state['Sq']
    #Wq = prev_state['Wq']

    delta_R = policy_input['ri_deposit']
    pool = prev_state['pool']
    #     P = asset.loc[asset['id']==asset_id]['P']
    #P = pool.get_price(asset_id) 
    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id) 

    # if policy_input['ri_deposit'] == 0:
    #     token_amount = 0
    # else:
    #     token_amount = int(policy_input['ri_deposit'])
    # BTR = Sq / Q
    # delta_Q = delta_R * P
    # delta_Sq = delta_Q * BTR
    # print('addliq - Sq delta Sq', delta_Sq)

    # delta_W = delta_Q * Wq / Q
    # delta_S = delta_W * Sq / Wq
    # print('addliq - Sq delta S', delta_S)
    
    delta_S = S * (delta_R / R )
    
    return ('Sq', Sq + delta_S)

def addLiquidity_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns quantity Q after a deposit in a risk asset; spec 6-28-21
    
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_R = policy_input['ri_deposit']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    # Wq = prev_state['Wq']
    Sq = prev_state['Sq']
    # R = pool.get_reserve(asset_id)
    # W = pool.get_weight(asset_id) 
    # S = pool.get_share(asset_id) 
    P = pool.get_price(asset_id) 
   
    Ri_plus = Ri + delta_R
    Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('Q', Q_plus)


def calc_price_q_i(Ki, Ri, Q, Si, Sq):
    """
    calculates the price according to specification from 3-3-21
    """
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
    first_term = Ki / Ri
    second_term_fraction = (Q / Sq) / (Ri / Si)
    price_q_i = first_term - second_term_fraction * np.log(Ri)
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
    return price_q_i


def r_to_r_swap_Qh(params, substep, state_history, prev_state, policy_input):
    """
    calculates the quantity Q that results from a swap of two assets according to the specification from 3-3-21
    """
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Ki = params['Ki']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)

    if delta_Ri == 0:
        return ('Q', Q)
    else:
        W_ratio = Wi / Wk
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Rk = Rk * (1 - power)

        Si = pool.get_share(asset_id)
        Sk = pool.get_share(purchased_asset_id) 

        price_q_i = calc_price_q_i(Ki, Ri, Q, Si, Sq)
        delta_Q_i = price_q_i * delta_Ri

        price_q_k = calc_price_q_i(Ki, Rk, Q, Sk, Sq)
        delta_Q_k = price_q_k * delta_Rk


########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

        return ('Q', Q + delta_Q_i - delta_Q_k)

def addLiquidity_pool(params, substep, state_history, prev_state, policy_input):
    """
    Updates pool values after an add liquidity event
    Amended 9 July, 2021 to V2 Spec
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    
    # KP: can remove most code for spec 6-28-21
    #Q = prev_state['Q']
    #Sq = prev_state['Sq']
    #Wq = prev_state['Wq']
    # print('Timestep === ', prev_state['timestep'], prev_state['Wq'])
    
    delta_R = policy_input['ri_deposit']
    a = params['a']

    R = pool.get_reserve(asset_id)
    S = pool.get_share(asset_id)
    C = pool.get_coefficient(asset_id)

    a = params['a']
    #P = pool.get_price(asset_id)
    #BTR = Sq / Q
    #delta_Q = delta_R * P
    #delta_Sq = delta_Q * BTR
    #delta_W = delta_Q * Wq / Q
    # JS May 19: weight adjustment for price invariance, a neq 1
    #a = params['a']
    #delta_W = delta_Q * (Wq / Q)**a
    
    #P = pool.get_price(asset_id)
    #BTR = Sq / Q
    #delta_Q = delta_R * P
    #delta_Sq = delta_Q * BTR
    #delta_W = delta_Q * Wq / Q
    # JS May 19: weight adjustment for price invariance, a neq 1
    #delta_W = delta_Q * (Wq / Q)**a

    delta_S = S * ( delta_R / R )
    delta_C = C * ( ((R + delta_R) / R) ** (a+1) - 1 )

    # print('delta S', delta_Sq)
    # print('POOL ADD LIQUIDITY Timestep === ', prev_state['timestep'],'delta W ', delta_W)
    #a = params['a']
    
    pool.add_liquidity_pool(asset_id, delta_R, delta_S, delta_C)
    
    # pool.update_price_a(a,Q, Wq)

    return ('pool', pool)

def removeLiquidity_Sq(params, substep, state_history, prev_state, policy_input):
# def removeLiquidity_Sq(params, substep, state_history, prev_state, policy_input):
    """
    This function returns shares Sq after a liquidity removal in a specific risk asset.
    Sq = Sq - delta_Sq
    The delta_Sq is taken prom the policy_input as the amount 'UNI_burn'
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_S = policy_input['UNI_burn']
    Sq = prev_state['Sq']
    
    return ('Sq', Sq - delta_S)

def r_to_q_Sq_discrete(params, substep, state_history, prev_state, policy_input):
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

    if delta_Ri == 0:
        return ('Sq', Sq)

    else:
        # Make negative from this perspective agent gains delta q
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool
        # print('delta_Q', delta_Q)
        delta_Sq = delta_Q / Q * Sq
        # print('delta_Sq', delta_Sq)
        
        return ('Sq', Sq + delta_Sq)

def r_to_q_Sq(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)

    if delta_Ri == 0:
        return ('Sq', Sq)

    else:

###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si

########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

        return ('Sq', Sq + delta_Si)

def r_to_r_swap_Sq(params, substep, state_history, prev_state, policy_input):
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################    
    asset_id = policy_input['asset_id'] # defines asset subscript
    Sq = prev_state['Sq']
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)

    if delta_Ri == 0:
        return ('Sq', Sq)
    else:
        W_ratio = Wi / Wk
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Rk = Rk * (1 - power)

        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si

        Sk = pool.get_share(purchased_asset_id) 
        delta_Sk = - delta_Rk / Rk * Sk
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

    return ('Sq', Sq + delta_Si + delta_Sk) # delta_Sk already negative

def resolve_addLiquidity_H(params, substep, state_history, prev_state, policy_input):
    """
    This function returns the toal amount of H in the system after a deposit in a specific risk asset.
    H = H + delta_Q
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_R = policy_input['ri_deposit']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    # Wq = prev_state['Wq']
    Sq = prev_state['Sq']
    # R = pool.get_reserve(asset_id)
    # W = pool.get_weight(asset_id) 
    # S = pool.get_share(asset_id) 
    P = pool.get_price(asset_id) 
   
    Ri_plus = Ri + delta_R
    Ci_plus = Ci * ((Ri + delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri + delta_R) ** (-a))) ** (- (1 / a))

    Q_plus = Q * (Ci / Ci_plus) * ((Y / Y_plus) ** (a)) * ((Ri_plus / Ri) ** (a + 1))

    return ('H', Q_plus)


def r_to_r_swap_H(params, substep, state_history, prev_state, policy_input):
    H = prev_state['H']
    """
    calculates the total amount H that results from a swap between two risk assets according to the specification from 3-3-21
    """
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Ki = params['Ki']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)

    if delta_Ri == 0:
        return ('H', H)
    else:
        W_ratio = Wi / Wk
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Rk = Rk * (1 - power)

        Si = pool.get_share(asset_id)
        Sk = pool.get_share(purchased_asset_id) 

        price_q_i = calc_price_q_i(Ki, Ri, Q, Si, Sq)
        delta_Q_i = price_q_i * delta_Ri

        price_q_k = calc_price_q_i(Ki, Rk, Q, Sk, Sq)
        delta_Q_k = price_q_k * delta_Rk


########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

        delta_Q = delta_Q_i - delta_Q_k

        return ('H', H + delta_Q)

def resolve_remove_Liquidity_H(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    a = params['a']
    delta_S = policy_input['UNI_burn']
    Ri = pool.pool[asset_id]['R']
    Y = prev_state['Y']
    Ci = pool.get_coefficient(asset_id)
    Q = prev_state['Q']
    # Wq = prev_state['Wq']
    Sq = prev_state['Sq']
    # R = pool.get_reserve(asset_id)
    # W = pool.get_weight(asset_id) 
    # S = pool.get_share(asset_id) 
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
    # Wq = prev_state['Wq']
    Sq = prev_state['Sq']
    # R = pool.get_reserve(asset_id)
    # W = pool.get_weight(asset_id) 
    # S = pool.get_share(asset_id) 
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
    
    #Wi = pool.get_weight(asset_id)
    #Wq = prev_state['Wq']
    #Sq = prev_state['Sq']
    #Wi = pool.get_weight(asset_id)
    #Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    # Before weight to share conversion
    # delta_R = delta_S / S * R
    # JS May 19: reserve adjustment for price invariance, a neq 1
    # Q = prev_state['Q']
    # P = pool.get_price(asset_id)
    # delta_R = (delta_S / Sq) * (Q / P)
    # delta_W = (delta_R / R) * Wi

    delta_R = R * ( delta_S / S )
    delta_C = C * ( ((R - delta_R) / R) ** (a+1) - 1 )

    # Q = prev_state['Q']
    # P = pool.get_price(asset_id)
    # delta_R = (delta_S / Sq) * (Q / P)

    # a = params['a']
    #delta_W = (delta_R / R) * Wi
    # delta_R = (Wq / Wi) * (delta_S / Sq) * R
    # delta_W = Wq * delta_S / Sq
    # print('POOL REMOVE LIQUIDITY Timestep === ', prev_state['timestep'],'delta W ', delta_W, 'delta S ', delta_S)
    # print(f"POOL REMOVE LIQUIDITY share fraction of pool = {delta_S / Sq}")
    # print(f"POOL REMOVE LIQUIDITY share fraction of reserve = {delta_R / R}")
    # print(f"POOL REMOVE LIQUIDITY weight fraction of pool = {delta_W / Wi}")
    
    # JS July 9, 2021: Note the minus sign added to delta_C in function call below: 
    # delta_C < 0, but remove_liquidity_pool expects a positive number
    pool.remove_liquidity_pool(asset_id, delta_R, delta_S, -delta_C, a)

    return ('pool', pool)

def q_to_r_Qh_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    if delta_Q == 0:
        return ('Q', Q)
    else:
        # # CHECKED SIGN because it is being subtracted, made negative, to make positive
        # delta_Ri =- Ri * ((Q / (Q + delta_Q))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool


        # delta_Sq = delta_Q / Q * Sq  # making negative
        # delta_Si = delta_Sq
        return ('Q', Q + delta_Q)

def r_to_q_Qh_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
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

    if delta_Ri == 0:
        return ('Q', Q)
    else:

        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool
        # print('delta_Q', delta_Q)

        return ('Q', Q + delta_Q)

def q_to_r_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns Q after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    if delta_Q == 0:
        return ('Q', Q)
    else:
        W_ratio = Wi / Wq
        inner_term = Q /(Q+ delta_Q*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Ri = Ri * (1 - power)
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        
        delta_prime_Q = delta_Q + Q * delta_Si / Sq
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
    return ('Q', Q + delta_Q - delta_prime_Q)

def q_to_r_Sq_discrete(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    if delta_Q == 0:
        return ('Sq', Sq)
    else:
        # # CHECKED SIGN because it is being subtracted, made negative, to make positive
        # delta_Ri =- Ri * ((Q / (Q + delta_Q))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool


        delta_Sq = delta_Q / Q * Sq  # making negative
        # delta_Si = delta_Sq
    
        return ('Sq', Sq + delta_Sq) 
        
def q_to_r_Sq(params, substep, state_history, prev_state, policy_input):
    asset_id = policy_input['asset_id'] # defines asset subscript
    Sq = prev_state['Sq']
    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    if delta_Q == 0:
        return ('Sq', Sq)

    else:
        W_ratio = Wq / Wi
        inner_term = Q /(Q+ delta_Q*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Ri = Ri * (1 - power)
    

###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = - delta_Ri / Ri * Si  # making negative
        

########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
        return ('Sq', Sq - delta_Si) 

def q_to_r_pool_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade where delta_Q is the amount being sold according to the specification from 3-3-21
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

    if delta_Q == 0:
        return ('pool', pool)
    else:
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Ri =- Ri * ((Q / (Q + delta_Q))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool
  
        pool.q_to_r_pool(asset_id, delta_Ri) # subtracts from pool

        delta_Sq = delta_Q / Q * Sq  # making negative
        delta_Si = delta_Sq
        pool.swap_share_pool(asset_id, delta_Si) #adds
        # print('delta_Ri', delta_Ri)
        # print('delta_Si', delta_Si)

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
    #Wq = prev_state['Wq']

    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)
    #Wi = pool.get_weight(asset_id)

    a = params['a']

    if delta_Q == 0 or delta_Q < 0:
        return ('pool', pool)
    else:
        delta_Ri = ( (1/Ci) * ((Q*Y) / (Q + delta_Q))**(-a) - (Y**(-a) / Ci) + Ri**(-a) )**(1/a) - Ri
    
        pool.q_to_r_pool(asset_id, delta_Ri)

        return ('pool', pool)

def r_to_q_pool_discrete(params, substep, state_history, prev_state, policy_input):
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

    if delta_Ri == 0:
        return ('pool', pool)

    else:
        pool.r_to_q_pool(asset_id, delta_Ri) # adds delta_Ri to pool
        delta_Q = Q * ((Ri / (Ri + delta_Ri))**(Sq / Si) - 1) # blue box force negative because sebtracted from pool
        # print("delta Q", delta_Q)
        
        delta_Sq = delta_Q / Q * Sq  # making negative

        delta_Si = delta_Sq
        pool.swap_share_pool(asset_id, delta_Si)

########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
        return ('pool', pool)

def r_to_q_pool(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between a risk asset and the base asset where delta_R is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']

    Ri = pool.get_reserve(asset_id)

    if delta_Ri == 0:
        return ('pool', pool)

    else:
        pool.r_to_q_pool(asset_id, delta_Ri)

###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        pool.swap_share_pool(asset_id, delta_Si)

########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
        return ('pool', pool)

def r_to_r_pool(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the pool variable after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)

    if delta_Ri == 0:
        return ('pool', pool)
    else:
        W_ratio = Wi / Wk
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Rk = Rk * (1 - power)
    
        pool.r_to_q_pool(asset_id, delta_Ri) # adds Ri to pool
        pool.q_to_r_pool(purchased_asset_id, delta_Rk) # removes Rk from pool
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        pool.swap_share_pool(asset_id, delta_Si)

        Sk = pool.get_share(purchased_asset_id) 
        delta_Sk = - delta_Rk / Rk * Sk
        pool.swap_share_pool(purchased_asset_id, delta_Sk)
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

        return ('pool', pool)

def r_to_r_pool_discrete(params, substep, state_history, prev_state, policy_input):
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
        delta_Si = - (delta_Ri / Ri) * Si**2
        # print('delta_Ri', delta_Ri)
        # print('Ri', Ri)
        # print('Si', Si)
        # print('delta_Si', delta_Si)

        Si_ratio = delta_Si / Si
        delta_Sk = Sk * ((1/(1+Si_ratio))-1)

        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool
        # print('delta_Rk', delta_Rk)
        # print('delta_Sk', delta_Sk)

        pool.r_to_q_pool(asset_id, delta_Ri) # adds Ri to pool
        pool.q_to_r_pool(purchased_asset_id, delta_Rk) # SUBTRACTS removes Rk from pool

        pool.swap_share_pool(asset_id, delta_Si) # adds share
        pool.swap_share_pool(purchased_asset_id, delta_Sk) # adds share
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################

        return ('pool', pool)

def r_to_r_pool_temp(params, substep, state_history, prev_state, policy_input):
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
        delta_Si = - (delta_Ri / Ri) * Si**2
        # print('delta_Ri', delta_Ri)
        # print('Ri', Ri)
        # print('Si', Si)
        # print('delta_Si', delta_Si)
        delta_Rk = (delta_Ri / Ri) * (Si / Sk) * Rk # first version not blue box force negative because sebtracted from pool

        delta_Sk = - delta_Si / Si * Sk


        pool.r_to_q_pool(asset_id, delta_Ri) # adds Ri to pool
        pool.q_to_r_pool(purchased_asset_id, delta_Rk) # removes Rk from pool

        pool.swap_share_pool(asset_id, delta_Si) # adds share
        pool.swap_share_pool(purchased_asset_id, delta_Sk) # adds share
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################

        return ('pool', pool)

def r_to_r_swap_Qh_discrete(params, substep, state_history, prev_state, policy_input):
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
        return ('Q', Q)

    else:
        delta_Si = - (delta_Ri / Ri) * Si**2

        Si_ratio = delta_Si / Si
        delta_Sk = Sk * ((1/(1+Si_ratio))-1)
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool


        delta_Sq = delta_Si + delta_Sk

        delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################
        # print("Q Swap")
        # print("delta Q", delta_Q)
        return ('Q', Q + delta_Q)

def r_to_r_swap_Qh_temp(params, substep, state_history, prev_state, policy_input):
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
        return ('Q', Q)

    else:
        delta_Si = - (delta_Ri / Ri) * Si**2

        delta_Rk = (delta_Ri / Ri) * (Si / Sk) * Rk # first version not blue box force negative because sebtracted from pool

        delta_Sk = - delta_Si / Si * Sk

        delta_Sq = delta_Si + delta_Sk

        delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################
        # print("Q Swap")
        # print('Q', Q)
        # print("delta Q", delta_Q)
        return ('Q', Q + delta_Q)

def r_to_r_swap_H_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity H after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    H = prev_state['H']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('H', H)

    else:
        delta_Si = - (delta_Ri / Ri) * Si**2
        
        Si_ratio = delta_Si / Si
        delta_Sk = Sk * ((1/(1+Si_ratio))-1)
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool


        delta_Sq = delta_Si + delta_Sk

        delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################
        # print("H Swap")
        # print('H', H)
        # print("delta Q", delta_Q)
        return ('H', H + delta_Q)
        
def r_to_r_swap_H_temp(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity H after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    purchased_asset_id = policy_input['purchased_asset_id'] # defines asset subscript
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    H = prev_state['H']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)
    Si = pool.get_share(asset_id)

    Rk = pool.get_reserve(purchased_asset_id)
    Wk = pool.get_weight(purchased_asset_id)
    Sk = pool.get_share(purchased_asset_id) 

    if delta_Ri == 0:
        return ('H', H)

    else:
        delta_Si = - (delta_Ri / Ri) * Si**2

        delta_Rk = (delta_Ri / Ri) * (Si / Sk) * Rk # first version not blue box force negative because sebtracted from pool

        delta_Sk = - delta_Si / Si * Sk

        delta_Sq = delta_Si + delta_Sk

        delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################
        # print("H Swap")
        # print('H', H)
        # print("delta Q", delta_Q)
        return ('H', H + delta_Q)

def r_to_r_swap_Sq_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Sq after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
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
        delta_Si = - (delta_Ri / Ri) * Si**2
        Si_ratio = delta_Si / Si
        delta_Sk = Sk * ((1/(1+Si_ratio))-1)
        # CHECKED SIGN because it is being subtracted, made negative, to make positive
        delta_Rk = - Rk * ((Ri / (Ri + delta_Ri))**(Si / Sk) - 1) # blue box force negative because sebtracted from pool



        delta_Sq = delta_Si + delta_Sk

        # delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################

        return ('Sq', Sq + delta_Sq)
def r_to_r_swap_Sq_temp(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Sq after a trade between two risk assets where delta_R is the amount being sold according to the specification from 3-18-21
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
        delta_Si = - (delta_Ri / Ri) * Si**2

        delta_Rk = (delta_Ri / Ri) * (Si / Sk) * Rk # first version not blue box force negative because sebtracted from pool

        delta_Sk = - delta_Si / Si * Sk

        delta_Sq = delta_Si + delta_Sk

        # delta_Q = delta_Sq / Sq * Q
###############################################################################################
########### Hydra Mechanism Design, Trade/Swap 3-18-21 #######################################

        return ('Sq', Sq + delta_Sq)

####################### POOL CLASS VERSION ########################################################
def r_to_q_Qh(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between a risk asset and the base asset where delta_Ri is the amount being sold according to the specification from 3-3-21
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)


    if delta_Ri == 0:
        return ('Q', Q)
    else:
        W_ratio = Wi / Wq
        inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio

        delta_Q = Q * (1 - power)

###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        
        delta_prime_Q = delta_Q + Q * delta_Si / Sq
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################


        return ('Q', Q - delta_Q + delta_prime_Q)

def r_to_q_H_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity H after a trade between a risk asset and the base asset where delta_Ri is the amount being sold according to the specification from 3-3-21
    """
    H = prev_state['H']
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user

    if delta_Ri == 0:
        return ('H', H)
    else:
        return ('H', H)

def r_to_q_H(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity H after a trade between a risk asset and the base asset where delta_Ri is the amount being sold according to the specification from 3-3-21
    """
    H = prev_state['H']

    asset_id = policy_input['asset_id'] # defines asset subscript
    delta_Ri = policy_input['ri_sold'] #amount of Q being sold by the user
    pool = prev_state['pool']
    Q = prev_state['Q']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)


    if delta_Ri == 0:
        return ('H', H)
    else:
        # W_ratio = Wi / Wq
        # inner_term = Ri /(Ri + delta_Ri*(params['fee_numerator']/params['fee_denominator']))
        # power = inner_term**W_ratio

        # delta_Q = Q * (1 - power)

###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        
        delta_H = Q * delta_Si / Sq
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################

        return ('H', H + delta_H)

def q_to_r_H_discrete(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between the base asset and a risk asset where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    H = prev_state['H']
    return ('H',H)

def q_to_r_H(params, substep, state_history, prev_state, policy_input):
    """
    This function calculates and returns the quantity Q after a trade between the base asset and a risk asset where delta_Q is the amount being sold according to the specification from 3-3-21
    """
    H = prev_state['H']

    asset_id = policy_input['asset_id'] # defines asset subscript

    delta_Q = policy_input['q_sold'] #amount of Q being sold by the user
    Q = prev_state['Q']
    pool = prev_state['pool']
    Wq = prev_state['Wq']
    Sq = prev_state['Sq']

    Ri = pool.get_reserve(asset_id)
    Wi = pool.get_weight(asset_id)

    if delta_Q == 0:
        return ('H', H)
    else:
        W_ratio = Wi / Wq
        inner_term = Q /(Q+ delta_Q*(params['fee_numerator']/params['fee_denominator']))
        power = inner_term**W_ratio
        delta_Ri = Ri * (1 - power)
###############################################################################################
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
        Si = pool.get_share(asset_id)
        delta_Si = delta_Ri / Ri * Si
        
        delta_H = Q * delta_Si / Sq
########### YELLOW BOX HYDRA SPEC SWAP RISK ASSETS 3-3-21 #####################################
###############################################################################################
        return ('H', H + delta_H)


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
    Wq = prev_state['Wq']

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
    Wq = prev_state['Wq']
    Si = pool.get_share(asset_id)

    P = pool.get_price(asset_id) 
    delta_R = (delta_S / Si) * (Q / P)
    
    Ri_plus = Ri - delta_R
    print('Ri_plus = ', Ri_plus)
    Ci_plus = Ci * ((Ri - delta_R) / Ri) ** (a+1)
    Y_plus = ((Y ** (-a)) - Ci * (Ri ** (-a)) + Ci_plus * ((Ri - delta_R) ** (-a))) ** (- (1 / a))
    

    return ('Y', Y_plus)


