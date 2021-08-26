import numpy as np

def addLiquidity_C(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns the coefficient C after a liquidity add, according to specification 6-28-21
    C = C + (R^+ / R) ** (a+1)
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_Ri = policy_input['ri_deposit']
    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)
    a = params['a']

    if delta_Ri == 0:
        return ('Ci', Ci)
    else:
        Ri_plus = Ri + delta_Ri
        Ci_plus = Ci + (Ri_plus / Ri) ** (a+1)
        return ('Ci', Ci_plus)

def removeLiquidity_C(params, substep, state_history, prev_state, policy_input):
    """
    This function updates and returns the coefficient C after a liquidity remove, according to specification 6-28-21
    C = C + (R^+ / R) ** (a+1)
    """
    asset_id = policy_input['asset_id'] # defines asset subscript
    pool = prev_state['pool']
    delta_S = policy_input['UNI_burn']
    Ri = pool.get_reserve(asset_id)
    Ci = pool.get_coefficient(asset_id)
    a = params['a']
    Q = prev_state['Q']
    Sq = prev_state['Sq']
    P = pool.get_price(asset_id) 

    delta_Ri = (delta_S / Sq) * (Q / P)

    if delta_Ri == 0:
            return ('Ci', Ci)
    else:
            Ri_plus = Ri + delta_Ri
            Ci_plus = Ci + (Ri_plus / Ri) ** (a+1)
            return ('Ci', Ci_plus)
