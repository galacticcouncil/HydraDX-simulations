from .utils import *
import copy

def s_swap_price_i(params, substep, state_history, prev_state, policy_input):
    """
    Calculates and returns the swap price for a trade Qi for Ri
    """
    Q_reserve = int(prev_state['UNI_Qi'])
    R_reserve = int(prev_state['UNI_Ri'])

    # input 1000 q token get r
    delta_q = 1000
    delta_r = getInputPrice(delta_q, Q_reserve, R_reserve, params)
   
    return ('UNI_P_RQi', delta_q/delta_r)

def s_swap_price_j(params, substep, state_history, prev_state, policy_input):
    """
    Calculates and returns the swap price for a trade Qj for Rj
    """
    Q_reserve = int(prev_state['UNI_Qj'])
    R_reserve = int(prev_state['UNI_Rj'])

    # input 1000 q token get r
    delta_q = 1000
    delta_r = getInputPrice(delta_q, Q_reserve, R_reserve, params)
   
    return ('UNI_P_RQj', delta_q/delta_r)

def s_swap_price_ij(params, substep, state_history, prev_state, policy_input):
    """
    Calculates and returns the swap price for a trade Ri for Rj
    """
    Q_reserve = int(prev_state['UNI_ij'])
    R_reserve = int(prev_state['UNI_ji'])

    # input 1000 q token get r
    delta_q = 1000
    delta_r = getInputPrice(delta_q, Q_reserve, R_reserve, params)
    # print('swap price here')
   
    return ('UNI_P_ij', delta_q/delta_r)

def getAssetBasePrice(Q, Sq, R, S):
    """
    Calculates and returns the base asset price
    """
    return (Q/Sq)/(R/S)

def s_asset_price(params, substep, state_history, prev_state, policy_input):
    """
    Calculates the asset price using getAssetBasePrice and returns the asset variable.
    """
    asset = prev_state['asset']

    # MAKE ACROSS ALL ASSETS
    # 'id': 'i', # name of asset ex. i, j , k
   
    Q = prev_state['Q']
    Sq = prev_state['Sq']

    asset.P = asset.P.apply(lambda x: getAssetBasePrice(Q, Sq, asset.R, asset.S))

    return ('asset', asset)

def s_pool_price(params, substep, state_history, prev_state, policy_input):
    """
    Calculates the pool price using update_price and returns the pool variable
    JS July 8, 2021: updated method call signature according to V2 Spec
    """
    pool = copy.deepcopy(prev_state['pool'])

    Q = prev_state['Q']
    Y = prev_state['Y']
    Sq = prev_state['Sq']

    a = params['a']

    # pool.update_price(Q, Sq)
    pool.update_price_a(Q, Y, a,)
    

    return ('pool', pool)

def s_share_constant(params, substep, state_history, prev_state, policy_input):
    """
    Checks share multiplicative constant. Use for i,j,k for now
    """
    # MAKE a Product function loop through all assets dynamic
    # print('constant here 1')

    pool = prev_state['pool']

    Si = pool.get_share('i')
    Sj = pool.get_share('j')
    Sk = pool.get_share('k')
    print(type(Si))
    print(type(Sj))
    print(type(Sk))
    C = Si * Sj #* Sk    
    # print('constant here 2')

    return ('C', C)
