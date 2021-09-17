"""
Partial state update block. 

Here the partial state update blocks are configurated by setting
- policies
- variables

for each state update block individually
"""
from .parts.uniswap import *
from .parts.metrics import *
from .parts.action_list import *
from .parts.v2_hydra import *
partial_state_update_block = [
    {
    #     # Resolve H and Weights
        'policies': {
    #         # 'external_action': exogoneous_process
        },
        'variables': {
    #         # UNISWAP WORLD
            # 'H': s_resolve_H,
            # 'asset' : s_asset_weight,
            'asset_random_choice': s_asset_random,
            'trade_random_size': s_trade_random,
            'trade_random_direction': s_direction_random,
        }
    },
     #{
     #    # uniswap.py asset i AND j
     #    'policies': {
     #        'user_action': actionDecoder
     #    },
     #    'variables': {
     #        # UNISWAP WORLD
     #        'UNI_Ri': mechanismHub_Ri,
     #        'UNI_Qi': mechanismHub_Q,
     #        'UNI_Si': mechanismHub_Si,
     #        'uni_agents': agenthub,
     #        'UNI_Rj': mechanismHub_Ri,
     #        'UNI_Qj': mechanismHub_Q,
     #        'UNI_Sj': mechanismHub_Si,
     #        'UNI_ij': mechanismHub_ij,
     #        'UNI_ji': mechanismHub_ji,
     #    }
     #},

        {
        ############ v2_hydra.py  #################################################
        'policies': {
            'user_action': actionDecoder
        },
        'variables': {
            # HYDRA WORLD
            # 'asset' : mechanismHub_asset,
            'Sq' : mechanismHub_Sq,
            'Q' : mechanismHub_Q_Hydra, 
            'H': mechanismHub_H_Hydra,
            'Y' : mechanismHub_Y,
            'hydra_agents': H_agenthub,
            'pool': mechanismHub_pool, # must be last or else updated would be used in omnipool updates
            'purchased_asset_id': s_purchased_asset_id, # writes from the action policy the outgoing risk asset
        }
    },
        ############ v2_hydra.py  #################################################
    {
        # Metrics
        'policies': {
            # 'trade_action': p_random_action
        },
        'variables': {
            # UNISWAP WORLD
            'UNI_P_RQi': s_swap_price_i,
            'UNI_P_RQj': s_swap_price_j,
            'UNI_P_ij': s_swap_price_ij,
            # 'UNI_agent_value': mechanismHub_Q,
            # 'asset' : s_asset_price,
            'pool': s_pool_price,
            # 'C': s_share_constant,

        }
    },
]
