from .system import *
from .amm_loader import *


partial_state_update_block = [
    {
        # system.py
        'policies': {
            
        },
        'variables': {
            'AMM_list': init_amms,
            'AMM_parent': amm_load_parent,
            'AMM_child': amm_load_child, 
             
        }
    },
    {
        # system.py
        'policies': {
        },
        'variables': {
            'external': externalHub
        }
    },

    {
        # system.py 
        'policies': {
            'user_action': actionDecoder
        },
        'variables': {
            'AMM': mechanismHub_AMM,
            'uni_agents': agenthub,
        }
    },

    {
        'policies': {
        },
        'variables': {
            'AMM': posthub
        }
    },

]
