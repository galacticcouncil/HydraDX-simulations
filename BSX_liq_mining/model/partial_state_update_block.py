from .system import *

partial_state_update_block = [
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
            #'uni_agents': agenthub,
        }
    },

]
