from .system import *

partial_state_update_block = [
    {
        # system.py
        'policies': {
        },
        'variables': {
            'state': evolve_market
        }
    },
    {
        # system.py
        'policies': {
        },
        'variables': {
            'state': execute_trades
        }
    },

    # {
    #     # system.py
    #     'policies': {
    #         'user_action': actionDecoder
    #     },
    #     'variables': {
    #         'AMM': mechanismHub_AMM,
    #         'uni_agents': agenthub,
    #     }
    # },
    #
    # {
    #     'policies': {
    #     },
    #     'variables': {
    #         'AMM': posthub
    #     }
    # },

]
