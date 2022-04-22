from .system import *
from .simulation_updates import execute_trades

partial_state_update_block = [
    {
        # system.py
        'policies': {
        },
        'variables': {
            # 'external': externalHub
            'WorldState': execute_trades
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
