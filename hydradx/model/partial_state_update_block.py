from .simulation_updates import execute_trades

partial_state_update_block = [
    {
        'policies': {
        },
        'variables': {
            # 'external': externalHub
            'WorldState': execute_trades
        }
    },

    # {
    #     'policies': {
    #         'user_action': actionDecoder
    #     },
    #     'variables': {
    #         'AMM': mechanismHub_AMM,
    #         'uni_agents': agenthub,
    #     }
    # },
]
