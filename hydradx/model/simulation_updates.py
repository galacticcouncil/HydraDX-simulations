from .amm import amm


# Behaviors
def actionDecoder(params, step, history, prev_state) -> dict:
    # t = len(history) - 1
    # # print(t)
    # action_list = params['action_list']
    # action_dict = params['action_dict']
    # if action_list[t] not in action_dict:
    #     return {}
    # policy_input = action_dict[action_list[t]]
    # We translate path dependent actions like market arb here
    return {}  # policy_input


def execute_trades(params, substep, state_history, prev_state, policy_input):
    new_state: amm.WorldState = prev_state['WorldState'].copy()
    for agent in new_state.agents.values():
        if agent.tradeStrategy:
            agent.tradeStrategy.execute(agent=agent, market=new_state.exchange)
    return 'WorldState', new_state
