from .amm import amm


def execute_trades(params, substep, state_history, prev_state, policy_input):
    new_global_state = prev_state['state']
    agents = prev_state['state'].agents
    for agent_id, agent in agents.items():
        if agent.trade_strategy:
            agent.trade_strategy.execute(new_global_state, agent)
    return 'state', new_global_state
