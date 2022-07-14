from ..amm.global_state import GlobalState


def execute_trades(params, substep, state_history, prev_state, policy_input):
    state: GlobalState = prev_state['state']
    agents = state.agents
    for agent_id, agent in agents.items():
        if agent.trade_strategy:
            state = agent.trade_strategy.execute(state=state, agent_id=agent.unique_id)
    return 'state', state


def evolve_market(params, substep, state_history, prev_state, policy_input):
    state: GlobalState = prev_state['state']
    if state.evolve_function:
        state = state.evolve()
    return 'state', state
