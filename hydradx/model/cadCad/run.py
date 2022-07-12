from .amm.global_state import GlobalState
import time


def run(initial_state: GlobalState, time_steps: int) -> list:
    """
    Definition:
    Run simulation
    """

    start_time = time.time()
    events = [{'state': initial_state}]
    print('Starting simulation...')
    for i in range(time_steps):
        new_global_state = events[-1]['state'].copy()
        # agent actions
        agents = new_global_state.agents
        for agent_id, agent in agents.items():
            if agent.trade_strategy:
                new_global_state = agent.trade_strategy.execute(new_global_state, agent.unique_id)
        # market evolutions
        new_global_state.evolve()
        events.append({'timestep': i, 'state': new_global_state})

    print(f'Execution time: {round(time.time() - start_time, 3)} seconds.')
    return events[1:]
