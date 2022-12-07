from .amm.global_state import GlobalState
import time
from .cadCad import init_utils
from .cadCad.run import config as config_cadcad, run as run_cadcad


def run(initial_state: GlobalState, time_steps: int, silent: bool = False, use_cadcad: bool = False) -> list:
    """
    Definition:
    Run simulation
    """

    if use_cadcad:
        config_dict = init_utils.get_configuration(time_steps)
        config_cadcad(config_dict, {'state': initial_state})
        return run_cadcad()[1:]

    start_time = time.time()
    events = [None] * time_steps
    new_global_state = initial_state.copy()

    if not silent:
        print('Starting simulation...')

    for i in range(time_steps):

        # market evolutions
        new_global_state.evolve()

        # agent actions
        agents = new_global_state.agents
        for agent_id, agent in agents.items():
            if agent.trade_strategy:
                new_global_state = agent.trade_strategy.execute(new_global_state, agent.unique_id)

        events[new_global_state.time_step-1] = new_global_state.archive()

    if not silent:
        print(f'Execution time: {round(time.time() - start_time, 3)} seconds.')
    return events
