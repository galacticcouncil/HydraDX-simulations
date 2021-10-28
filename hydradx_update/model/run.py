from cadCAD import configs
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor

from .partial_state_update_block import partial_state_update_block


def config(config_dict, initial_state) -> None:
    config_dict = config_dict
    initial_state = initial_state

    exp = Experiment()

    exp.append_configs(
        sim_configs=config_sim(config_dict),
        initial_state=initial_state,
        partial_state_update_blocks=partial_state_update_block
    )


def run() -> list:
    '''
    Definition:
    Run simulation
    '''
    # config = input_config
    # Single
    exec_mode = ExecutionMode()
    local_mode_ctx = ExecutionContext(context=exec_mode.local_mode)

    simulation = Executor(exec_context=local_mode_ctx, configs=configs)
    raw_system_events, tensor_field, sessions = simulation.execute()

    # return postprocessing(raw_system_events)
    return raw_system_events
