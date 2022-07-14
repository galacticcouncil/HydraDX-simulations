from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from .state_variables import initial_state
from .partial_state_update_block import partial_state_update_block
from .sys_params import params , initial_values
from .sim_setup import SIMULATION_TIME_STEPS, MONTE_CARLO_RUNS
from .parts.v2_asset_utils import V2_Asset

# from copy import deepcopy
from cadCAD import configs
# sys_params: Dict[str, List[int]] = sys_params
import numpy as np
import math
import copy

# Initialize random seed for numpy random initialization, for replication of results
np.random.seed(42)

sim_config = config_sim(
    {
        'N': MONTE_CARLO_RUNS, 
        'T': range(SIMULATION_TIME_STEPS), # number of timesteps
        'M': params,
    }
)

exp = Experiment()

exp.append_configs(
    sim_configs=sim_config,
    initial_state=initial_state,
    partial_state_update_blocks=partial_state_update_block
    # config_list=configs
        )
