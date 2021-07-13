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

def init_price(R, C, Q, Y, a):
    """
    V2 Spec June 28, 2021 definition of initial price
    """  
    return (Q * Y**(a)) * (C / R**(a+1))

def liquidity_randomizer(fake_mc_runs, starting_L_mu, starting_L_sigma, distribution):
    if distribution == 'normal': 
        array_random = np.random.normal(starting_L_mu, starting_L_sigma,size=fake_mc_runs)
    elif distribution == 'lognormal':
        mean_underlying = math.log(starting_L_mu)
        std_underlying = math.sqrt(math.log (1 + starting_L_sigma**2 / starting_L_mu**2))
        array_random = np.random.lognormal(mean_underlying, std_underlying, size=fake_mc_runs)

    # print('array_random', array_random)

    return array_random

fake_mc_runs = 1
block_one_mean_std_ratio = 50
starting_L_mu = 100000
starting_L_sigma = starting_L_mu / block_one_mean_std_ratio 
#starting sigma should be zero for block2 experiments
#starting_L_sigma = 0
Ri_array = liquidity_randomizer(fake_mc_runs, starting_L_mu, starting_L_sigma, 'lognormal')
Rj_array = liquidity_randomizer(fake_mc_runs, starting_L_mu, starting_L_sigma, 'lognormal')
print(Ri_array, Rj_array)

for n in range(fake_mc_runs):
    # print(n)

    initial_state = copy.deepcopy(initial_state)

    # initial_state = updated initial_state
    initial_state['UNI_Ri'] = Ri_array[n]
    initial_state['UNI_Qi'] = 2 * Ri_array[n]

    initial_state['UNI_Rj'] = Rj_array[n]
    initial_state['UNI_Qj'] = 2 * Rj_array[n]
    # print('n', n, 'Ri', initial_state['UNI_Ri'])
    # print('n', n, 'Rj', initial_state['UNI_Rj'])
    initial_state['UNI_ij'] = Ri_array[n]
    initial_state['UNI_ji'] = Rj_array[n]
    initial_state['UNI_Sij'] = Ri_array[n] * Rj_array[n]
    initial_state['UNI_P_ij'] = Ri_array[n] / Rj_array[n]
    # print('n', n, 'Sij', initial_state['UNI_Sij'])

    initial_state['Q'] =  2 * Ri_array[n] + 2 * Rj_array[n]
    initial_state['H'] =  2 * Ri_array[n] + 2 * Rj_array[n]

    # pool = V2_Asset('i', initial_state['UNI_Ri'], initial_values['Si'], (initial_state['Q']/initial_values['Sq'])/(initial_state['UNI_Ri']/initial_values['Si']))
    # pool.add_new_asset('j', initial_state['UNI_Rj'], initial_values['Sj'], (initial_state['Q']/initial_values['Sq'])/(initial_state['UNI_Rj']/initial_values['Sj']))
    # pool.add_new_asset('k', initial_state['UNI_Rj'], initial_values['Sj'], (initial_state['Q']/initial_values['Sq'])/(initial_state['UNI_Rj']/initial_values['Sj']))

    # JS July 8, 2021: Initialization of pool using V2 Spec, with placeholder a = 1 value (because 'pool' var
    # is over-written in config loop below)
    # TODO: Add new asset 'k' correctly so pool has 3 risk assets

    pool = V2_Asset('i', initial_state['UNI_Ri'], initial_values['Ci'],
        init_price(initial_state['UNI_Ri'], initial_values['Ci'], initial_values['Q'], initial_values['Y'], 1)
    )
    pool.add_new_asset('j', initial_state['UNI_Rj'], initial_values['Cj'],
        init_price(initial_state['UNI_Rj'], initial_values['Cj'], initial_values['Q'], initial_values['Y'], 1)
    )

    initial_state['pool'] = pool
    exp.append_configs(
        sim_configs=sim_config,
        initial_state=initial_state,
        partial_state_update_blocks=partial_state_update_block
        # config_list=configs
        )
# print('EXP DIR == ',dir(exp))
# print('configs third',configs)
# print('configs length ',len(configs))

for i in configs:
    configs =  copy.deepcopy(configs)

    # print('i ========', i)
    # Initial state from the dictionary in state_variables.py 
    initial_state = copy.deepcopy(initial_state)
    # attribs = dir(i)
    # # a = M['a']
    # print('attribs ', attribs)

    # Get initial_state from configs
    config_init_state =  copy.deepcopy(i.initial_state)
    # Get parameter a from configs
    config_param = copy.deepcopy(i.sim_config)
    a = config_param['M']['a']

    Omni_P_RQi = init_price(config_init_state['pool']['i']['R'], config_init_state['pool']['i']['C'], config_init_state['Q'], config_init_state['Y'], a)
    Omni_P_RQj = init_price(config_init_state['pool']['j']['R'], config_init_state['pool']['j']['C'], config_init_state['Q'], config_init_state['Y'], a)

    # Initial base asset amount is value in HDX of pool
    config_init_state['Q'] = Omni_P_RQi * config_init_state['pool']['i']['R'] + Omni_P_RQj * config_init_state['pool']['j']['R']

    # print('Omni_P_RQi ==================', Omni_P_RQi)
    # print(f"Initial Q HDX: {config_init_state['Q']}")

    config_pool = V2_Asset('i', config_init_state['pool']['i']['R'], config_init_state['pool']['i']['C'], Omni_P_RQi)
    config_pool.add_new_asset('j', config_init_state['pool']['j']['R'], config_init_state['pool']['j']['C'], Omni_P_RQj)
    # TODO: Add asset 'k' in a consistent fashion with V2 Spec, so there are 3 risk assets in OMNIPool
    # config_pool.add_new_asset('j', config_init_state['pool']['k']['R'], config_init_state['pool']['k']['C'], Omni_P_RQk)

    #config_pool = V2_Asset('i', config_init_state['UNI_Ri'], initial_values['Si'], Omni_P_RQi)
    #config_pool.add_new_asset('j', config_init_state['UNI_Rj'], initial_values['Sj'],Omni_P_RQj)
    #config_pool.add_new_asset('k', config_init_state['UNI_Rj'], initial_values['Sj'], Omni_P_RQj)
    
    config_init_state['pool'] = config_pool

    # Update Initial State in config object
    # i.initial_state.update(config_init_state) # THIS DID NOT WORK!!! Seemed to work, but then config only took last value
    i.initial_state = config_init_state

    print('config_init_state ', i.initial_state['pool'])
    # print('config_param a value ============= ', a)

# def get_M(k, v):
#     if k == 'sim_config':
#         k, v = 'M', v['M']
#     return k, v

# config_ids = [
#     dict(
#         get_M(k, v) for k, v in config.__dict__.items() if k in ['simulation_id', 'run_id', 'sim_config', 'subset_id']
#     ) for config in configs
# ]
# print('Sim config_ids', config_ids)
# print('configs fourth',configs)
print('configs length ',len(configs))
# exp.
# for c in exp.configs:
#     c.initial_state = deepcopy(c.initial_state)

#     # print("Params (config.py) : ", c.sim_config['M'])

#     c.initial_state['kappa'] = c.sim_config['M']['starting_kappa']
#     c.initial_state['alpha'] = c.sim_config['M']['starting_alpha']
#     c.initial_state['reserve'] = c.sim_config['M']['money_raised']
#     c.initial_state['supply'] = c.initial_state['kappa'] * \
#         c.sim_config['M']['money_raised'] / c.initial_state['spot_price']
#     c.initial_state['supply_free'] = c.initial_state['supply']
#     c.initial_state['invariant_V'] = (
#         c.initial_state['supply']**c.initial_state['kappa']) / c.initial_state['reserve']
#     c.initial_state['invariant_I'] = c.initial_state['reserve'] + \
#         (c.sim_config['M']['C'] * c.initial_state['alpha'])