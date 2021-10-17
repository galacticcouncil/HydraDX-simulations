"""
State variables are defined here.

- agent parameters (agent dataframe)
- asset parameters (asset dataframe)
- potential new asset parameters (new_asset dataframe)

- the Hydra pool
- all Uniswap pools
- the initial state object
"""

# Dependences
from collections import defaultdict
from .parts.utils import *
from .sys_params import params, initial_values, temp_a, C
from .parts.v2_asset_utils import V2_Asset
import pandas as pd
########## AGENT INITIALIZATION ##########
number_of_agents = 8

# Configure agents for agent-based model
agents_df = pd.DataFrame({
    'r_i_out': 0.0, # reserve asset not in pool
    'r_i_in': 0.0, # reserve asset put into pool- virtual
    'h': 0.0, # base asset not in pool
    'q_i': 0.0, # base asset in pool- virtual (if added Q)
    's_i': 0.0, # i_reserve asset share of pool
    's_q': 0.0, # q_base asset share of pool
    'r_j_out': 0.0, # reserve asset not in pool
    'r_j_in': 0.0, # reserve asset put into pool- virtual
    'q_j': 0.0, # base asset in pool- virtual (if added Q)
    's_j': 0.0, # i_reserve asset share of pool
    }, index=[0], dtype=float)
agents_df = pd.concat([agents_df]*number_of_agents, ignore_index=True)
# Adding IDs to agents
agents_df.insert(0, 'm', range(0, len(agents_df)))

agents_df['r_i_out'] = 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 1000.0, 1000.0
agents_df['h'] =  140000, 140000.0, 140000, 140000, 140000, 140000, 10.0, 0.0
agents_df['s_i'] =   150000, 150000.0, 0.0, 150000, 150000, 150000, 0.0, 0.0
agents_df['q_i'] =  170000, 170000.0, 170000, 170000, 170000, 170000, 0.0, 0.0
agents_df['r_i_in'] =  110000, 110000.0, 110000, 110000, 110000, 110000, 0.0, 0.0
agents_df['r_j_out'] = 120000, 120000.0, 120000, 120000, 120000, 120000, 0.0, 0.0
agents_df['s_j'] =  160000, 160000.0, 160000, 160000, 160000, 160000, 0.0, 0.0
agents_df['r_j_in'] =  130000, 130000.0, 130000, 130000, 130000, 130000, 0.0, 0.0
agents_df['q_j'] =  180000, 180000.0, 180000, 180000, 180000, 180000, 0.0, 0.0
##################### ASSET TYPES ###################################
############
#pool = Asset('i', initial_values['Ri'], initial_values['Si'], (initial_values['Q']/initial_values['Sq'])/(initial_values['Ri']/initial_values['Si']))
#pool.add_new_asset('j', initial_values['Rj'], initial_values['Sj'], (initial_values['Q']/initial_values['Sq'])/(initial_values['Rj']/initial_values['Sj']))
#pool.add_new_asset('k', initial_values['Rj'], initial_values['Sj'], (initial_values['Q']/initial_values['Sq'])/(initial_values['Rj']/initial_values['Sj']))

# JS July 8th, 2021: Add initial price function
a = temp_a
def initial_price_in_Q(R, C, Q, Y, a):
    return (Q * Y**(a)) * (C / R**(a+1))

pool = V2_Asset('i', initial_values['Ri'], initial_values['Ci'],
        initial_price_in_Q(initial_values['Ri'], initial_values['Ci'], initial_values['Q'], initial_values['Y'], a)
)
pool.add_new_asset('j', initial_values['Rj'], initial_values['Cj'],
        initial_price_in_Q(initial_values['Rj'], initial_values['Cj'], initial_values['Q'], initial_values['Y'], a)
)

# JS July 8th, 2021: Adding of asset 'k' should be done consistently with V2 Spec--perhaps in sys_params.py as an 'equal player' to assets 'i' and 'j'
#pool.add_new_asset('k', initial_values['Rj'], initial_values['Sj'], (initial_values['Q']/initial_values['Sq'])/(initial_values['Rj']/initial_values['Sj']))

#############################################################################################
UNI_P_RQi = initial_values['UNI_Qi'] / initial_values['UNI_Ri']
UNI_P_RQj = initial_values['UNI_Qj'] / initial_values['UNI_Rj']
UNI_P_ij = initial_values['UNI_ij'] / initial_values['UNI_ji']



## Initial state object
initial_state = {
    # UNISWAP Global Vars
    'UNI_Qi': initial_values['UNI_Qi'],
    'UNI_Ri': initial_values['UNI_Ri'],
    'UNI_Si': initial_values['UNI_Si'],
    'UNI_Qj': initial_values['UNI_Qj'],
    'UNI_Rj': initial_values['UNI_Rj'],
    'UNI_Sj': initial_values['UNI_Sj'],
    'UNI_ij': initial_values['UNI_ij'],
    'UNI_ji': initial_values['UNI_ji'],
    'UNI_Sij': initial_values['UNI_Sij'],
    # Uniswap Local Vars
    'uni_agents': agents_df,
    # Metrics
    'UNI_P_RQi' : UNI_P_RQi,
    'UNI_P_RQj' : UNI_P_RQj,
    'UNI_P_ij' : UNI_P_ij,
    # assets
    # 'asset' : asset_df,
    'pool' : pool, 
    # Hydra
    'Q': initial_values['Q'],
    'H': initial_values['H'],
    'Sq': initial_values['Sq'],
    # Hydra Y Risk Asset Pool Constant
    'Y': initial_values['Y'],
    # Hydra Local Vars
    'hydra_agents': agents_df,
    'C': C,
    'asset_random_choice': 'i',
    'trade_random_size': 1000,
    'trade_random_direction': 'test_q_for_r',
    'purchased_asset_id': 'N/A',
    'fee_revenue': defaultdict(lambda: 0), # Dictionary where the key is the asset & the value is the fee revenue generated from asset
    'fee_revenue_2': [0, 0, 0], # Empty array to represent the different revenues collected from different fee percentages
    'oracle_price_i': 2,
    'oracle_price_j': 3,
}

