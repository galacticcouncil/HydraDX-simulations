# Behaviors
# from hydra_multi_class.model.sys_params import ACTION_LIST
# print("running file: action_list.py")
import numpy as np
import random
import math
import copy

def actionDecoder(params, step, history, prev_state):
    '''
    In this simplified model of Uniswap, we have not modeled user behavior. Instead, we map events to actions. Depending on the input system parameters 'params' a given action sequence is induced.
    '''
    action = {}
    action.clear()


    hydra_agents = copy.deepcopy(prev_state['hydra_agents'])
    action = {
        'q_sold': 0.0,     # q to r swap
        'ri_sold': 0.0,     # r to q swap
        'ri_deposit': 0.0,   # add liq
        'q_deposit': 0.0,  # if symmetric add liq
        'Si_burn': 0.0,    # remove liq    
        'action_id' : str(),
        'agent_id' : 0,
        'asset_id' : str(),
        'direction': str(),
        'direction_q': str()
    }
    
    # ACTION_LIST =  params['ACTION_LIST']
    # print(ACTION_LIST)


    timestep = prev_state['timestep']
    pool = prev_state['pool']


    # action['asset_id'] = prev_state['asset_random_choice']
    # action['q_sold'] = prev_state['trade_random_size'] * 2
    # action['ri_sold'] = prev_state['trade_random_size']
    # action['direction_q'] = prev_state['trade_random_direction']

    # action['ri_sold'] = prev_state['trade_random_size']

    ############# CREATE AGENT ID's ################
    
    agent0_id = 0
    agent1_id = 1
    agent2_id = 2
    agent3_id = 3
    agent4_id = 4
    agent5_id = 5
    agent6_id = 6
    agent7_id = 7
    
    ############# CREATE AGENT ID's ################
   
    ############## SET RANDOM SEQUENCE ##################
    # print('getting here', params['exo_random_sequence'], timestep)
    if params['exo_random_sequence'] == 'on':
        # print('getting in', params['exo_random_sequence'], timestep)
        if timestep == 1:
            action['asset_id'] = random.choice(['i'])
            action['purchased_asset_id'] = 'N/A'

        elif timestep == 2:

            action['asset_id'] = random.choice(['j'])
            action['purchased_asset_id'] = 'N/A'
  
        else:
            action['asset_id'] = random.choice(['i','j'])
            action['purchased_asset_id'] = 'N/A'

    # if params['exo_random_sequence'] == 'q_to_i_i_q':
    #     if timestep == 1:
    #         action['asset_id'] = random.choice(['i'])

    #         action['action_id'] = 'Ri_Purchase'
    #         action['purchased_asset_id'] = 'i'
    #         action['agent_id'] = hydra_agents['m'][agent6_id]
    #         action['q_sold'] = hydra_agents['h'][agent6_id]
    #     elif timestep == 2:
    #         action['asset_id'] = random.choice(['i'])

    #         action['ri_sold'] = hydra_agents['r_i_out'][agent6_id] * 0.1
    #         action['action_id'] = 'Q_Purchase'
    #         action['purchased_asset_id'] = 'q'
    #         action['agent_id'] = hydra_agents['m'][agent6_id]


    # ###############################################

    # ########## TEMP TEST ADD LIQ ############
    # ####### AGENT 2 ######################
    # if params['exo_liq'] == 'test_add':
    #     action['ri_deposit'] = 5000
    #     action['action_id'] = 'AddLiquidity'
    #     action['purchased_asset_id'] = 'N/A'

    #     # temp choose first agent
    #     if timestep == 10:
    #         action['ri_deposit'] = 50000

    #         action['agent_id'] = prev_state['uni_agents']['m'][agent2_id]
    #     else:
    #         action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]

    #     if action['asset_id'] == 'j':
    #         action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]
    #         action['ri_deposit'] = 5000
    #         action['purchased_asset_id'] = 'N/A'

    # ###############################################

    # ########## TEMP TEST REMOVE LIQ ############
    # ####### AGENT 3 ######################
    # if params['exo_liq'] == 'test_remove':
    #     print(prev_state['uni_agents']['s_i'][agent2_id])
    #     #print(removal)
    #     action['UNI_burn'] = prev_state['uni_agents']['s_i'][agent4_id] * 0.001

    #     action['purchased_asset_id'] = 'N/A'

    #     action['action_id'] = 'RemoveLiquidity'
    #     if timestep == 490:
    #         action['agent_id'] = prev_state['uni_agents']['m'][agent2_id]
    #         action['UNI_burn'] = prev_state['uni_agents']['s_i'][agent2_id] - 150000

    #     else:
    #         action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]

    #     if action['asset_id'] == 'j':
    #         # print('remove j',step,action['asset_id'])
    #         action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]
    #         action['UNI_burn'] = prev_state['uni_agents']['s_i'][agent4_id] * 0.001
    #         action['purchased_asset_id'] = 'N/A'

    # ###############################################

    # ########## TEMP TEST SELL R FOR R ############
    # ####### AGENT 7 ######################
    if params['exo_trade'] == 'test_r_for_r':
        # print("I want to trade:")
        action['ri_sold'] = hydra_agents['r_i_out'][agent7_id]
        action['action_id'] = 'R_Swap'
        action['purchased_asset_id'] = 'j'
        action['direction'] = 'ij'

        # temp choose first agent
        action['agent_id'] = hydra_agents['m'][agent7_id]
        # print('prev_state hydra_agents -->', prev_state['hydra_agents']['r_i_out'][agent7_id])
        if action['asset_id'] == 'j':
            action['agent_id'] = hydra_agents['m'][agent7_id]
            action['ri_sold'] = hydra_agents['r_j_out'][agent7_id]
            action['purchased_asset_id'] = 'i'
    #         action['direction'] = 'ji'           
    print("action['ri_sold']",action['ri_sold'], ' of ', action['asset_id'])
    return action

def s_purchased_asset_id(params, step, history, prev_state, policy_input):
    purchased_asset_id = policy_input['purchased_asset_id']
    
    return 'purchased_asset_id', purchased_asset_id

def s_asset_random(params, step, history, prev_state, policy_input):
    if params['exo_trade'] == 'pass' and params['exo_liq'] == 'pass':
        # there are no trades to be made this timestep
        return 'asset_random_choice', np.nan
    else:
    # print('here')
    # sigma = params['sigma']
    # mu = params['mu']
        asset_random_choice = random.choice(['i', 'j'])
        # asset_random_choice = random.choice(['i'])
    # trade_size_random_choice =math.ceil(np.random.normal(mu, sigma))
    # print('not here')
        return 'asset_random_choice', asset_random_choice

def s_trade_random(params, step, history, prev_state, policy_input):
    if params['exo_trade'] == 'pass' and params['exo_liq'] == 'pass':
        # there are no trades to be made this timestep
        return 'trade_random_size', np.nan
    else:
        sigma = params['sigma']
        mu = params['mu']
        # asset_random_choice = random.choice(['i', 'j'])
        trade_size_random_choice = math.ceil(np.random.normal(mu, sigma))
        return 'trade_random_size', trade_size_random_choice
    
def s_direction_random(params, step, history, prev_state, policy_input):
    if params['exo_trade'] == 'pass' and params['exo_liq'] == 'pass':
        # there are no trades to be made this timestep
        return 'trade_random_direction', 'no_trade'
    else:
        direction_random_choice = random.choice(['test_q_for_r', 'test_r_for_q', 'test_r_for_r', 'test_add', 'test_remove'])
        # direction_random_choice = random.choice(['test_q_for_r', 'test_r_for_q', 'test_r_for_r'])

        return 'trade_random_direction', direction_random_choice

print("end of file: action_list.py")