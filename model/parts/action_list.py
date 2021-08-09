# Behaviors
# from hydra_multi_class.model.sys_params import ACTION_LIST
import numpy as np
import random
import math

# Set numpy random seed for replication
np.random.seed(42)

def actionDecoder(params, step, history, prev_state):
    '''
    In this simplified model of Uniswap, we have not modeled user behavior. Instead, we map events to actions. Depending on the input system parameters 'params' a given action sequence is induced.
    '''
    
    action = {
        'q_sold': 0,     # q to r swap
        'ri_sold': 0,     # r to q swap
        'ri_deposit': 0,   # add liq
        'q_deposit': 0,  # if symmetric add liq
        'Si_burn': 0,    # remove liq    
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


    action['asset_id'] = prev_state['asset_random_choice']
    action['q_sold'] = prev_state['trade_random_size'] * 2
    action['ri_sold'] = prev_state['trade_random_size']
    action['direction_q'] = prev_state['trade_random_direction']


    # print(params)
    
   # print('Hello, ' + os.getlogin() + '! How are you?')
    ############# CREATE AGENT ID's ################
    
    agent0_id = 0
    agent1_id = 1
    agent2_id = 2
    agent3_id = 3
    agent4_id = 4
    agent5_id = 5
    
    ############# CREATE AGENT ID's ################

   
    ############## SET RANDOM SEQUENCE ##################
    # print('getting here', params['exo_random_sequence'], timestep)
    if params['exo_random_sequence'] == 'on':
        # print('getting in', params['exo_random_sequence'], timestep)
        if timestep == 10:
            # print('getting in-for add', params['exo_random_sequence'], timestep)
            params['exo_liq'] = 'test_add'
            params['exo_trade'] = 'pass'
            #params['exo_trade'] = random.choice(['test_r_for_r'])
            action['asset_id'] = random.choice(['i'])
            action['purchased_asset_id'] = 'N/A'

        #elif timestep == 3:
           # removal = prev_state['uni_agents']['s_i'][agent2_id] - 150000
           # print("Assigned removal")
           # print(removal)
        elif timestep == 90:
            # print('getting in-for remove', params['exo_random_sequence'], timestep)
            params['exo_liq'] = 'test_remove'
            params['exo_trade'] = 'pass'
            action['asset_id'] = random.choice(['i'])
            action['purchased_asset_id'] = 'N/A'
    
        else:
            # print('skipping', params['exo_random_sequence'], timestep)
            #params['exo_liq'] = 'test_remove'
            #params['exo_trade'] = random.choice(['pass'])
            #params['exo_trade'] = random.choice(['test_q_for_r', 'test_r_for_q'])
            params['exo_trade'] = prev_state['trade_random_direction']
            #action['asset_id'] = random.choice(['i', 'j'])
            action['asset_id'] = prev_state['asset_random_choice']
            params['exo_liq'] = 'pass' #prev_state['trade_random_direction']
            # params['exo_liq'] = prev_state['trade_random_direction']
         
                    
    if params['exo_random_sequence'] == 'on':
        agent2_id = 2
    

  
    ############## CREATE RANDOM TRADE SEQUENCE ##############
    # print("I'll print firstrun variable:")
    # print(params['firstrun'])
    
    
    # if params['firstrun'] == 1:
    #     params['firstrun'] = 2
    #     trade_time = []
    #     trade_asset = [] 
    #     trade_size = []
    #     for k in range(0, 101):
    #         trade_time.append(k)
    #         trade_asset.append(random.choice(['i', 'j']))
    #         trade_size.append(math.ceil(np.random.normal(mu, sigma)))    
    #     print(timestep, trade_time, trade_size, trade_asset)
    #     print(timestep)
    #     print(trade_time[2])
    #     print(trade_size[2])
    #     print(trade_asset[2])
    
    # print("I'll print firstrun variable:")
    # print(params['firstrun'])
    #print(trade_time[timestep], trade_size[timestep], trade_asset[timestep])
    
    ############# CREATE RANDOM TRADE SEQUENCE ############
    #print("Actions at timestep are:")#
    #print(action)
    #print(agent2_id, agent3_id)
    
    ############ CHOOSE COMPOSITE ACTION TYPE #############################
    ### WILL USE A PARAM TO CHOOSE COMPOSITE AND ASSET TYPE TRANSACTIONS 
    
    # if params['exo_composite'] == 'alternating':
    #if timestep % len(ACTION_LIST) == 0:
            #params['exo_trade'] = ACTION_LIST[0] # automate this
            # params['exo_trade'] = 'test_r_for_q' # automate this
            #params['exo_liq'] = ACTION_LIST[0]
            # params['exo_liq'] = 'test_remove'
            # params['exo_trade'] = 'pass'
            # params['exo_trade'] = 'test_r_for_r' # automate this
            # params['exo_trade'] = 'test_r_for_r' # automate this
    #elif timestep % len(ACTION_LIST)  == 1:
            #params['exo_trade'] = ACTION_LIST[1] # automate this
            # params['exo_trade'] = 'test_r_for_q' # automate this
            #params['exo_liq'] = ACTION_LIST[1]
            # params['exo_liq'] = 'test_remove'
    # elif timestep % len(ACTION_LIST)  == 2:
    #         params['exo_trade'] = ACTION_LIST[2] # automate this
    #         # params['exo_trade'] = 'test_r_for_q' # automate this
    #         params['exo_liq'] = ACTION_LIST[2]
    #         # params['exo_liq'] = 'test_remove'        
    # elif timestep % len(ACTION_LIST)  == 3:
    #         params['exo_trade'] = ACTION_LIST[3] # automate this
    #         # params['exo_trade'] = 'test_r_for_q' # automate this
    #         params['exo_liq'] = ACTION_LIST[3]
    #         # params['exo_liq'] = 'test_remove'    
    # elif timestep % len(ACTION_LIST)  == 4:
    #         params['exo_trade'] = ACTION_LIST[4] # automate this
    #         # params['exo_trade'] = 'test_r_for_q' # automate this
    #         params['exo_liq'] = ACTION_LIST[4]
    #         # params['exo_liq'] = 'test_remove'   

        #list_index = timestep % 3 + 1
       # params['exo_trade'] = ACTION_LIST[list_index]
        #params['exo_liq'] = ACTION_LIST[list_index]

        
 ########## TEMP TEST SELL Q FOR R ############
    
    ####### AGENT 0 ######################
    if params['exo_trade'] == 'test_q_for_r':
        action['action_id'] = 'Ri_Purchase'
        action['purchased_asset_id'] = 'i'
        P = pool.get_price(action['purchased_asset_id']) 
        action['q_sold'] = prev_state['trade_random_size'] * 2


        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][agent0_id]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][agent0_id]
            action['purchased_asset_id'] = 'j'
            P = pool.get_price(action['purchased_asset_id']) 
            action['q_sold'] = prev_state['trade_random_size'] * 2
    ###############################################

########## TEMP TEST SELL R FOR Q ############
    ####### AGENT 1 ######################
    if params['exo_trade'] == 'test_r_for_q':
        action['ri_sold'] = prev_state['trade_random_size']
        action['action_id'] = 'Q_Purchase'
        action['purchased_asset_id'] = 'q'
        P = pool.get_price('i') 
        action['ri_sold'] = prev_state['trade_random_size'] 
        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][agent1_id]
        if action['asset_id'] == 'j':
            P = pool.get_price(action['asset_id']) 
            # print('ACTION LIST PRICE ===== ', P)
            action['agent_id'] = prev_state['uni_agents']['m'][agent1_id] 
            action['ri_sold'] = prev_state['trade_random_size'] 
            action['purchased_asset_id'] = 'q'

    ###############################################

    ########## TEMP TEST ADD LIQ ############
    ####### AGENT 2 ######################
    if params['exo_liq'] == 'test_add':
        action['ri_deposit'] = 5000
        action['action_id'] = 'AddLiquidity'
        action['purchased_asset_id'] = 'N/A'

        # temp choose first agent
        if timestep == 10:
            action['ri_deposit'] = 50000

            action['agent_id'] = prev_state['uni_agents']['m'][agent2_id]
        else:
            action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]

        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]
            action['ri_deposit'] = 5000
            action['purchased_asset_id'] = 'N/A'

    ###############################################

    ########## TEMP TEST REMOVE LIQ ############
    ####### AGENT 3 ######################
    if params['exo_liq'] == 'test_remove':
        print(prev_state['hydra_agents']['s_i'][agent2_id])
        #print(removal)
        # action['UNI_burn'] = prev_state['uni_agents']['s_i'][agent4_id] #* 0.001

        action['purchased_asset_id'] = 'N/A'

        action['action_id'] = 'RemoveLiquidity'
        if timestep == 90:
            print('removing at timestep 90: ', prev_state['hydra_agents']['s_i'][agent2_id])
            action['agent_id'] = prev_state['hydra_agents']['m'][agent2_id]
            action['UNI_burn'] = prev_state['hydra_agents']['s_i'][agent2_id]  # starting value subtract - 150000
            #action['UNI_burn'] = 199433.56 -150000 #a=0.5
            #action['UNI_burn'] = 201370.96 - 150000 #a=1.0
            # action['UNI_burn'] = 816230.51 - 150000 #a=1.5
            ######## Something is pretty strange here ########################### 
            # mismatch between shares and Q removed
            # the approx 200,000 shares created, if removed result in negative Qs and Rs
            # which is the cause of math errors in Y upon removal
            # that math needs to be addressed
            # this small amount of shares represents the amount close to the amount of Q added, then removed
            # not THE ANSWER.  
            # action['UNI_burn'] = 550 #a=1.5

        # else:
        #     action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]

        if action['asset_id'] == 'j':
            # print('remove j',step,action['asset_id'])
            action['agent_id'] = prev_state['uni_agents']['m'][agent4_id]
            action['UNI_burn'] = prev_state['uni_agents']['s_i'][agent4_id] #* 0.001
            action['purchased_asset_id'] = 'N/A'

    ###############################################

    ########## TEMP TEST SELL R FOR R ############
    ####### AGENT 5 ######################
    if params['exo_trade'] == 'test_r_for_r':
        # print("I want to trade:")
        action['ri_sold'] = prev_state['trade_random_size']
        action['action_id'] = 'R_Swap'
        action['purchased_asset_id'] = 'j'
        action['direction'] = 'ij'

        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][agent5_id]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][agent5_id]
            action['ri_sold'] = prev_state['trade_random_size']
            action['purchased_asset_id'] = 'i'
            action['direction'] = 'ji'           

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
    
def s_trade_deterministic(params, step, history, prev_state, policy_input):
    return 'trade_random_size', prev_state['trade_random_size']
    
def s_direction_random(params, step, history, prev_state, policy_input):
    if params['exo_trade'] == 'pass' and params['exo_liq'] == 'pass':
        # there are no trades to be made this timestep
        return 'trade_random_direction', 'no_trade'
    else:
        # direction_random_choice = random.choice(['test_q_for_r', 'test_r_for_q', 'test_r_for_r', 'test_add', 'test_remove'])
        # direction_random_choice = random.choice(['test_q_for_r', 'test_r_for_q', 'test_r_for_r'])
        direction_random_choice = random.choice(['test_r_for_r'])

        return 'trade_random_direction', direction_random_choice
