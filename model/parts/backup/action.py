# Behaviors
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
        'direction': str() 
    }
    


    timestep = prev_state['timestep']

    ############ CHOOSE ASSET TYPE #############################
    ####### PREVIOUS MULTIPLE SUBSTEP TO EXECUTE EACH INSTANCE ###########
    # if step == 1 or step == 2:
    #     action['asset_id'] = 'i'
    # if step == 3 or step == 4:
    #     action['asset_id'] = 'j'
    ############ CHOOSE ASSET TYPE #############################


    ############ CHOOSE ASSET TYPE #############################
    ### USE A PARAM TO CHOOSE COMPOSITE AND ASSET TYPE TRANSACTIONS 
    
    if params['exo_asset'] == 'alternating':
        if timestep % 2 == 0:
            action['asset_id'] = 'i'
        elif timestep % 2 == 1:
            action['asset_id'] = 'j'

    if params['exo_asset'] == 'i_only':
        action['asset_id'] = 'i'

    if params['exo_asset'] == 'j_only':
        action['asset_id'] = 'j'

    ############ CHOOSE ASSET TYPE #############################
    
    ############ CHOOSE COMPOSITE ACTION TYPE #############################
    ### WILL USE A PARAM TO CHOOSE COMPOSITE AND ASSET TYPE TRANSACTIONS 
    
    if params['exo_composite'] == 'alternating':
        if timestep % 2 == 0:
            params['exo_trade'] = 'test_q_for_r' # automate this
            # params['exo_trade'] = 'test_r_for_q' # automate this
            params['exo_liq'] = 'pass'
            # params['exo_liq'] = 'test_remove'
            # params['exo_trade'] = 'pass'
            # params['exo_trade'] = 'test_r_for_r' # automate this
            # params['exo_trade'] = 'test_r_for_r' # automate this



            # print(timestep, params['exo_trade'], action['asset_id'])
        elif timestep % 2 == 1:
            params['exo_liq'] = 'pass'
            # params['exo_liq'] = 'test_add'
            # params['exo_liq'] = 'test_remove'
            # params['exo_trade'] = 'pass'
            params['exo_trade'] = 'test_r_for_q' # automate this

            # params['exo_trade'] = 'test_r_for_r' # automate this
            
            # print(timestep, params['exo_liq'], action['asset_id'] )
    ############ CHOOSE COMPOSITE ACTION TYPE  #############################
    # print(timestep, params['exo_trade'])
    # print(timestep, params['exo_liq'])

    ########## TEMP TEST SELL Q FOR R ############
    ####### AGENT 0 ######################
    if params['exo_trade'] == 'test_q_for_r':
        action['q_sold'] = 100
        action['action_id'] = 'Ri_Purchase'
        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][0]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][4]
            action['q_sold'] = 20
    ###############################################

    ########## TEMP TEST SELL Q FOR R ############
    ####### AGENT 1 ######################
    if params['exo_trade'] == 'test_r_for_q':
        action['ri_sold'] = 1000
        action['action_id'] = 'Q_Purchase'
        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][1]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][5]
            action['ri_sold'] = 50
    ###############################################

    ########## TEMP TEST ADD LIQ ############
    ####### AGENT 2 ######################
    if params['exo_liq'] == 'test_add':
        action['ri_deposit'] = 1000
        action['action_id'] = 'AddLiquidity'
        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][2]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][6]
            action['ri_deposit'] = 51
    ###############################################

    ########## TEMP TEST REMOVE LIQ ############
    ####### AGENT 3 ######################
    if params['exo_liq'] == 'test_remove':
        action['UNI_burn'] = 10000000
        action['action_id'] = 'RemoveLiquidity'
        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][3]
        if action['asset_id'] == 'j':
            # print('remove j',step,action['asset_id'])
            action['agent_id'] = prev_state['uni_agents']['m'][7]
            action['UNI_burn'] = 500000
    ###############################################

    ########## TEMP TEST SELL R FOR R ############
    ####### AGENT 5 ######################
    if params['exo_trade'] == 'test_r_for_r':
        action['ri_sold'] = 1000
        action['action_id'] = 'R_Swap'
        action['purchased_asset_id'] = 'j'
        action['direction'] = 'ij'

        # temp choose first agent
        action['agent_id'] = prev_state['uni_agents']['m'][1]
        if action['asset_id'] == 'j':
            action['agent_id'] = prev_state['uni_agents']['m'][5]
            action['ri_sold'] = 500
            action['purchased_asset_id'] = 'i'
            action['direction'] = 'ji'


    ###############################################

    # print(step,action['asset_id'])
    # print(timestep, action)
    return action