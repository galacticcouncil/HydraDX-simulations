import random

'''
def get_trade(r_sold, q_sold, i_buying, agent_id):
    return {
        'r_sold': r_sold,
        'q_sold': q_sold,
        'i_buying': i_buying,
        'agent_id': agent_id,
        'action_id': 'Trade'
    }

def get_arb_market(settle_asset, agent_id):
    return {
        'settle_asset': settle_asset,
        'agent_id': agent_id,
        'action_id': 'ArbMarket'
    }

def get_liq_add(r_deposit, agent_id):
    return {
        'r_deposit': r_deposit,
        'agent_id': agent_id,
        'action_id': 'AddLiquidity'
    }

def get_liq_rem(s_burn, agent_id):
    return {
        's_burn': s_burn,
        'agent_id': agent_id,
        'action_id': 'RemoveLiquidity'
    }

def get_liq_rem_percentage(r_percent, agent_id):
    return {
        'r_percent': r_percent,
        'agent_id': agent_id,
        'action_id': 'RemoveLiquidityPercent'
    }
    
'''


def get_action_list(init_list, action_dict, seed=42):
    # getting agent paths here standardizes the agent actions across different tests
    random.seed(seed)
    action_list = []
    for x in init_list:
        if x[0] not in action_dict:
            action_list.extend([x[0]] * x[1])
        #elif len(x) == 100:
        #    action_list_x = 'provide_r1_liq'
        #    action_list.extend(action_list_x)
        #elif len(x) == 4900:
        #    action_list_x = 'remove_r1_liq'
        #    action_list.extend(action_list_x)   
        else:
            action_d = action_dict[x[0]]
            action_list_x = random.choices(list(action_d.keys()), weights=list(action_d.values()), k=x[1])
            action_list.extend(action_list_x)
            #print(action_list)
    return action_list

def assign_liquidity_actions(init_list):
    init_list[100] = 'provide_r1_liq'
    init_list[4900] = 'remove_r1_liq'
    action_list = init_list
    print("The action at timestep 101 is:", action_list[100])
    print("The action at timestep 4901 is:", action_list[4900])
    print("The action list lenght is:", len(action_list))
    return action_list
