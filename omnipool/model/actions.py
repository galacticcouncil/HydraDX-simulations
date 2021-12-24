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
        elif 'trade_types' in action_dict[x[0]]:
            action_d = action_dict[x[0]]['trade_types']
            n = 1
            if 'n' in action_dict[x[0]]:
                n = action_dict[x[0]]['n']
            action_list_x = tuple(random.choices(list(action_d.keys()), weights=list(action_d.values()), k=x[1])
                                  for i in range(n))
            action_list_x = zip(*action_list_x)
            action_list.extend(action_list_x)

        else:
            action_d = action_dict[x[0]]
            action_list_x = random.choices(list(action_d.keys()), weights=list(action_d.values()), k=x[1])
            action_list.extend(action_list_x)
    return action_list
