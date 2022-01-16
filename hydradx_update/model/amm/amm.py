import copy
from multiprocessing.sharedctypes import Value
import string

add_log_value_assets=False
add_log_value_holdings=False
add_log_withdraw_all_liquidity=False
add_log_remove_liquidity=False

with open(r"./select_model.txt") as f:
        contents = f.readlines()
        if contents[0].replace("\n", "")=="Model=Omnipool_reweighting":
            from ..amm import reweighting_amm as oamm  
        else:
            from ..amm import omnipool_amm as oamm

#from ..amm import omnipool_amm as oamm
#from ..amm import reweighting_amm as oamm

def initialize_LPs(state_d: dict, init_LPs: list) -> dict:
    agent_d = {}
    for i in range(len(init_LPs)):
        s = [0] * len(state_d['S'])
        s[i] = state_d['S'][i]
        agent_d[init_LPs[i]] = {
            's': s,  # simply assigning each LP all of an asset
            'h': 0  # no HDX shares to start
        }
    return agent_d


def initialize_state(init_d: dict, token_list: list, agents_d: dict = None) -> dict:
    #print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in init_d.items()) + "}")
    # initialize tokens
    tokens_state = oamm.initialize_token_counts(init_d)  # shares will be wrong here, but it doesn't matter
    
    #with open(r"./select_model.txt") as f:
    #    contents = f.readlines()
    #    if contents[0].replace("\n", "")=="Model=Omnipool_reweighting":
    #        tokens_state['a'] = init_d['a']
    
    #print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in tokens_state.items()) + "}")
    # initialize LPs
    if agents_d is not None:
        converted_agents_d = convert_agents(tokens_state, token_list, agents_d)
    else:
        converted_agents_d = None
    # initialize AMM shares
    state = oamm.initialize_shares(tokens_state, init_d, converted_agents_d)  # shares will be correct here
    return state


def swap(old_state: dict, old_agents: dict, trade: dict) -> tuple:
    """Translates from user-friendly trade API to internal API

    swap['token_buy'] is the token being bought
    swap['tokens_sell'] is the list of tokens being sold
    swap['token_sell'] is the token being sold
    swap['amount_sell'] is the amount of the token being sold
    """
    assert trade['token_buy'] != trade['token_sell'], "Cannot trade a token for itself"

    i_buy = -1
    i_sell = -1
    if trade['token_buy'] != 'HDX':
        i_buy = old_state['token_list'].index(trade['token_buy'])
    if trade['token_sell'] != 'HDX':
        i_sell = old_state['token_list'].index(trade['token_sell'])

    if trade['token_sell'] == 'HDX':
        delta_Q = trade['amount_sell']
        delta_R = 0
    else:
        delta_Q = 0
        delta_R = trade['amount_sell']

    if i_buy < 0 or i_sell < 0:
        return oamm.swap_hdx(old_state, old_agents, trade['agent_id'], delta_R, delta_Q, max(i_buy, i_sell),
                             old_state['fee_HDX'] + old_state['fee_assets'])
    else:
        return oamm.swap_assets(old_state, old_agents, trade['agent_id'], trade['amount_sell'], i_buy, i_sell,
                                old_state['fee_assets'], old_state['fee_HDX'])


def price_i(state: dict, i: int) -> float:
    return oamm.price_i(state, i)


def adjust_supply(state: dict) -> dict:
    return oamm.adjust_supply(state)


def remove_liquidity(old_state: dict, old_agents: dict, transaction: dict) -> tuple:
    global add_log_remove_liquidity
      
    
    assert transaction['token_remove'] in old_state['token_list']
    agent_id = transaction['agent_id']
    shares_burn = transaction['shares_remove']
    i = old_state['token_list'].index(transaction['token_remove'])

    if add_log_withdraw_all_liquidity: print("remove_liquidity agent_id: " + str(i))
    if add_log_withdraw_all_liquidity: print("remove_liquidity shares_burn: " + str(shares_burn))
    if add_log_withdraw_all_liquidity: print("remove_liquidity i: " + str(i))

    value = oamm.remove_risk_liquidity(old_state, old_agents, agent_id, shares_burn, i) 
    
    if add_log_withdraw_all_liquidity: print("remove_liquidity value: " + str(value))

    return value


def add_liquidity(old_state: dict, old_agents: dict, transaction: dict) -> tuple:
    assert transaction['token_add'] in old_state['token_list']
    agent_id = transaction['agent_id']
    amount_add = transaction['amount_add']
    i = old_state['token_list'].index(transaction['token_add'])
    return oamm.add_risk_liquidity(old_state, old_agents, agent_id, amount_add, i)


def value_assets(state: dict, assets: dict, prices: list = None) -> float:
    global add_log_value_assets
    if prices is None:
        prices = [price_i(state, i) for i in range(len(state['R']))] 
                        
    if add_log_value_assets == True:
        print("value_assets assets['q']: " + str(assets['q']))
        print("value_assets assets['r'][i]: " + str(assets['r'][:]))  
        print("value_assets prices[i]: " + str(prices))
        print("value_assets value = assets['q'] + sum([assets['r'][i] * prices[i] for i in range(len(state['R']))])")    
        value = assets['q'] + sum([assets['r'][i] * prices[i] for i in range(len(state['R']))]) 
        print("value_assets value: " + str(value))
    return assets['q'] + sum([assets['r'][i] * prices[i] for i in range(len(state['R']))])


def withdraw_all_liquidity(state: dict, agent_d: dict, agent_id: string) -> tuple:
    global add_log_withdraw_all_liquidity

    n = len(state['R'])
    new_agents = {agent_id: agent_d}
    new_state = copy.deepcopy(state)

    if add_log_withdraw_all_liquidity: print("withdraw_all_liquidity new_agents: " + str(new_agents))
    if add_log_withdraw_all_liquidity: print("withdraw_all_liquidity new_state: " + str(new_state))

    for i in range(n):
        transaction = {
            'token_remove': 'R' + str(i + 1),
            'agent_id': agent_id,
            'shares_remove': -agent_d['s'][i]
        }
        
        if add_log_withdraw_all_liquidity: print("withdraw_all_liquidity transaction: " + str(transaction))        
        
        new_state, new_agents = remove_liquidity(new_state, new_agents, transaction)
        if add_log_withdraw_all_liquidity: print("withdraw_all_liquidity new_agents: " + str(new_agents)) 
        if add_log_withdraw_all_liquidity: print("withdraw_all_liquidity new_state: " + str(new_state)) 

    return new_state, new_agents


def value_holdings(state: dict, agent_d: dict, agent_id: string) -> float:
    global add_log_value_holdings

    prices = [price_i(state, i) for i in range(len(state['R']))]

    if add_log_value_holdings: print("value_holdings prices: "+ str(prices))

    if add_log_value_holdings: print("value_holdings state: " + str(state)) 
    if add_log_value_holdings: print("value_holdings agent_d: " + str(agent_d)) 
    if add_log_value_holdings: print("value_holdings agent_id: " + str(agent_id)) 

    new_state, new_agents = withdraw_all_liquidity(state, agent_d, agent_id)
    
    if add_log_value_holdings: print("value_holdings new_state: " + str(new_state)) 
    if add_log_value_holdings: print("value_holdings new_agents: " + str(new_agents)) 

    value = value_assets(new_state, new_agents[agent_id], prices)

    if add_log_value_holdings: print("value_holdings value: " + str(value)) 

    return value


def convert_agent(state: dict, token_list: list, agent_dict: dict) -> dict:
    """Return agent dict compatible with this amm"""
    n = len(state['R'])
    d = {'q': 0, 's': [0] * n, 'r': [0] * n, 'p': [0] * n}

    # iterate through tokens held by AMM, look for both tokens and shares. Ignore the rest
    if 'HDX' in agent_dict:
        d['q'] = agent_dict['HDX']
    for i in range(n):
        if token_list[i] in agent_dict:
            d['r'][i] = agent_dict[token_list[i]]
        if 'omni' + token_list[i] in agent_dict:
            d['s'][i] = agent_dict['omni' + token_list[i]]
            # absent other information, assumes LPs contributed at current prices
            d['p'][i] = price_i(state, i)

    return d


def convert_agents(state: dict, token_list: list, agents_dict: dict) -> dict:
    d = {}
    for agent_id in agents_dict:
        d[agent_id] = convert_agent(state, token_list, agents_dict[agent_id])
    return d
