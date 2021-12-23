import copy
import math
import string
add_log_swap=False
add_log_liqadd=False
add_log_liqrem=False

def asset_invariant(state: dict, i: int) -> float:
    """Invariant for specific asset"""
    global add_log_swap
    if add_log_swap: 
        equation = "inv=(state['R'][i]**(-state['a'][i]) + state['Q'][i]**(-state['a'][i]))**(-1/state['a'][i])"
        inv=(state['R'][i]**(-state['a'][i]) + state['Q'][i]**(-state['a'][i]))**(-1/state['a'][i])
        print(equation)
        equation1 = equation.replace('inv', str(inv)).replace("state['Q'][i]", str(state['Q'][i])).replace("state['R'][i]", str(state['R'][i])).replace("state['a'][i]", str(state['a'][i]))
        print(equation1)
    return (state['R'][i]**(-state['a'][i]) + state['Q'][i]**(-state['a'][i]))**(-1/state['a'][i])
    #return state['R'][i] * state['Q'][i]


def swap_hdx_delta_Qi(old_state: dict, delta_Ri: float, i: int) -> float:
    global add_log_swap
    if add_log_swap: 
        print("swap_hdx_delta_Qi")
        print("old_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_state.items()) + "}")
        delta_Qi = (old_state['Q'][i]**(-old_state['a'][i]) + old_state['R'][i]**(-old_state['a'][i]) - (delta_Ri + old_state['R'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['Q'][i]
        equation = "delta_Qi=(old_state['Q'][i]**(-old_state['a'][i]) + old_state['R'][i]**(-old_state['a'][i]) - (delta_Ri + old_state['R'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['Q'][i]"
        print(equation)
        equation1 = equation.replace('delta_Qi', str(delta_Qi)).replace("old_state['Q'][i]", str(old_state['Q'][i])).replace("delta_Ri", str(delta_Ri)).replace("old_state['R'][i]", str(old_state['R'][i])).replace("old_state['a'][i]", str(old_state['a'][i]))
        print(equation1)
        print()    
    
    return (old_state['Q'][i]**(-old_state['a'][i]) + old_state['R'][i]**(-old_state['a'][i]) - (delta_Ri + old_state['R'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['Q'][i]
    #return old_state['Q'][i] * (- delta_Ri / (old_state['R'][i] + delta_Ri))


def swap_hdx_delta_Ri(old_state: dict, delta_Qi: float, i: int) -> float:
    global add_log_swap
    if add_log_swap: 
        print("swap_hdx_delta_Ri")
        print("old_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_state.items()) + "}")

        delta_Ri = (old_state['R'][i]**(-old_state['a'][i]) + old_state['Q'][i]**(-old_state['a'][i]) - (delta_Qi + old_state['Q'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['R'][i]
        equation = "delta_Ri = (old_state['R'][i]**(-old_state['a'][i]) + old_state['Q'][i]**(-old_state['a'][i]) - (delta_Qi + old_state['Q'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['R'][i]"
        print(equation)
        equation1 = equation.replace('delta_Qi', str(delta_Qi)).replace("old_state['Q'][i]", str(old_state['Q'][i])).replace("delta_Ri", str(delta_Ri)).replace("old_state['R'][i]", str(old_state['R'][i])).replace("old_state['a'][i]", str(old_state['a'][i]))
        print(equation1)
        print()     
    
    
    return (old_state['R'][i]**(-old_state['a'][i]) + old_state['Q'][i]**(-old_state['a'][i]) - (delta_Qi + old_state['Q'][i])**(-old_state['a'][i]))**(-1/old_state['a'][i]) - old_state['R'][i]
    #return old_state['R'][i] * (- delta_Qi / (old_state['Q'][i] + delta_Qi))


def weight_i(state: dict, i: int) -> float:
    return state['Q'][i] / sum(state['Q'])


def price_i(state: dict, i: int, fee: float = 0) -> float:
    """Price of i denominated in HDX"""
    if state['R'][i] == 0:
        return 0
    else:
        return (state['Q'][i] / state['R'][i]) * (1 - fee)


def adjust_supply(old_state: dict):

    if old_state['H'] <= old_state['T']:
        return old_state

    over_supply = old_state['H'] - old_state['T']
    Q = sum(old_state['Q'])
    Q_burn = min(over_supply, old_state['burn_rate']*Q)

    new_state = copy.deepcopy(old_state)
    for i in range(len(new_state['Q'])):
        new_state['Q'][i] += Q_burn * old_state['Q'][i]/Q

    new_state['H'] -= Q_burn

    return new_state


def initialize_token_counts(init_d=None) -> dict:
    if init_d is None:
        init_d = {}
    state = {
        'R': copy.deepcopy(init_d['R']),
        'Q': [init_d['P'][i] * init_d['R'][i] for i in range(len(init_d['R']))]
    }
    return state


def initialize_shares(token_counts, init_d=None, agent_d=None) -> dict:
    if agent_d is None:
        agent_d = {}
    if init_d is None:
        init_d = {}

    n = len(token_counts['R'])
    state = copy.deepcopy(token_counts)
    state['S'] = copy.deepcopy(state['R'])
    state['A'] = [0]*n

    agent_shares = [sum([agent_d[agent_id]['s'][i] for agent_id in agent_d]) for i in range(n)]
    state['B'] = [state['S'][i] - agent_shares[i] for i in range(n)]

    state['D'] = 0
    state['T'] = init_d['T'] if 'T' in init_d else None
    state['H'] = init_d['H'] if 'H' in init_d else None

    return state


def initialize_pool_state(init_d=None, agent_d=None) -> dict:
    token_counts = initialize_token_counts(init_d)
    return initialize_shares(token_counts, init_d)


def swap_hdx(
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_R: float,
        delta_Q: float,
        i: int,
        fee: float = 0
) -> tuple:
    """Compute new state after HDX swap"""

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_Q == 0 and delta_R != 0:
        delta_Q = swap_hdx_delta_Qi(old_state, delta_R, i)
    elif delta_R == 0 and delta_Q != 0:
        delta_R = swap_hdx_delta_Ri(old_state, delta_Q, i)
    else:
        return new_state, new_agents

    # Token amounts update
    # Fee is taken from the "out" leg
    if delta_Q < 0:
        new_state['R'][i] += delta_R
        new_state['Q'][i] += delta_Q
        new_agents[trader_id]['r'][i] -= delta_R
        new_agents[trader_id]['q'] -= delta_Q * (1 - fee)

        new_state['D'] -= delta_Q * fee
    else:
        new_state['R'][i] += delta_R
        new_state['Q'][i] += delta_Q
        new_agents[trader_id]['r'][i] -= delta_R * (1 - fee)
        new_agents[trader_id]['q'] -= delta_Q

        # distribute fees
        new_state['A'][i] -= delta_R * fee * (new_state['B'][i] / new_state['S'][i])
        for agent_id in new_agents:
            agent = new_agents[agent_id]
            agent['r'][i] -= delta_R * fee * (agent['s'][i] / new_state['S'][i])

    return new_state, new_agents


def swap_assets(
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_sell: float,
        i_buy: int,
        i_sell: int,
        fee_assets: float = 0,
        fee_HDX: float = 0
) -> tuple:
    # swap asset in for HDX
    first_state, first_agents = swap_hdx(old_state, old_agents, trader_id, delta_sell, 0, i_sell, fee_HDX)
    # swap HDX in for asset
    delta_Q = first_agents[trader_id]['q'] - old_agents[trader_id]['q']
    return swap_hdx(first_state, first_agents, trader_id, 0, delta_Q, i_buy, fee_assets)


def add_risk_liquidity(
        old_state: dict,
        old_agents: dict,
        LP_id: string,
        delta_R: float,
        i: int
) -> tuple:
    """Compute new state after liquidity addition"""
    global add_log_liqadd
    
    if add_log_liqadd: 
        print("add_risk_liquidity")
        print() 

        print("old_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_state.items()) + "}")
        print() 
        
        print("old_agents")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_agents.items()) + "}")
        print() 


    assert delta_R > 0, "delta_R must be positive: " + str(delta_R)
    assert i >= 0, "invalid value for i: " + str(i)

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    # Token amounts update
    new_state['R'][i] += delta_R
    new_agents[LP_id]['r'][i] -= delta_R

    # Share update
    new_state['S'][i] *= new_state['R'][i] / old_state['R'][i]
    new_agents[LP_id]['s'][i] += new_state['S'][i] - old_state['S'][i]

    # HDX add
    delta_Q = price_i(old_state, i) * delta_R
    new_state['Q'][i] += delta_Q

    # set price at which liquidity was added
    new_agents[LP_id]['p'][i] = price_i(new_state, i)

    if add_log_liqadd:        
        print("new_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in new_state.items()) + "}")
        print() 
        
        print("new_agents")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in new_agents.items()) + "}")
        print() 

        print("add_risk_liquidity finish")
        print() 

    
    return new_state, new_agents

def remove_risk_liquidity(
        old_state: dict,
        old_agents: dict,
        LP_id: string,
        delta_S: float,
        i: int
) -> tuple:
    """Compute new state after liquidity removal"""
    assert delta_S <= 0, "delta_S cannot be positive: " + str(delta_S)
    assert i >= 0, "invalid value for i: " + str(i)
    
    global add_log_liqrem
    
    if add_log_liqrem: 
        print("remove_risk_liquidity")
        print() 

        print("old_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_state.items()) + "}")
        print() 
        
        print("old_agents")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in old_agents.items()) + "}")
        print() 
    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_S == 0:
        return new_state, new_agents

    # Share update    
    new_state['S'][i] += delta_S 
    new_agents[LP_id]['s'][i] += delta_S

    # Token amounts update
    delta_R = old_state['R'][i] / old_state['S'][i] * delta_S 
    new_state['R'][i] += delta_R
    new_agents[LP_id]['r'][i] -= delta_R
    
    # HDX burn
    delta_Q = price_i(old_state, i) * delta_R
    new_state['Q'][i] += delta_Q
    
    if add_log_liqrem:        
        print("new_state")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in new_state.items()) + "}")
        print() 
        
        print("new_agents")
        print("i="+str(i))
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in new_agents.items()) + "}")
        print() 

        print("remove_risk_liquidity finish")
        print() 
    return new_state, new_agents