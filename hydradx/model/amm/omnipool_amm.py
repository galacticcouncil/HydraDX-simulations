import copy
import math
import string


def state_dict(
        token_list: list,
        r_values: list,
        q_values: list = None,
        p_values: list = None,
        b_values: list = None,
        s_values: list = None,
        omega_values: list = None,
        L: float = 0,
        C: float = math.inf,
        fee_assets: float = 0.0,
        fee_lrna: float = 0.0,
        preferred_stablecoin: str = 'USD'
) -> dict:
    assert 'HDX' in token_list, 'HDX not included in token list'
    assert len(r_values) == len(token_list), 'lengths of token_list and r_values do not match'
    # get initial value of T (total value locked)

    if not omega_values:
        omega_values = [1 for _ in range(len(token_list))]

    if not q_values:
        q_values = [r_values[i] * p_values[i] for i in range(len(token_list))]
    elif not p_values:
        p_values = [q_values[i] / r_values[i] for i in range(len(token_list))]
    else:
        assert False, 'Either LRNA quantities per pool or assets prices in LRNA must be specified.'

    if not b_values:
        b_values = [0] * len(token_list)

    if not s_values:
        s_values = copy.copy(r_values)

    stablecoin_index = token_list.index(preferred_stablecoin)
    t_values = [
        q_values[n] * r_values[stablecoin_index] / q_values[stablecoin_index]
        for n in range(len(token_list))
    ]

    assert len(r_values) == len(q_values) == len(b_values) == len(s_values) ==\
           len(t_values) == len(omega_values) == len(p_values)

    state = {
        'token_list': token_list,
        'R': r_values,  # Risk asset quantities
        'P': p_values,  # prices of risks assets denominated in LRNA
        'Q': q_values,  # LRNA quantities in each pool
        'B': b_values,  # quantity of shares in each asset owned by the protocol
        'S': s_values,  # quantity of LP shares in each pool
        'T': t_values,  # tvl per pool in usd
        'L': L,  # LRNA imbalance
        'C': C,  # TVL soft cap
        'O': omega_values,  # per-asset cap on what fraction of TVL can be stored
        'fee_assets': fee_assets,
        'fee_LRNA': fee_lrna,
        'preferred_stablecoin': preferred_stablecoin,
        'stablecoin_index': stablecoin_index
    }
    return state


def asset_invariant(state: dict, i: int) -> float:
    """Invariant for specific asset"""
    return state['R'][i] * state['Q'][i]


def swap_lrna_delta_Qi(old_state: dict, delta_Ri: float, i: int) -> float:
    return old_state['Q'][i] * (- delta_Ri / (old_state['R'][i] + delta_Ri))


def swap_lrna_delta_Ri(old_state: dict, delta_Qi: float, i: int) -> float:
    return old_state['R'][i] * (- delta_Qi / (old_state['Q'][i] + delta_Qi))


def weight_i(state: dict, i: int) -> float:
    return state['Q'][i] / sum(state['Q'])


def price_i(state: dict, i: int, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if state['R'][i] == 0:
        return 0
    else:
        return (state['Q'][i] / state['R'][i]) * (1 - fee)


def adjust_supply(old_state: dict):
    if old_state['H'] <= old_state['T']:
        return old_state

    over_supply = old_state['H'] - old_state['T']
    Q = sum(old_state['Q'])
    Q_burn = min(over_supply, old_state['burn_rate'] * Q)

    new_state = copy.deepcopy(old_state)
    for i in range(len(new_state['Q'])):
        new_state['Q'][i] += Q_burn * old_state['Q'][i] / Q

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
    state['A'] = [0] * n

    agent_shares = [sum([agent_d[agent_id]['s'][i] for agent_id in agent_d]) for i in range(n)]
    state['B'] = [state['S'][i] - agent_shares[i] for i in range(n)]

    state['T'] = init_d['T'] if 'T' in init_d else None
    state['H'] = init_d['H'] if 'H' in init_d else None

    return state


def initialize_pool_state(init_d=None, agent_d=None) -> dict:
    token_counts = initialize_token_counts(init_d)
    return initialize_shares(token_counts, init_d)


def swap_lrna(
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_Ra: float,
        delta_Qa: float,
        i: int,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    """Compute new state after LRNA swap"""

    if delta_Ra >= old_state['R'][i] * (1 - fee_assets):
        # insufficient assets in pool, transaction fails
        return old_state, old_agents

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_Qa < 0:
        delta_Q = -delta_Qa
        delta_R = old_state['R'][i] * -delta_Q / (delta_Q + old_state['Q'][i]) * (1 - fee_assets)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state['Q'][i] / (old_state['Q'][i] + delta_Q))
        delta_Ra = -delta_R
    elif delta_Ra > 0:
        delta_R = -delta_Ra
        delta_Q = old_state['Q'][i] * -delta_R / (old_state['R'][i] * (1 - fee_assets) + delta_R)
        delta_L = -delta_Q * (1 + (1 - fee_assets) * old_state['Q'][i] / (old_state['Q'][i] + delta_Q))
        delta_Qa = -delta_Q
    else:
        # print(f'Invalid swap (delta_Qa {delta_Qa}, delta_Ra {delta_Ra}')
        return old_state, old_agents

    if delta_Qa + old_agents[trader_id]['q'] < 0:
        # agent doesn't have enough lrna
        return old_state, old_agents
    elif delta_Ra + old_agents[trader_id]['r'][i] < 0:
        # agent doesn't have enough asset[i]
        return old_state, old_agents

    new_agents[trader_id]['q'] += delta_Qa
    new_agents[trader_id]['r'][i] += delta_Ra
    new_state['Q'][i] += delta_Q
    new_state['R'][i] += delta_R
    new_state['L'] += delta_L

    return new_state, new_agents


def swap_assets_direct(
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        delta_token: float,
        i_buy: int,
        i_sell: int,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    i = i_sell
    j = i_buy
    delta_Ri = delta_token
    assert delta_Ri > 0, 'sell amount must be greater than zero'

    delta_Qi = old_state['Q'][i] * -delta_Ri / (old_state['R'][i] + delta_Ri)
    delta_Qj = -delta_Qi * (1 - fee_lrna)
    delta_Rj = old_state['R'][j] * -delta_Qj / (old_state['Q'][j] + delta_Qj) * (1 - fee_assets)
    delta_L = min(-delta_Qi * fee_lrna, -old_state['L'])
    delta_QH = -fee_lrna * delta_Qi - delta_L

    new_state = copy.deepcopy(old_state)
    new_state['Q'][i] += delta_Qi
    new_state['Q'][j] += delta_Qj
    new_state['R'][i] += delta_Ri
    new_state['R'][j] += delta_Rj
    new_state['Q'][new_state['token_list'].index('HDX')] += delta_QH
    new_state['L'] += delta_L

    new_agents = copy.deepcopy(old_agents)
    new_agents[trader_id]['r'][i] -= delta_Ri
    new_agents[trader_id]['r'][j] -= delta_Rj

    return new_state, new_agents


def swap_assets(
        old_state: dict,
        old_agents: dict,
        trader_id: string,
        trade_type: string,
        delta_token: float,
        i_buy: int,
        i_sell: int,
        fee_assets: float = 0,
        fee_lrna: float = 0
) -> tuple:
    if trade_type == 'sell':

        new_state, new_agents = swap_assets_direct(
            old_state=old_state,
            old_agents=old_agents,
            trader_id=trader_id,
            delta_token=delta_token,
            i_buy=i_buy,
            i_sell=i_sell,
            fee_assets=fee_assets,
            fee_lrna=fee_lrna
        )

        return new_state, new_agents
    elif trade_type == 'buy':
        # back into correct delta_Ri, then execute sell
        delta_Qj = -old_state['Q'][i_buy] * delta_token / (old_state['R'][i_buy]*(1 - fee_assets) + delta_token)
        delta_Qi = -delta_Qj/(1 - fee_lrna)
        delta_Ri = -old_state['R'][i_sell] * delta_Qi / (old_state['Q'][i_sell] + delta_Qi)
        return swap_assets(old_state, old_agents, trader_id, 'sell', delta_Ri, i_buy, i_sell, fee_assets, fee_lrna)

    else:
        raise


def add_risk_liquidity(
        old_state: dict,
        old_agents: dict,
        LP_id: string,
        delta_R: float,
        i: int
) -> tuple:
    """Compute new state after liquidity addition"""

    assert delta_R > 0, "delta_R must be positive: " + str(delta_R)
    assert i >= 0, "invalid value for i: " + str(i)

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    # Token amounts update
    new_state['R'][i] += delta_R
    if LP_id:
        new_agents[LP_id]['r'][i] -= delta_R
        if new_agents[LP_id]['r'][i] < 0:
            # print('Transaction rejected because agent has insufficient funds.')
            # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
            return old_state, old_agents

    # Share update
    if new_state['S']:
        new_state['S'][i] *= new_state['R'][i] / old_state['R'][i]
    else:
        new_state['S'] = 1

    if LP_id:
        # shares go to provisioning agent
        new_agents[LP_id]['s'][i] += new_state['S'][i] - old_state['S'][i]
    else:
        # shares go to protocol
        new_state['B'] += new_state['S'][i] - old_state['S'][i]

    # LRNA add (mint)
    delta_Q = price_i(old_state, i) * delta_R
    new_state['Q'][i] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_R * old_state['Q'][i]/old_state['R'][i] * old_state['L']/sum(old_state['Q'])
    new_state['L'] += delta_L

    # T update: TVL soft cap
    stable_index = new_state['stablecoin_index']
    delta_t = new_state['Q'][i] * new_state['R'][stable_index]/new_state['Q'][stable_index] - new_state['T'][i]
    new_state['T'][i] += delta_t

    if 'O' in new_state and new_state['Q'][i] / sum(new_state['Q']) > new_state['O'][i]:
        # print(f'Transaction rejected because it would exceed the weight cap in pool[{i}].')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return old_state, old_agents

    if 'C' in new_state and sum(new_state['T']) > new_state['C']:
        # print('Transaction rejected because it would exceed the TVL cap.')
        # print(f'agent {LP_id}, asset {new_state["token_list"][i]}, amount {delta_R}')
        return old_state, old_agents

    # set price at which liquidity was added
    if LP_id:
        new_agents[LP_id]['p'][i] = price_i(new_state, i)

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

    new_state = copy.deepcopy(old_state)
    new_agents = copy.deepcopy(old_agents)

    if delta_S == 0:
        return new_state, new_agents

    piq = price_i(old_state, i)
    p0 = new_agents[LP_id]['p'][i]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_B = max(mult * delta_S, 0)
    new_state['B'][i] += delta_B
    new_state['S'][i] += delta_S + delta_B
    new_agents[LP_id]['s'][i] += delta_S

    # Token amounts update
    delta_R = old_state['R'][i] * max((delta_S + delta_B) / old_state['S'][i], -1)
    new_state['R'][i] += delta_R
    new_agents[LP_id]['r'][i] -= delta_R
    if piq >= p0:  # prevents rounding errors
        new_agents[LP_id]['q'] -= piq * (
                2 * piq / (piq + p0) * delta_S / old_state['S'][i] * old_state['R'][i] - delta_R)

    # LRNA burn
    delta_Q = price_i(old_state, i) * delta_R
    new_state['Q'][i] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = delta_R * old_state['Q'][i]/old_state['R'][i] * old_state['L']/sum(old_state['Q'])
    new_state['L'] += delta_L

    return new_state, new_agents
