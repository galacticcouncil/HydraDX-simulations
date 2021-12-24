import copy
import string

import ipdb

from ..amm import omnipool_amm as oamm


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
    # initialize tokens
    tokens_state = oamm.initialize_token_counts(init_d)  # shares will be wrong here, but it doesn't matter
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
        return oamm.swap_hdx(old_state, old_agents, trade['agent_id'], delta_R, delta_Q, max(i_buy, i_sell))
        #return oamm.swap_hdx(old_state, old_agents, trade['agent_id'], delta_R, delta_Q, max(i_buy, i_sell),
                             #old_state['fee_HDX'] + old_state['fee_assets'])
    else:
        raise
        #return oamm.swap_assets(old_state, old_agents, trade['agent_id'], trade['amount_sell'], i_buy, i_sell,
                                #old_state['fee_assets'], old_state['fee_HDX'])


def price_i(state: dict, i: int) -> float:
    return oamm.price_i(state, i)


def adjust_supply(state: dict) -> dict:
    return oamm.adjust_supply(state)


def remove_liquidity(old_state: dict, old_agents: dict, transaction: dict) -> tuple:
    assert transaction['token_remove'] in old_state['token_list']
    agent_id = transaction['agent_id']
    shares_burn = transaction['shares_remove']
    i = old_state['token_list'].index(transaction['token_remove'])
    return oamm.remove_risk_liquidity(old_state, old_agents, agent_id, shares_burn, i)


def add_liquidity(old_state: dict, old_agents: dict, transaction: dict) -> tuple:
    assert transaction['token_add'] in old_state['token_list']
    agent_id = transaction['agent_id']
    amount_add = transaction['amount_add']
    i = old_state['token_list'].index(transaction['token_add'])
    return oamm.add_risk_liquidity(old_state, old_agents, agent_id, amount_add, i)


def value_assets(state: dict, assets: dict, prices: list = None) -> float:
    if prices is None:
        prices = [price_i(state, i) for i in range(len(state['R']))]
    return assets['q'] + sum([assets['r'][i] * prices[i] for i in range(len(state['R']))])


def withdraw_all_liquidity(state: dict, agent_d: dict, agent_id: string) -> tuple:
    n = len(state['R'])
    new_agents = {agent_id: agent_d}
    new_state = copy.deepcopy(state)

    for i in range(n):
        transaction = {
            'token_remove': 'R' + str(i + 1),
            'agent_id': agent_id,
            'shares_remove': -agent_d['s'][i]
        }

        new_state, new_agents = remove_liquidity(new_state, new_agents, transaction)
    return new_state, new_agents


def process_transactions(state: dict, agent_d: dict, policy_inputs: list) -> tuple:
    batch_trade_d = {}  # keys will be (asset, tkn specified)
                        # values will be net amt specified with positive values indicating
                        # that traders want to buy from pool
    new_state = copy.deepcopy(state)
    protocol_agents = {}
    n = len(state['R'])

    # Accumulate net trades
    for trade in policy_inputs:
        # break up trades into LHDX trades
        # batch for each TKN
        if trade['token_sell'] == 'HDX':
            asset = trade['token_buy']
            if 'amount_buy' in trade:
                tkn_spec = asset
                other_tkn = 'HDX'
                amt_spec = trade['amount_buy']
            elif 'amount_sell' in trade:
                tkn_spec = 'HDX'
                other_tkn = asset
                amt_spec = -trade['amount_sell']
            else:
                raise
        elif trade['token_buy'] == 'HDX':
            asset = trade['token_sell']
            if 'amount_sell' in trade:
                tkn_spec = asset
                other_tkn = 'HDX'
                amt_spec = -trade['amount_sell']
            elif 'amount_buy' in trade:
                tkn_spec = 'HDX'
                other_tkn = asset
                amt_spec = trade['amount_buy']
            else:
                raise
        else:
            raise

        if (asset, tkn_spec, other_tkn) not in batch_trade_d:
            batch_trade_d[(asset, tkn_spec, other_tkn)] = 0

        batch_trade_d[(asset, tkn_spec, other_tkn)] += amt_spec

        # prepare a different "agent" for interacting with each TKN/HDX AMM
        protocol_agents[asset] = {'q': 0, 's': [0] * n, 'r': [0] * n, 'p': [0] * n}

    # Construct net trades
    net_trades = {}
    for (asset, tkn_spec, other_tkn) in batch_trade_d:
        amt_spec = batch_trade_d[(asset, tkn_spec, other_tkn)]
        if amt_spec < 0:  # traders on net want to sell token to pool
            trade = {
                'token_sell': tkn_spec,
                'token_buy': other_tkn,
                'amount_sell': -amt_spec,
                'action_id': 'Trade',
                'agent_id': asset
            }
        else:  # traders on net want to buy token from pool
            # need to include the amt_spec = 0 case here because this may change when slippage limits are applied
            trade = {
                'token_buy': tkn_spec,
                'token_sell': other_tkn,
                'amount_buy': amt_spec,
                'action_id': 'Trade',
                'agent_id': asset
            }
        if asset not in net_trades:
            net_trades[asset] = {}
        net_trades[asset][tkn_spec] = trade

        # Execute net trades against AMM (without fee)
        #new_state, protocol_agents = swap(new_state, protocol_agents, trade)

    clearing_prices = {}
    for asset in protocol_agents:
        assert asset in net_trades
        i = new_state['token_list'].index(asset)
        #delta_r = protocol_agents[asset]['r'][i]
        #delta_q = protocol_agents[asset]['q']
        delta_q_hdx = 0
        delta_r_asset = 0
        if 'HDX' in net_trades[asset]:
            trade = net_trades[asset]['HDX']
            delta_q_hdx += trade['amount_sell'] if 'amount_sell' in trade else trade['amount_buy']
        if asset in net_trades[asset]:
            trade = net_trades[asset][asset]
            delta_r_asset += trade['amount_sell'] if 'amount_sell' in trade else trade['amount_buy']

        if delta_r_asset == 0:
            clearing_prices[asset] = None
        else:
            clearing_prices[asset] = (new_state['Q'][i] + delta_q_hdx)/(new_state['R'][i] + delta_r_asset)

    new_agents = copy.deepcopy(agent_d)

    # process trades with agents
    for trade in policy_inputs:
        if 'amount_sell' in trade:
            if trade['token_sell'] == 'HDX':
                asset = trade['token_buy']
                if clearing_prices[asset] is not None:
                    i = new_state['token_list'].index(asset)
                    agent = new_agents[trade['agent_id']]
                    agent['q'] -= trade['amount_sell']
                    agent['r'][i] += trade['amount_sell']/clearing_prices[asset]
                    new_state['Q'][i] += trade['amount_sell']
                    new_state['R'][i] -= trade['amount_sell']/clearing_prices[asset]
            else:
                asset = trade['token_sell']
                if clearing_prices[asset] is not None:
                    i = new_state['token_list'].index(asset)
                    agent = new_agents[trade['agent_id']]
                    agent['r'][i] -= trade['amount_sell']
                    agent['q'] += trade['amount_sell'] * clearing_prices[asset]
                    new_state['R'][i] += trade['amount_sell']
                    new_state['Q'][i] -= trade['amount_sell'] * clearing_prices[asset]
        elif 'amount_buy' in trade:
            if trade['token_buy'] == 'HDX':
                asset = trade['token_sell']
                if clearing_prices[asset] is not None:
                    i = new_state['token_list'].index(asset)
                    agent = new_agents[trade['agent_id']]
                    agent['q'] += trade['amount_buy']
                    agent['r'][i] -= trade['amount_buy']/clearing_prices[asset]
                    new_state['Q'][i] -= trade['amount_buy']
                    new_state['R'][i] += trade['amount_buy']/clearing_prices[asset]
            else:
                asset = trade['token_buy']
                if clearing_prices[asset] is not None:
                    i = new_state['token_list'].index(asset)
                    agent = new_agents[trade['agent_id']]
                    agent['r'][i] += trade['amount_buy']
                    agent['q'] -= trade['amount_buy'] * clearing_prices[asset]
                    new_state['R'][i] -= trade['amount_buy']
                    new_state['Q'][i] += trade['amount_buy'] * clearing_prices[asset]
        else:
            raise

    return new_state, new_agents


def value_holdings(state: dict, agent_d: dict, agent_id: string) -> float:
    prices = [price_i(state, i) for i in range(len(state['R']))]
    new_state, new_agents = withdraw_all_liquidity(state, agent_d, agent_id)
    return value_assets(new_state, new_agents[agent_id], prices)


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
