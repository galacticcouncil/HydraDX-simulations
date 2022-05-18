import copy
import string

from ..amm import omnipool_amm as oamm


# def initialize_state(init_d: dict, token_list: list, agents_d: dict = None) -> dict:
#     # initialize tokens
#     tokens_state = oamm.initialize_token_counts(init_d)  # shares will be wrong here, but it doesn't matter
#     # initialize LPs
#     if agents_d is not None:
#         converted_agents_d = convert_agents(tokens_state, token_list, agents_d)
#     else:
#         converted_agents_d = None
#     # initialize AMM shares
#     state = oamm.initialize_shares(tokens_state, init_d, converted_agents_d)  # shares will be correct here
#     return state


def swap(old_state: oamm.MarketState, old_agents: dict, trade: dict) -> tuple:
    """Translates from user-friendly trade API to internal API

    swap['token_buy'] is the token being bought
    swap['tokens_sell'] is the list of tokens being sold
    swap['token_sell'] is the token being sold
    swap['amount_sell'] is the amount of the token being sold
    """
    assert trade['token_buy'] != trade['token_sell'], "Cannot trade a token for itself"
    i_buy = trade['token_buy']
    i_sell = trade['token_sell']

    if 'amount_sell' in trade:
        trade_type = 'sell'
        if trade['token_sell'] == 'LRNA':
            delta_Q = trade['amount_sell']
            delta_R = 0
        else:
            delta_Q = 0
            delta_R = trade['amount_sell']
    elif 'amount_buy' in trade:
        trade_type = 'buy'
        if trade['token_buy'] == 'LRNA':
            delta_Q = -trade['amount_buy']
            delta_R = 0
        else:
            delta_Q = 0
            delta_R = -trade['amount_buy']
    else:
        raise

    if i_buy == 'LRNA' or i_sell == 'LRNA':
        return oamm.swap_lrna(
            old_state=old_state,
            old_agents=old_agents,
            trader_id=trade['agent_id'],
            delta_ra=delta_R,
            delta_qa=delta_Q,
            i=i_buy if i_sell == 'LRNA' else i_sell,
            fee_assets=old_state.asset_fee,
            fee_lrna=old_state.lrna_fee)
    elif trade_type == 'sell':
        return oamm.swap_assets(old_state, old_agents, trade['agent_id'], trade_type, trade['amount_sell'],
                                i_buy, i_sell,
                                old_state.asset_fee, old_state.lrna_fee)

    elif trade_type == 'buy':
        return oamm.swap_assets(old_state, old_agents, trade['agent_id'], trade_type, -trade['amount_buy'],
                                i_buy, i_sell,
                                old_state.asset_fee, old_state.lrna_fee)

    else:
        raise


def price_i(state: oamm.MarketState, i: str) -> float:
    return oamm.price_i(state, i)


def remove_liquidity(old_state: oamm.MarketState, old_agents: dict, transaction: dict) -> tuple:
    assert transaction['token_remove'] in old_state.asset_list
    agent_id = transaction['agent_id']
    shares_burn = transaction['shares_remove']
    i = transaction['token_remove']
    return oamm.remove_risk_liquidity(old_state, old_agents, agent_id, shares_burn, i)


def add_liquidity(old_state: oamm.MarketState, old_agents: dict, transaction: dict) -> tuple:
    assert transaction['token_add'] in old_state.asset_list
    agent_id = transaction['agent_id']
    amount_add = transaction['amount_add']
    i = transaction['token_add']
    return oamm.add_risk_liquidity(old_state, old_agents, agent_id, amount_add, i)


def value_assets(state: oamm.MarketState, assets: dict, prices: list = None) -> float:
    if prices is None:
        prices = [price_i(state, i) for i in state.asset_list]
    return assets['q'] + sum([assets['r'][i] * prices[i] for i in state.asset_list])


def withdraw_all_liquidity(state: oamm.MarketState, agent_d: dict, agent_id: string) -> tuple:
    token_count = len(state.asset_list)
    new_agents = {agent_id: agent_d}
    new_state = copy.deepcopy(state)

    for i in range(token_count):
        transaction = {
            'token_remove': new_state['token_list'][i],
            'agent_id': agent_id,
            'shares_remove': -agent_d['s'][i]
        }

        new_state, new_agents = remove_liquidity(new_state, new_agents, transaction)
    return new_state, new_agents


def value_holdings(state: oamm.MarketState, agent_d: dict, agent_id: string) -> float:
    prices = [price_i(state, i) for i in state.asset_list]
    new_state, new_agents = withdraw_all_liquidity(state, agent_d, agent_id)
    return value_assets(new_state, new_agents[agent_id], prices)


def convert_agent(state: oamm.MarketState, token_list: list, agent_dict: dict) -> dict:
    """Return agent dict compatible with this amm"""
    token_count = len(state.asset_list)
    d = {'q': 0, 's': [0] * token_count, 'r': [0] * token_count, 'p': [0] * token_count}

    # iterate through tokens held by AMM, look for both tokens and shares. Ignore the rest
    if 'LRNA' in agent_dict:
        d['q'] = agent_dict['LRNA']
    for i in range(token_count):
        if token_list[i] in agent_dict:
            d['r'][i] = agent_dict[token_list[i]]
        if 'omni' + token_list[i] in agent_dict:
            d['s'][i] = agent_dict['omni' + token_list[i]]
            # absent other information, assumes LPs contributed at current prices
            d['p'][i] = price_i(state, state.asset_list[i])

    return d


def convert_agents(state: oamm.MarketState, token_list: list, agents_dict: dict) -> dict:
    d = {}
    for agent_id in agents_dict:
        d[agent_id] = convert_agent(state, token_list, agents_dict[agent_id])
    return d
