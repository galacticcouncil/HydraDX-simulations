import copy
from ..amm import omnipool_amm as oamm


def swap(old_state: oamm.OmnipoolState, old_agents: dict, trade: dict) -> tuple[oamm.OmnipoolState, dict]:
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


def remove_liquidity(
        old_state: oamm.OmnipoolState,
        old_agents: dict, transaction: dict
        ) -> tuple[oamm.OmnipoolState, dict]:
    assert transaction['token_remove'] in old_state.asset_list
    agent_id = transaction['agent_id']
    shares_remove = transaction['shares_remove']
    i = transaction['token_remove']
    return oamm.remove_risk_liquidity(old_state, old_agents, agent_id, shares_remove, i)


def add_liquidity(
        old_state: oamm.OmnipoolState,
        old_agents: dict, transaction: dict
        ) -> tuple[oamm.OmnipoolState, dict]:
    assert transaction['token_add'] in old_state.asset_list
    agent_id = transaction['agent_id']
    amount_add = transaction['amount_add']
    i = transaction['token_add']
    return oamm.add_risk_liquidity(old_state, old_agents, agent_id, amount_add, i)


def value_assets(state: oamm.OmnipoolState, agent: dict) -> float:
    return agent['q'] + sum([
        agent['r'][i] * oamm.price_i(state, i)
        for i in state.asset_list
    ])


def withdraw_all_liquidity(state: oamm.OmnipoolState, agent: dict) -> tuple:
    new_agents = {1: agent}
    new_state = copy.deepcopy(state)

    for i in new_state.asset_list:
        transaction = {
            'token_remove': i,
            'agent_id': 1,
            'shares_remove': -agent['s'][i]
        }

        new_state, new_agents = remove_liquidity(new_state, new_agents, transaction)
    return new_state, new_agents[1]


def cash_out(state: oamm.OmnipoolState, agent: dict) -> float:
    new_state, new_agent = withdraw_all_liquidity(state, agent)
    return value_assets(new_state, new_agent)
