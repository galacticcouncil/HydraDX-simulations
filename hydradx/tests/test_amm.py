import copy
import math

import pytest
from hypothesis import given, strategies as st, assume

from hydradx.model.amm import amm

# Token counts
tkn_ct_strat = st.floats(min_value=10000, max_value=10000000, allow_nan=False, allow_infinity=False)


def get_tkn_ct_strat(n):
    args = [tkn_ct_strat] * n
    return st.tuples(*args)

def get_state_from_strat(x, key_list):
    """x is a tuple, key_list is a list of keys each of which follow strategy x"""
    d = {}
    if isinstance(key_list, list):
        assert len(x[0]) == len(key_list)
        for j in range(len(key_list)):
            k = key_list[j]
            d[k] = [None] * len(x)
            for i in range(len(x)):
                d[k][i] = x[i][j]
    else:
        assert not (isinstance(x, tuple))
        k = key_list
        d[k] = copy.deepcopy(x)
    return d

# States with R, Q lists
QiRi_strat = get_tkn_ct_strat(2)
QR_strat = st.lists(QiRi_strat, min_size=2, max_size=5).map(lambda x: get_state_from_strat(x, ['Q', 'R']))

@given(QR_strat)
def test_swap(old_state) -> tuple:
    n = len(old_state['R'])
    old_state['token_list'] = ['DOT', 'DAI', 'HDX']
    old_state['fee_assets'] = 0
    old_state['fee_HDX'] = 0
    old_state['D'] = 0

    trader_id = 'trader'
    LP_id = 'LP'
    old_agents = {
        trader_id: {
            'r': [10000] * n,
            'q': 10000,
            's': [0] * n
        },
        LP_id: {
            'r': [0] * n,
            'q': 0,
            's': [900] * n
        }
    }

    delta_sell = 100
    trade_sell = {
        'token_sell': 'DOT',
        'amount_sell': delta_sell,
        'token_buy': 'DAI',
        'agent_id': 'trader'
    }

    sell_state, sell_agents = amm.swap(old_state, old_agents, trade_sell)
    i_sell = old_state['token_list'].index(trade_sell['token_sell'])
    assert sell_agents[trader_id]['r'][i_sell] - old_agents[trader_id]['r'][i_sell] == pytest.approx(-delta_sell)

    for j in range(len(old_state['R'])):
        assert old_state['R'][j] + old_agents[trader_id]['r'][j] == pytest.approx(sell_state['R'][j] + sell_agents[trader_id]['r'][j])

    i_buy = old_state['token_list'].index(trade_sell['token_buy'])
    trade_buy = copy.deepcopy(trade_sell)
    del trade_buy['amount_sell']
    trade_buy['amount_buy'] = sell_agents[trader_id]['r'][i_buy] - old_agents[trader_id]['r'][i_buy]
    buy_state, buy_agents = amm.swap(old_state, old_agents, trade_buy)

    for j in range(len(old_state['R'])):
        assert buy_state['R'][j] == pytest.approx(sell_state['R'][j])
        assert buy_state['Q'][j] == pytest.approx(sell_state['Q'][j])
        assert buy_agents[trader_id]['r'][j] == pytest.approx(sell_agents[trader_id]['r'][j])
    assert buy_agents[trader_id]['q'] == pytest.approx(sell_agents[trader_id]['q'])


if __name__ == '__main__':
    test_swap()
