import copy
import random

import pytest
from hypothesis import given, strategies as st
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm import amm

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=10000, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    lrna_price_usd = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA_price': 1 / lrna_price_usd
        }
    }
    return_dict.update({
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): {
            'liquidity': draw(asset_quantity_strategy),
            'LRNA': draw(asset_quantity_strategy)
        } for _ in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0
) -> oamm.OmnipoolState:
    asset_dict = asset_dict or draw(assets_config(token_count))
    return oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee
    )


@given(omnipool_config(token_count=3))
def test_swap(old_state: oamm.OmnipoolState):
    token_count = len(old_state.asset_list)
    # old_state['token_list'] = ['DOT', 'DAI', 'HDX']
    # old_state['fee_assets'] = 0
    # old_state['fee_LRNA'] = 0
    # old_state['L'] = 0

    trader_id = 'trader'
    LP_id = 'LP'
    old_agents = {
        trader_id: {
            'r': {token: 1000 for token in old_state.asset_list},
            'q': 1000,
            's': {token: 0 for token in old_state.asset_list}
        },
        LP_id: {
            'r': {token: 0 for token in old_state.asset_list},
            'q': 0,
            's': {token: 900 for token in old_state.asset_list}
        }
    }

    delta_sell = 100
    i_sell = old_state.asset_list[2]
    i_buy = old_state.asset_list[1]

    trade_sell = {
        'token_sell': i_sell,
        'amount_sell': delta_sell,
        'token_buy': i_buy,
        'agent_id': 'trader'
    }

    sell_state, sell_agents = amm.swap(old_state, old_agents, trade_sell)
    assert sell_agents[trader_id]['r'][i_sell] - old_agents[trader_id]['r'][i_sell] == pytest.approx(-delta_sell)

    for j in old_state.asset_list:
        assert old_state.liquidity[j] + old_agents[trader_id]['r'][j] == \
               pytest.approx(sell_state.liquidity[j] + sell_agents[trader_id]['r'][j])

    trade_buy = copy.deepcopy(trade_sell)
    del trade_buy['amount_sell']
    trade_buy['amount_buy'] = sell_agents[trader_id]['r'][i_buy] - old_agents[trader_id]['r'][i_buy]
    buy_state, buy_agents = amm.swap(old_state, old_agents, trade_buy)

    for j in old_state.asset_list:
        assert buy_state.liquidity[j] == pytest.approx(sell_state.liquidity[j])
        assert buy_state.lrna[j] == pytest.approx(sell_state.lrna[j])
        assert buy_agents[trader_id]['r'][j] == pytest.approx(sell_agents[trader_id]['r'][j])
    assert buy_agents[trader_id]['q'] == pytest.approx(sell_agents[trader_id]['q'])


if __name__ == '__main__':
    test_swap()
