import copy
import pytest
import random
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import basilisk_amm as bamm
from hydradx.model.amm.amm import agent_dict

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_quantity_strategy = st.floats(min_value=1000, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def assets_config(draw) -> dict:
    token_count = 2
    return_dict = {
        ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3)): draw(asset_quantity_strategy)
        for _ in range(token_count)
    }
    return return_dict


@st.composite
def basilisk_config(
        draw,
        asset_dict=None,
        trade_fee=None,
) -> bamm.BasiliskPoolState:
    asset_dict = asset_dict or draw(assets_config())
    return bamm.BasiliskPoolState(
        tokens=asset_dict,
        trade_fee=draw(st.floats(min_value=0, max_value=0.1)) if trade_fee is None else trade_fee,
    )


@given(basilisk_config())
def test_basilisk_construction(initial_state):
    assert isinstance(initial_state, bamm.BasiliskPoolState)


@given(basilisk_config())
def test_swap(initial_state: bamm.BasiliskPoolState):
    old_state = initial_state
    trader = {
        'r': {token: 100 for token in initial_state.asset_list}
    }
    old_agents = {
        'trader': trader
    }
    tkn_sell = initial_state.asset_list[0]
    tkn_buy = initial_state.asset_list[1]
    new_state, new_agents = bamm.swap(
        old_state,
        old_agents,
        'trader',
        sell_quantity=100,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy
    )
    if ((old_agents['trader']['r'][tkn_buy] + old_agents['trader']['r'][tkn_sell]
         + old_state.liquidity[tkn_buy] + old_state.liquidity[tkn_sell])
            != pytest.approx(new_agents['trader']['r'][tkn_buy] + new_agents['trader']['r'][tkn_sell]
                             + new_state.liquidity[tkn_buy] + new_state.liquidity[tkn_sell])):
        raise AssertionError('Asset quantity is not constant after swap!')


@given(basilisk_config(trade_fee=0))
def test_swap_pool_invariant(initial_state: bamm.BasiliskPoolState):
    old_state = initial_state
    trader = {
        'r': {token: 100 for token in initial_state.asset_list}
    }
    old_agents = {
        'trader': trader
    }
    tkn_sell = initial_state.asset_list[0]
    tkn_buy = initial_state.asset_list[1]
    new_state, new_agents = bamm.swap(
        old_state,
        old_agents,
        'trader',
        sell_quantity=100,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy
    )
    if ((old_state.liquidity[tkn_buy] * old_state.liquidity[tkn_sell])
            != pytest.approx(new_state.liquidity[tkn_buy] * new_state.liquidity[tkn_sell])):
        raise AssertionError('Pool invariant has varied.')


@given(basilisk_config(trade_fee=0), asset_quantity_strategy)
def test_add_remove_liquidity(initial_state: bamm.BasiliskPoolState, delta_token: float):
    old_state = initial_state
    old_agents = {
        'lp': agent_dict(r_values={token: 1000000 for token in initial_state.asset_list})
    }
    tkn_add = initial_state.asset_list[0]
    other_tkn = initial_state.asset_list[1]
    new_state, new_agents = bamm.add_risk_liquidity(
        old_state, old_agents,
        lp_id='lp',
        quantity=delta_token,
        tkn_add=tkn_add
    )
    if (old_state.liquidity[tkn_add] / old_state.liquidity[other_tkn]
            != pytest.approx(new_state.liquidity[tkn_add] / new_state.liquidity[other_tkn])):
        raise AssertionError('Asset ratios not constant after liquidity add!')

    if ((old_agents['lp']['r'][tkn_add] + old_agents['lp']['r'][other_tkn]
         + old_state.liquidity[tkn_add] + old_state.liquidity[other_tkn])
            != pytest.approx(new_agents['lp']['r'][tkn_add] + new_agents['lp']['r'][other_tkn]
                             + new_state.liquidity[tkn_add] + new_state.liquidity[other_tkn])):
        raise AssertionError('Asset quantity is not constant after liquidity add!')

    # if that transaction was successful, see if we can reverse it using remove_liquidity
    if not new_state.fail:
        revert_state, revert_agents = bamm.remove_liquidity(
            new_state, new_agents,
            lp_id='lp',
            quantity=new_agents['lp']['s'][new_state.unique_id],
            tkn_remove=tkn_add
        )
        if (
            revert_state.liquidity[tkn_add] != pytest.approx(old_state.liquidity[tkn_add])
            or revert_state.liquidity[other_tkn] != pytest.approx(old_state.liquidity[other_tkn])
            or revert_state.shares != pytest.approx(old_state.shares)
            or revert_agents['lp']['r'][tkn_add] != pytest.approx(revert_agents['lp']['r'][tkn_add])
            or revert_agents['lp']['r'][other_tkn] != pytest.approx(revert_agents['lp']['r'][other_tkn])
            or revert_agents['lp']['s'][old_state.unique_id] != pytest.approx(revert_agents['lp']['s'][old_state.unique_id])
        ):
            raise AssertionError('Withdrawal failed to return to previous state.')


if __name__ == '__main__':
    test_basilisk_construction()
    test_swap()
    test_swap_pool_invariant()
    test_add_remove_liquidity()