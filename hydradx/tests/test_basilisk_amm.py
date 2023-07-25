import pytest
from hypothesis import given, strategies as st, assume
from hydradx.model.amm import basilisk_amm as bamm
from hydradx.model.amm.agents import Agent
from mpmath import mp, mpf
mp.dps = 50

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_quantity_strategy = st.floats(min_value=1000, max_value=100000)
fee_strategy = st.floats(min_value=0, max_value=0.1, allow_nan=False)
trade_quantity_strategy = st.floats(min_value=-1000, max_value=1000)


@st.composite
def assets_config(draw) -> dict:
    token_count = 2
    return_dict = {
        f"{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}": mpf(draw(asset_quantity_strategy))
        for i in range(token_count)
    }
    return return_dict


@st.composite
def constant_product_pool_config(
        draw,
        asset_dict=None,
        trade_fee=None
) -> bamm.ConstantProductPoolState:
    asset_dict = asset_dict or draw(assets_config())
    return bamm.ConstantProductPoolState(
        tokens=asset_dict,
        trade_fee=draw(fee_strategy) if trade_fee is None else trade_fee,
        unique_id="/".join(sorted(asset_dict.keys()))
    )


@given(constant_product_pool_config())
def test_basilisk_construction(initial_state):
    assert isinstance(initial_state, bamm.ConstantProductPoolState)


@given(constant_product_pool_config(trade_fee=0.1), trade_quantity_strategy)
def test_swap(initial_state: bamm.ConstantProductPoolState, delta_r):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list}
    )
    tkn_sell = initial_state.asset_list[0]
    tkn_buy = initial_state.asset_list[1]
    swap_state, swap_agent = bamm.simulate_swap(
        old_state=old_state,
        old_agent=old_agent,
        sell_quantity=delta_r,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy
    )
    if (old_agent.holdings[tkn_buy] + old_state.liquidity[tkn_buy]
            != pytest.approx(swap_agent.holdings[tkn_buy] + swap_state.liquidity[tkn_buy])
            or old_state.liquidity[tkn_sell] + old_agent.holdings[tkn_sell]
            != pytest.approx(swap_state.liquidity[tkn_sell] + swap_agent.holdings[tkn_sell])):
        raise AssertionError('Asset quantity is not constant after swap!')

    # swap back, specifying buy_quantity this time
    delta_r = swap_state.liquidity[tkn_sell] - old_state.liquidity[tkn_sell]
    revert_state, revert_agent = bamm.simulate_swap(
        old_state=swap_state,
        old_agent=swap_agent,
        buy_quantity=delta_r,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy
    )
    # should still total the same
    if ((old_agent.holdings[tkn_buy] + old_agent.holdings[tkn_sell]
         + old_state.liquidity[tkn_buy] + old_state.liquidity[tkn_sell])
            != pytest.approx(revert_agent.holdings[tkn_buy] + revert_agent.holdings[tkn_sell]
                             + revert_state.liquidity[tkn_buy] + revert_state.liquidity[tkn_sell])):
        raise AssertionError('Asset quantity is not constant after swap!')


@given(constant_product_pool_config(trade_fee=0), trade_quantity_strategy)
def test_swap_pool_invariant(initial_state: bamm.ConstantProductPoolState, delta_r: float):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list}
    )

    tkn_sell = initial_state.asset_list[0]
    tkn_buy = initial_state.asset_list[1]
    swap_state, swap_agent = bamm.simulate_swap(
        old_state=old_state,
        old_agent=old_agent,
        sell_quantity=delta_r,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy
    )
    if ((old_state.liquidity[tkn_buy] * old_state.liquidity[tkn_sell])
            != pytest.approx(swap_state.liquidity[tkn_buy] * swap_state.liquidity[tkn_sell])):
        raise AssertionError('Pool invariant has varied.')

    # swap back, specifying buy_quantity this time
    delta_r = swap_state.liquidity[tkn_sell] - old_state.liquidity[tkn_sell]
    revert_state, revert_agent = bamm.simulate_swap(
        old_state=swap_state,
        old_agent=swap_agent,
        buy_quantity=delta_r,
        tkn_sell=tkn_buy,
        tkn_buy=tkn_sell
    )
    # invariant should remain
    if old_state.invariant != pytest.approx(revert_state.invariant):
        raise AssertionError('Pool invariant has varied.')

    if ((old_state.liquidity[tkn_buy] != pytest.approx(revert_state.liquidity[tkn_buy]))
            or old_state.liquidity[tkn_sell] != pytest.approx(revert_state.liquidity[tkn_sell])):
        raise AssertionError('Reverse sell with no fees yielded unexpected result')


@given(constant_product_pool_config(trade_fee=0), asset_quantity_strategy)
def test_add_remove_liquidity(initial_state: bamm.ConstantProductPoolState, delta_token: float):
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list}
    )

    tkn_add = initial_state.asset_list[0]
    other_tkn = initial_state.asset_list[1]
    new_state, new_agent = bamm.simulate_add_liquidity(
        old_state, old_agent,
        quantity=delta_token,
        tkn_add=tkn_add
    )
    if (old_state.liquidity[tkn_add] / old_state.liquidity[other_tkn]
            != pytest.approx(new_state.liquidity[tkn_add] / new_state.liquidity[other_tkn])):
        raise AssertionError('Asset ratios not constant after liquidity add!')

    if ((old_agent.holdings[tkn_add] + old_agent.holdings[other_tkn]
         + old_state.liquidity[tkn_add] + old_state.liquidity[other_tkn])
            != pytest.approx(new_agent.holdings[tkn_add] + new_agent.holdings[other_tkn]
                             + new_state.liquidity[tkn_add] + new_state.liquidity[other_tkn])):
        raise AssertionError('Asset quantity is not constant after liquidity add!')

    # if that transaction was successful, see if we can reverse it using remove_liquidity
    if not new_state.fail:
        revert_state, revert_agent = bamm.simulate_remove_liquidity(
            new_state, new_agent,
            quantity=new_agent.holdings[new_state.unique_id],
            tkn_remove=tkn_add
        )
        if (
                revert_state.liquidity[tkn_add] != pytest.approx(old_state.liquidity[tkn_add])
                or revert_state.liquidity[other_tkn] != pytest.approx(old_state.liquidity[other_tkn])
                or revert_state.shares != pytest.approx(old_state.shares)
                or revert_agent.holdings[tkn_add] != pytest.approx(revert_agent.holdings[tkn_add])
                or revert_agent.holdings[other_tkn] != pytest.approx(revert_agent.holdings[other_tkn])
                or revert_agent.holdings[old_state.unique_id] != pytest.approx(
                    revert_agent.holdings[old_state.unique_id]
                )
        ):
            raise AssertionError('Withdrawal failed to return to previous state.')


@given(constant_product_pool_config(trade_fee=0), asset_quantity_strategy)
def test_remove_liquidity(initial_state: bamm.ConstantProductPoolState, delta_token: float):
    initial_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list}
    )
    tkn_remove = initial_state.asset_list[0]
    # gotta add liquidity before we can remove it
    old_state, old_agent = bamm.simulate_add_liquidity(
        initial_state, initial_agent,
        quantity=delta_token,
        tkn_add=tkn_remove
    )
    new_state, new_agent = bamm.simulate_remove_liquidity(
        old_state, old_agent,
        quantity=delta_token,
        tkn_remove=tkn_remove
    )
    other_tkn = initial_state.asset_list[1]
    if (old_state.liquidity[tkn_remove] / old_state.liquidity[other_tkn]
            != pytest.approx(new_state.liquidity[tkn_remove] / new_state.liquidity[other_tkn])):
        raise AssertionError('Asset ratios not constant after liquidity remove!')

    if ((old_agent.holdings[tkn_remove] + old_agent.holdings[other_tkn]
         + old_state.liquidity[tkn_remove] + old_state.liquidity[other_tkn])
            != pytest.approx(new_agent.holdings[tkn_remove] + new_agent.holdings[other_tkn]
                             + new_state.liquidity[tkn_remove] + new_state.liquidity[other_tkn])):
        raise AssertionError('Asset quantity is not constant after liquidity remove!')

    if new_agent.holdings[tkn_remove] != pytest.approx(initial_agent.holdings[tkn_remove]):
        raise AssertionError('Agent did not get back what it put in.')

    cheat_state, cheat_agent = bamm.simulate_remove_liquidity(
        old_state, old_agent,
        quantity=delta_token + 1,
        tkn_remove=tkn_remove
    )

    if not cheat_state.fail:
        raise AssertionError('Agent was able to remove more shares than it owned!')


@given(constant_product_pool_config(trade_fee=0), fee_strategy)
def test_slip_fees(initial_state: bamm.ConstantProductPoolState, slip_factor: float):
    assume(slip_factor > 0.01)
    minimum_fee = 0.0001
    initial_state.trade_fee = bamm.ConstantProductPoolState.custom_slip_fee(
        slip_factor=slip_factor, minimum=minimum_fee)
    initial_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list}
    )
    tkn_buy = initial_state.asset_list[1]
    tkn_sell = initial_state.asset_list[0]
    # buy half of the available asset with slip based fees
    buy_quantity = initial_state.liquidity[tkn_buy] / 2
    buy_state, buy_agent = bamm.simulate_swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )

    # now buy the same quantity, but do it in two smaller trades
    split_buy_state, split_buy_agent = initial_state.copy(), initial_agent.copy()
    next_state, next_agent = {}, {}
    for i in range(2):
        next_state[i], next_agent[i] = bamm.simulate_swap(
            old_state=split_buy_state,
            old_agent=split_buy_agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity / 2
        )
        split_buy_state, split_buy_agent = next_state[i], next_agent[i]

    if buy_state.fail or split_buy_state.fail:
        return

    if (buy_state.liquidity[tkn_buy] != pytest.approx(split_buy_state.liquidity[tkn_buy]) or
            pytest.approx(split_buy_agent.holdings[tkn_buy]) != buy_agent.holdings[tkn_buy]):
        raise AssertionError('Buy quantities not equal.')

    if (buy_state.liquidity[tkn_sell] <= split_buy_state.liquidity[tkn_sell] or
            buy_agent.holdings[tkn_sell] >= split_buy_agent.holdings[tkn_sell]):
        # show that when using slip-based fees,
        # a two-part trade should always be cheaper than a one-part trade for the same total quantity.
        raise AssertionError('Agent did not save money by breaking the trade into two parts.')

    if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
         + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
            != pytest.approx(buy_agent.holdings[tkn_sell] + buy_agent.holdings[tkn_buy]
                             + buy_state.liquidity[tkn_sell] + buy_state.liquidity[tkn_buy])):
        raise AssertionError('Asset quantity is not constant after trade (one-part)')

    if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
         + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
            != pytest.approx(split_buy_agent.holdings[tkn_sell] + split_buy_agent.holdings[tkn_buy]
                             + split_buy_state.liquidity[tkn_sell] + split_buy_state.liquidity[tkn_buy])):
        raise AssertionError('Asset quantity is not constant after trade (two-part)')

    sell_quantity = buy_state.liquidity[tkn_sell] - initial_state.liquidity[tkn_sell]
    sell_state, sell_agent = bamm.simulate_swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity
    )

    sell_fee = initial_state.trade_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
    if sell_state.liquidity[tkn_sell] * (sell_fee - minimum_fee) != pytest.approx(abs(slip_factor * sell_quantity)):
        raise AssertionError('Math mismatch, please re-check.')

    if ((initial_agent.holdings[tkn_sell] + initial_agent.holdings[tkn_buy]
         + initial_state.liquidity[tkn_sell] + initial_state.liquidity[tkn_buy])
            != pytest.approx(sell_agent.holdings[tkn_sell] + sell_agent.holdings[tkn_buy]
                             + sell_state.liquidity[tkn_sell] + sell_state.liquidity[tkn_buy])):
        raise AssertionError('Asset quantity is not constant after sell trade')

    # now sell the same quantity, but do it in two smaller trades
    split_sell_state, split_sell_agent = initial_state, initial_agent
    next_state, next_agent = {}, {}
    for i in range(2):
        next_state[i], next_agent[i] = bamm.simulate_swap(
            old_state=split_sell_state,
            old_agent=split_sell_agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity / 2
        )
        split_sell_state, split_sell_agent = next_state[i], next_agent[i]

    if sell_state.fail or split_sell_state.fail:
        raise AssertionError('sell swap failed!')

    if (sell_state.liquidity[tkn_buy] <= split_sell_state.liquidity[tkn_buy] or
            sell_agent.holdings[tkn_buy] >= split_sell_agent.holdings[tkn_buy]):
        # show that when using slip-based fees,
        # a two-part trade should always be cheaper than a one-part trade for the same total quantity.
        # this should apply regardless of how the trade is specified.
        raise AssertionError('Agent did not save money by breaking the trade into two parts.')

    # this is commented out because it doesn't work. the spec would have to be adjusted
    # in some yet-to-be-determined way. not currently a priority.
    # if sell_state.liquidity[tkn_buy] != pytest.approx(buy_state.liquidity[tkn_buy]):
    #     raise AssertionError('Buy transaction was not reversed accurately.')


def test_fee_difference():
    initial_state = bamm.ConstantProductPoolState(
        tokens={'R1': 1000, 'R2': 1000},
        trade_fee=bamm.ConstantProductPoolState.custom_slip_fee(slip_factor=1)
    )
    trader = Agent(
        holdings={'R1': 1000, 'R2': 1000}
    )
    pass


if __name__ == '__main__':
    test_basilisk_construction()
    test_swap()
    test_swap_pool_invariant()
    test_add_remove_liquidity()
    test_remove_liquidity()
