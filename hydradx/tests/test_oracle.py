import pytest
import copy

from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.oracle import Oracle, Block
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.trade_strategies import schedule_swaps
from hydradx.model import run
from hypothesis import given, strategies as st, settings
from hydradx.tests.strategies_omnipool import asset_price_strategy, asset_quantity_strategy, \
    asset_quantity_bounded_strategy
from mpmath import mpf, mp
mp.dps = 50  # set decimal precision for mpmath

def test_oracle_multi_block_update():
    start_block = Block(
        OmnipoolState(
            tokens={
                'HDX': {'liquidity': 1000, 'LRNA': 1},
                'USD': {'liquidity': 10, 'LRNA': 1}
            }
        )
    )
    end_block = Block(
        OmnipoolState(
            tokens={
                'HDX': {'liquidity': 2000, 'LRNA': 2.2},
                'USD': {'liquidity': 11, 'LRNA': 0.99}
            }
        )
    )
    oracle_1 = Oracle(
        first_block=start_block,
        decay_factor=0.1
    )
    oracle_2 = Oracle(
        first_block=start_block,
        decay_factor=0.1
    )
    for i in range(10):
        end_block.time_step += 1
        oracle_1.update(end_block)
    oracle_2.update(end_block)

    for tkn in start_block.liquidity:
        for attr in ('liquidity', 'price', 'volume_in', 'volume_out'):
            if getattr(oracle_1, attr)[tkn] != pytest.approx(getattr(oracle_2, attr)[tkn], rel=1e-12):
                raise AssertionError(f"Oracle mismatch in {attr} for token {tkn}")


def test_update_every_block():
    omnipool1 = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 100000, 'LRNA': 100},
            'USD': {'liquidity': 1000, 'LRNA': 100}
        },
        unique_id='omnipool1'
    )
    omnipool2 = omnipool1.copy()
    omnipool2.update_function = lambda self: self.oracles['price'].update(self.current_block)
    omnipool2.unique_id = 'omnipool2'
    swaps = [
        *[None] * 4,
        {'tkn_sell': 'USD', 'tkn_buy': 'HDX', 'sell_quantity': 1},
        {'tkn_sell': 'HDX', 'tkn_buy': 'USD', 'buy_quantity': 1}
    ]
    trader1 = Agent(
        enforce_holdings=False,
        trade_strategy=schedule_swaps(pool_id='omnipool1', swaps=swaps)
    )
    trader2 = Agent(
        enforce_holdings=False,
        trade_strategy=schedule_swaps(pool_id='omnipool2', swaps=swaps)
    )
    initial_state = GlobalState(
        pools={'omnipool1': omnipool1, 'omnipool2': omnipool2},
        agents={'trader1': trader1, 'trader2': trader2},
    )
    events = run.run(initial_state, time_steps=10, silent=True)
    for pool in events[-1].pools.values():
        pool.oracles['price'].update(pool.current_block)

    final_oracles = [pool.oracles['price'] for pool in events[-1].pools.values()]
    for tkn in initial_state.asset_list:
        for attr in ('liquidity', 'price', 'volume_in', 'volume_out'):
            if getattr(final_oracles[0], attr)[tkn] != pytest.approx(getattr(final_oracles[1], attr)[tkn], rel=1e-12):
                raise AssertionError(f"Oracles don't match in {attr}.")



@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_oracle_one_empty_block(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                oracle_volume_in: list[float], oracle_volume_out: list[float],
                                oracle_prices: list[float], n):
    alpha = 2 / (n + 1)

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    initial_omnipool = OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        }
    )

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={}
    )

    events = run.run(initial_state=initial_state, time_steps=1, silent=True)
    omnipool_oracle = events[0].pools['omnipool'].oracles['price']
    # manually update oracle - it won't automatically update tokens that weren't used this block
    omnipool_oracle.update(events[-1].pools['omnipool'].current_block)
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle.liquidity[tkn] != pytest.approx(expected_liquidity, rel=1e-12):
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_in[tkn] != pytest.approx(expected_vol_in, rel=1e-12):
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_out[tkn] != pytest.approx(expected_vol_out, rel=1e-12):
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle.price[tkn] != pytest.approx(expected_price, rel=1e-12):
            raise AssertionError('Price is not correct.')


@given(
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.lists(st.floats(min_value=10, max_value=1000), min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
@settings(
    print_blob=True
)
def test_oracle_one_block_with_swaps(lrna: list[float], oracle_liquidity: list[float],
                                     oracle_volume_in: list[float], oracle_volume_out: list[float],
                                     oracle_prices: list[float], trade_sizes: list[float], n):
    alpha = mpf(2 / (n + 1))

    init_liquidity = {
        'HDX': {'liquidity': 1000, 'LRNA': mpf(lrna[0])},
        'USD': {'liquidity': 1000, 'LRNA': mpf(lrna[1])},
        'DOT': {'liquidity': 1000, 'LRNA': mpf(lrna[2])},
    }

    init_oracle = {
        'liquidity': {'HDX': mpf(oracle_liquidity[0]), 'USD': mpf(oracle_liquidity[1]), 'DOT': mpf(oracle_liquidity[2])},
        'volume_in': {'HDX': mpf(oracle_volume_in[0]), 'USD': mpf(oracle_volume_in[1]), 'DOT': mpf(oracle_volume_in[2])},
        'volume_out': {'HDX': mpf(oracle_volume_out[0]), 'USD': mpf(oracle_volume_out[1]), 'DOT': mpf(oracle_volume_out[2])},
        'price': {'HDX': mpf(oracle_prices[0]), 'USD': 1, 'DOT': mpf(oracle_prices[1])},
    }

    initial_omnipool = OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        }
    )

    omnipool_0 = initial_omnipool.update()
    omnipool_oracle_0 = omnipool_0.oracles['price'].update(omnipool_0.current_block)

    for tkn in ['HDX', 'USD', 'DOT']:
        # alpha_mod = alpha if vol_in[tkn] != 0 or vol_out[tkn] != 0 else 0
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle_0.liquidity[tkn] != pytest.approx(expected_liquidity, rel=1e-12):
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_in[tkn] != pytest.approx(expected_vol_in, rel=1e-12):
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_out[tkn] != pytest.approx(expected_vol_out, rel=1e-12):
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle_0.price[tkn] != pytest.approx(expected_price, rel=1e-12):
            raise AssertionError('Price is not correct.')

    trader = Agent(enforce_holdings=False)
    if trade_sizes[0] != trade_sizes[1]:
        er = 1
    omnipool_1 = omnipool_0.copy().swap(
        agent=trader,
        tkn_sell='DOT',
        tkn_buy='LRNA',
        sell_quantity=trade_sizes[0]
    ).swap(
        agent=trader,
        tkn_sell='LRNA',
        tkn_buy='DOT',
        buy_quantity=trade_sizes[1]
    ).update()
    vol_in = omnipool_1.last_block.volume_in
    vol_out = omnipool_1.last_block.volume_out
    omnipool_oracle_1 = omnipool_1.oracles['price'].update(omnipool_1.current_block, ['HDX', 'USD'])
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = omnipool_oracle_0.liquidity[tkn] * (1 - alpha) + alpha * omnipool_1.liquidity[tkn]
        if omnipool_oracle_1.liquidity[tkn] != pytest.approx(expected_liquidity, 1e-12):
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = omnipool_oracle_0.volume_in[tkn] * (1 - alpha) + alpha * vol_in[tkn]
        if omnipool_oracle_1.volume_in[tkn] != pytest.approx(expected_vol_in, 1e-12):
            raise AssertionError('Volume is not correct.')

        expected_vol_out = omnipool_oracle_0.volume_out[tkn] * (1 - alpha) + alpha * vol_out[tkn]
        if omnipool_oracle_1.volume_out[tkn] != pytest.approx(expected_vol_out, 1e-12):
            raise AssertionError('Volume out is not correct.')

        expected_price = omnipool_oracle_0.price[tkn] * (1 - alpha) + alpha * omnipool_1.lrna_price(tkn)
        if omnipool_oracle_1.price[tkn] != pytest.approx(expected_price, 1e-12):
            raise AssertionError('Price is not correct.')


def test_oracle_multi_block():
    init_liquidity = {
        'HDX': {'liquidity': 100000, 'LRNA': 100000},
        'USD': {'liquidity': 100000, 'LRNA': 100000},
        'DOT': {'liquidity': 100000, 'LRNA': 100000},
    }

    init_oracle = {
        'liquidity': {'HDX': 100000, 'USD': 100000, 'DOT': 100000},
        'volume_in': {'HDX': 0, 'USD': 0, 'DOT': 0},
        'volume_out': {'HDX': 0, 'USD': 0, 'DOT': 0},
        'price': {'HDX': 1.0, 'USD': 1.0, 'DOT': 1.0},
    }

    initial_omnipool = OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': 10
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        }
    )

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={'trader': Agent(
            enforce_holdings=False,
            trade_strategy=schedule_swaps(
                'omnipool', swaps=[
                    *[None] * 4,
                    {'tkn_sell': 'DOT', 'tkn_buy': 'HDX', 'sell_quantity': 1000},
                    *[None] * 4,
                    {'tkn_sell': 'HDX', 'tkn_buy': 'DOT', 'buy_quantity': 1000},
                ]
            )
        )}
    )

    events = run.run(initial_state=initial_state, time_steps=10, silent=True)
    final_omnipool = events[-1].pools['omnipool'].update()
    omnipool_oracle = final_omnipool.oracles['price']
    expected_vol_in = {'DOT': 66.66324219149098, 'HDX': 178.8008316214096, 'USD': 0}
    expected_vol_out = {'DOT': 181.8181818181818, 'HDX': 65.1604525453251, 'USD': 0}
    expected_liquidity = {'DOT': 100518.19722832012, 'HDX': 99494.56587073715, 'USD': 100000}
    expected_price = {'DOT': 0.9897985814516658, 'HDX': 1.0102943182908344, 'USD': 1.0}
    for tkn in initial_omnipool.asset_list:
        if omnipool_oracle.liquidity[tkn] != pytest.approx(expected_liquidity[tkn], rel=1e-12):
            raise AssertionError('Liquidity is not correct.')

        if omnipool_oracle.volume_in[tkn] != pytest.approx(expected_vol_in[tkn], rel=1e-12):
            raise AssertionError('Volume in is not correct.')

        if omnipool_oracle.volume_out[tkn] != pytest.approx(expected_vol_out[tkn], rel=1e-12):
            raise AssertionError('Volume out is not correct.')

        if omnipool_oracle.price[tkn] != pytest.approx(expected_price[tkn], rel=1e-12):
            raise AssertionError('Price is not correct.')

