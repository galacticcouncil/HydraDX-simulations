import random
import copy
import pytest
from hypothesis import given, strategies as st, settings
from hydradx.tests.test_omnipool_amm import omnipool_config
from hydradx.tests.test_basilisk_amm import constant_product_pool_config
from hydradx.model.amm.basilisk_amm import ConstantProductPoolState
from hydradx.model.amm.global_state import GlobalState, fluctuate_prices
from hydradx.model.amm.agents import Agent

from hydradx.model import run
from hydradx.model import processing
from hydradx.model.amm.trade_strategies import steady_swaps, invest_all, constant_product_arbitrage, invest_and_withdraw
from hydradx.model.amm.amm import AMM

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=1000, max_value=1000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@st.composite
def assets_config(draw, token_count: int = 0) -> dict:
    token_count = token_count or draw(asset_number_strategy)
    return_dict = {
        'HDX': draw(asset_price_strategy),
        'USD': 1
    }
    return_dict.update({
        f"{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}": draw(asset_price_strategy)
        for i in range(token_count - 2)
    })
    return return_dict


@st.composite
def agent_config(
    draw,
    holdings: dict = None,
    asset_list: list = None,
    trade_strategy: any = None
):
    return Agent(
        holdings=holdings or {
            tkn: draw(asset_quantity_strategy)
            for tkn in asset_list
        },
        trade_strategy=trade_strategy
    )


@st.composite
def global_state_config(
        draw,
        external_market: dict[str: float] = None,
        pools=None,
        agents=None,
        evolve_function=None,
        skip_omnipool=False
) -> GlobalState:
    # if asset_dict was not provided, generate one
    market_prices = external_market or draw(assets_config())
    # if prices were left blank, fill them in
    market_prices = {asset: (market_prices[asset] or draw(asset_price_strategy)) for asset in market_prices}
    # make sure USD is in there
    if 'USD' not in market_prices:
        market_prices['USD'] = 1

    asset_list = list(market_prices.keys())
    if not pools:
        pools = {}
        # a Basilisk pool for every asset pair
        for x in range(len(asset_list) - 1):
            for y in range(x + 1, len(asset_list)):
                x_quantity = draw(asset_quantity_strategy)
                pools.update({
                    f'{asset_list[x]}/{asset_list[y]}':
                    draw(constant_product_pool_config(
                        asset_dict={
                            asset_list[x]: x_quantity,
                            asset_list[y]: x_quantity * market_prices[asset_list[x]] / market_prices[asset_list[y]]
                        },
                        trade_fee=draw(fee_strategy)
                    ))
                })

        if not skip_omnipool:
            # add an Omnipool
            usd_price_lrna = 1  # draw(asset_price_strategy)
            market_prices.update({'LRNA': usd_price_lrna})
            liquidity = {tkn: 1000 for tkn in asset_list}
            pools.update({
                'omnipool': draw(omnipool_config(
                    asset_dict={
                        tkn: {
                            'liquidity': liquidity[tkn],
                            'LRNA': liquidity[tkn] * market_prices[tkn] / usd_price_lrna
                        }
                        for tkn in asset_list
                    },
                    lrna_fee=draw(fee_strategy),
                    asset_fee=draw(fee_strategy)
                ))
            })
    else:
        # if not otherwise specified, pool liquidity of each asset is inversely proportional to the value of the asset
        for pool in pools.values():
            for asset in pool.asset_list:
                pool.liquidity[asset] = pool.liquidity[asset] or 1000000 / market_prices[asset]
            if hasattr(pool, 'trade_fee') and pool.trade_fee.compute('', 0) < 0:
                pool.trade_fee = draw(fee_strategy)

    if not agents:
        agents = {
            f'Agent{_}': draw(agent_config(
                asset_list=asset_list
            ))
            for _ in range(5)
        }

    config = GlobalState(
        pools=pools,
        agents=agents,
        external_market=market_prices,
        evolve_function=evolve_function
    )
    return config


@given(global_state_config())
def test_simulation(initial_state: GlobalState):
    for a, agent in enumerate(initial_state.agents.values()):
        pool: AMM = initial_state.pools[list(initial_state.pools.keys())[a % len(initial_state.pools)]]
        agent.trade_strategy = [
            steady_swaps(pool_id=pool.unique_id, usd_amount=100),
            invest_all(pool_id=pool.unique_id)
        ][a % 2]

    # VVV -this would break the property test- VVV
    # initial_state.evolve_function = fluctuate_prices()

    initial_wealth = initial_state.total_wealth()
    events = run.run(initial_state, time_steps=5, silent=True)

    # property test: is there still the same total wealth held by all pools + agents?
    final_state = events[-1]
    if final_state.total_wealth() != pytest.approx(initial_wealth):
        raise AssertionError('total wealth quantity changed!')


@settings(deadline=500)
@given(global_state_config(
    external_market={
        'HDX': 0.08,
        'USD': 1
    },
    agents={
        'LP': Agent(
            holdings={
                'HDX': 0,
                'USD': 0
            },
            trade_strategy=invest_all('HDX/USD')
        ),
        'Trader1': Agent(
            holdings={
                'HDX': 80000,
                'USD': 1000
            },
            trade_strategy=steady_swaps('HDX/USD', 100, asset_list=['USD', 'HDX'])
        ),
        'Trader2': Agent(
            holdings={
                'HDX': 80000,
                'USD': 1000
            },
            trade_strategy=steady_swaps('HDX/USD', 100, asset_list=['HDX', 'USD'])
        )
    }
))
def test_LP(initial_state: GlobalState):
    initial_state.agents['LP'].holdings = {
        tkn: quantity for tkn, quantity in initial_state.pools['HDX/USD'].liquidity.items()
    }

    old_state = initial_state.copy()
    events = run.run(old_state, time_steps=10, silent=True)
    final_state: GlobalState = events[-1]

    if sum([final_state.agents['LP'].holdings[i] for i in initial_state.asset_list]) > 0:
        print('failed, not invested')
        raise AssertionError('Why does this LP not have all its assets in the pool???')
    if final_state.cash_out(final_state.agents['LP']) < initial_state.cash_out(initial_state.agents['LP']):
        print('failed, lost money.')
        raise AssertionError('The LP lost money!')
    # print('test passed.')


@given(global_state_config(
    pools={
        'HDX/BSX': ConstantProductPoolState(
            {
                'HDX': 2000000,
                'BSX': 1000000
            },
            trade_fee=0
        )
    },
    agents={
        'arbitrageur': Agent(
            holdings={'USD': float('inf')},
            trade_strategy=constant_product_arbitrage('HDX/BSX')
        )
    },
    external_market={'HDX': 0, 'BSX': 0},  # config function will fill these in with random values
    evolve_function=fluctuate_prices(volatility={'HDX': 1, 'BSX': 1}),
    skip_omnipool=True
))
def test_arbitrage_pool_balance(initial_state):
    # there are actually two things we would like to test:
    # one: with no fees, does the agent succeed in keeping the pool ratio balanced to the market prices?
    # two: with fees added, does the agent succeed in making money on every trade?
    # this test will focus on the first question

    events = run.run(initial_state, time_steps=50, silent=True)
    final_state = events[-1]
    final_pool_state = final_state.pools['HDX/BSX']
    if (pytest.approx(final_pool_state.liquidity['HDX'] / final_pool_state.liquidity['BSX'])
            != final_state.price('BSX') / final_state.price('HDX')):
        raise AssertionError('Price ratio does not match ratio in the pool!')


@settings(deadline=500)
@given(global_state_config(
    external_market={'HDX': 0, 'BSX': 0},  # config function will fill these in with random values
    pools={
        'HDX/BSX': ConstantProductPoolState(
            {
                'HDX': 0,
                'BSX': 0
            },
            trade_fee=0.1
        )
    },
    agents={
        'arbitrageur': Agent(
            holdings={'USD': 10000000000},  # lots
            trade_strategy=constant_product_arbitrage('HDX/BSX')
        )
    },
    evolve_function=fluctuate_prices(volatility={'HDX': 1, 'BSX': 1})
))
def test_arbitrage_profitability(initial_state: GlobalState):
    arb = initial_state.agents['arbitrageur']
    state = initial_state
    for i in range(50):
        state.evolve()
        prev_holdings = arb.holdings['USD']
        arb.trade_strategy.execute(state, 'arbitrageur')
        if arb.holdings['USD'] < prev_holdings:
            raise AssertionError('Arbitrageur lost money :(')


@given(global_state_config(
    external_market={'X': 0, 'Y': 0},  # config function will fill these in with random values
    pools={
        'X/Y': ConstantProductPoolState(
            {
                'X': 0,  # random via draw(asset_quantity_strategy)
                'Y': 0
            },
            trade_fee=-1  # i.e. choose one randomly via draw(fee_strategy)
        )
    },
    agents={
        'arbitrager': Agent()
    }
), asset_price_strategy)
def test_arbitrage_accuracy(initial_state: GlobalState, target_price: float):
    initial_state.external_market['Y'] = initial_state.external_market['X'] * target_price
    algebraic_function = constant_product_arbitrage('X/Y', minimum_profit=0, direct_calc=True)
    recursive_function = constant_product_arbitrage('X/Y', minimum_profit=0, direct_calc=False)

    def sell_spot(state: GlobalState):
        return (
                state.pools['X/Y'].liquidity['X']
                / state.pools['X/Y'].liquidity['Y']
                * (1 - state.pools['X/Y'].trade_fee.compute('', 0))
        )

    def buy_spot(state: GlobalState):
        return (
                state.pools['X/Y'].liquidity['X']
                / state.pools['X/Y'].liquidity['Y']
                / (1 - state.pools['X/Y'].trade_fee.compute('', 0))
        )

    algebraic_state: GlobalState = algebraic_function.execute(initial_state.copy(), 'arbitrager')
    recursive_state: GlobalState = recursive_function.execute(initial_state.copy(), 'arbitrager')
    algebraic_result = (algebraic_state.pools['X/Y'].liquidity['X']
                        / algebraic_state.pools['X/Y'].liquidity['Y'])
    recursive_result = (recursive_state.pools['X/Y'].liquidity['X']
                        / recursive_state.pools['X/Y'].liquidity['Y'])

    if algebraic_result != pytest.approx(recursive_result):
        raise AssertionError("Arbitrage calculation methods don't match.")

    if target_price < sell_spot(initial_state):
        if sell_spot(algebraic_state) != pytest.approx(target_price):
            raise AssertionError("Arbitrage calculation doesn't match expected result.")

    elif target_price > buy_spot(initial_state):
        if buy_spot(algebraic_state) != pytest.approx(target_price):
            raise AssertionError("Arbitrage calculation doesn't match expected result.")


@given(global_state_config(
    external_market={'R1': 2, 'R2': 3, 'R3': 4},
    agents={'trader': Agent(holdings={'USD': 1000, 'R1': 1000, 'R2': 1000, 'R3': 1000})},
    skip_omnipool=True
))
def test_buy_fee_derivation(initial_state: GlobalState):
    # hypothesis: fee_total = 1 - 1 / ((1 + fee_x) * (1 + fee_y) * (1 + fee_z) ...)
    # conclusion: this formula works only if there is infinite liquidity
    for pool in initial_state.pools.values():
        for tkn in pool.asset_list:
            pool.liquidity[tkn] = float('inf')
    feeless_state = initial_state.copy()
    for pool in feeless_state.pools.values():
        pool.trade_fee = 0
    asset_path = copy.copy(initial_state.asset_list)
    pool_path = []
    random.shuffle(asset_path)
    buy_amount = 1
    feeless_buy_amount = 1
    next_feeless_state = [feeless_state.copy()]
    next_state = [initial_state.copy()]
    for i in range(len(asset_path)-1):
        tkn_buy = asset_path[i]
        tkn_sell = asset_path[i+1]
        pool_id = f'{tkn_buy}/{tkn_sell}' if f'{tkn_buy}/{tkn_sell}' in initial_state.pools else f'{tkn_sell}/{tkn_buy}'
        pool_path.append(pool_id)
        next_feeless_state.append(next_feeless_state[-1].copy())
        next_feeless_state[-1].pools[pool_id].swap(
            agent=next_feeless_state[-1].agents['trader'],
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=feeless_buy_amount
        )
        feeless_buy_amount = (
            next_feeless_state[-2].agents['trader'].holdings[tkn_sell]
            - next_feeless_state[-1].agents['trader'].holdings[tkn_sell]
        )
        next_state.append(next_state[-1].copy())
        next_state[-1].pools[pool_id].swap(
            agent=next_state[-1].agents['trader'],
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=buy_amount
        )
        buy_amount = (
            next_state[-2].agents['trader'].holdings[tkn_sell]
            - next_state[-1].agents['trader'].holdings[tkn_sell]
        )
        if buy_amount == 0:
            return
    fee_amount = buy_amount / feeless_buy_amount - 1
    expected_fee_amount = 1
    for pool_id in pool_path:
        expected_fee_amount /= (1 - initial_state.pools[pool_id].trade_fee.compute('', 0))
    expected_fee_amount -= 1
    if fee_amount != pytest.approx(expected_fee_amount):
        raise ValueError(f'off by {abs(1-expected_fee_amount/fee_amount)}')
    # if fee_amount < expected_fee_amount:
    #     raise ValueError(f'fee is lower than expected')


@given(global_state_config(
    external_market={'R1': 1, 'R2': 1},
    agents={'trader': Agent(holdings={'USD': 1000, 'R1': 1000, 'R2': 1000})},
    skip_omnipool=True
))
def test_sell_fee_derivation(initial_state: GlobalState):
    for pool in initial_state.pools.values():
        for tkn in pool.asset_list:
            pool.liquidity[tkn] = float('inf')
    feeless_state = initial_state.copy()
    for pool in feeless_state.pools.values():
        pool.trade_fee = 0
    asset_path = copy.copy(initial_state.asset_list)
    pool_path = []
    random.shuffle(asset_path)
    sell_amount = 1
    feeless_sell_amount = 1
    next_feeless_state = [feeless_state.copy()]
    next_state = [initial_state.copy()]
    for i in range(len(asset_path)-1):
        tkn_buy = asset_path[i]
        tkn_sell = asset_path[i+1]
        pool_id = f'{tkn_buy}/{tkn_sell}' if f'{tkn_buy}/{tkn_sell}' in initial_state.pools else f'{tkn_sell}/{tkn_buy}'
        pool_path.append(pool_id)
        next_feeless_state.append(next_feeless_state[-1].copy())
        next_feeless_state[-1].pools[pool_id].swap(
            agent=next_feeless_state[-1].agents['trader'],
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            sell_quantity=feeless_sell_amount
        )
        feeless_sell_amount = (
            next_feeless_state[-1].agents['trader'].holdings[tkn_buy]
            - next_feeless_state[-2].agents['trader'].holdings[tkn_buy]
        )
        next_state.append(next_state[-1].copy())
        next_state[-1].pools[pool_id].swap(
            agent=next_state[-1].agents['trader'],
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            sell_quantity=sell_amount
        )
        sell_amount = (
            next_state[-1].agents['trader'].holdings[tkn_buy]
            - next_state[-2].agents['trader'].holdings[tkn_buy]
        )
        if sell_amount == 0:
            return
    fee_amount = 1 - sell_amount / feeless_sell_amount
    # what do we think it should be, if the derived formula is right?
    expected_fee_amount = 1
    for pool_id in pool_path:
        expected_fee_amount *= (1 - initial_state.pools[pool_id].trade_fee.compute('', 0))
    expected_fee_amount = 1 - expected_fee_amount
    if fee_amount != pytest.approx(expected_fee_amount):
        raise ValueError(f'off by {abs(1-expected_fee_amount/fee_amount)}')
    # if fee_amount > expected_fee_amount:
    #     raise ValueError('fee is higher than expected')


# @given(st.integers(min_value=1, max_value=100000))
def test_omnipool_arbitrage():
    import random

    from hydradx.model import run
    from hydradx.model.amm.omnipool_amm import OmnipoolState, dynamicadd_lrna_fee
    from hydradx.model.amm.agents import Agent
    from hydradx.model.amm.trade_strategies import omnipool_arbitrage, random_swaps
    from hydradx.model.amm.global_state import GlobalState
    from hydradx.model import plot_utils as pu

    # same seed, same parameters = same simulation result
    random.seed(42)

    assets = {
        'HDX': {'usd price': 0.05, 'weight': 0.10},
        'USD': {'usd price': 1, 'weight': 0.20},
        'AUSD': {'usd price': 1, 'weight': 0.10},
        'ETH': {'usd price': 2500, 'weight': 0.40},
        'DOT': {'usd price': 5.37, 'weight': 0.20}
    }

    lrna_price_usd = 0.07
    initial_omnipool_tvl = 10000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = initial_omnipool_tvl * info['weight'] / info['usd price']
        lrna[tkn] = initial_omnipool_tvl * info['weight'] / lrna_price_usd

    initial_state = GlobalState(
        pools={
            'Omnipool': OmnipoolState(
                tokens={
                    tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
                },
                oracles={
                    'short': 10,
                    'medium': 40,
                    'long': 160
                },
                lrna_fee=dynamicadd_lrna_fee(
                    minimum=0.0025,
                    amplification=10,
                    raise_oracle_name='medium',
                    decay=0.00001,
                    fee_max=0.1,
                ),
                asset_fee=0,
                preferred_stablecoin='USD'
            )
        },
        agents={
            'Arbitrageur': Agent(
                holdings={tkn: 100000000 for tkn in list(assets.keys()) + ['LRNA']},
                trade_strategy=omnipool_arbitrage('Omnipool')
            ),
            'Trader': Agent(
                holdings={tkn: 100000000 for tkn in list(assets.keys()) + ['LRNA']},
                trade_strategy=random_swaps(
                    pool_id='Omnipool',
                    amount={'USD': 1000, 'HDX': 1000, 'AUSD': 1000, 'ETH': 1000, 'DOT': 1000}
                )
            ),
            'LP': Agent(
                holdings={tkn: 100000000 for tkn in list(assets.keys()) + ['LRNA']},
                trade_strategy=invest_and_withdraw(pool_id='Omnipool')
            )
        },
        # evolve_function=fluctuate_prices(volatility={'DOT': 1, 'HDX': 1}),
        external_market={tkn: assets[tkn]['usd price'] for tkn in assets},
    )

    # print(initial_state)
    time_steps = 1000  # len(price_list) - 1
    events = run.run(initial_state, time_steps=time_steps)
    # asset_prices = pu.get_datastream(events, asset='all')
    # dot_prices = pu.get_datastream(events, asset='DOT')
    hdx_price = pu.get_datastream(events, pool='Omnipool', prop='usd_price', key='HDX')
    # pool_val = pu.get_datastream(events, pool='Omnipool', prop='pool_val')
    # oracles_hdx_liquidity = pu.get_datastream(events, pool='Omnipool', oracle='all', prop='liquidity', key='HDX')
    # oracles_hdx_price = pu.get_datastream(events, pool='Omnipool', oracle='all', prop='price', key='HDX')
    # short_oracle_usd = pu.get_datastream(events, pool='Omnipool', oracle='short', prop='all', key='USD')
    arb_holdings = pu.get_datastream(events, agent='Arbitrageur', prop='holdings', key='all')
    deposit_val = pu.get_datastream(events, agent='LP', prop='deposit_val')
    arb_holdings = [
        {tkn: arb_holdings[tkn][i] for tkn in arb_holdings} for i in range(len(arb_holdings['LRNA']))
    ]
    profit = (
        GlobalState.value_assets(arb_holdings[-1], initial_state.external_market)
        - GlobalState.value_assets(arb_holdings[0], initial_state.external_market)
    )
    if profit < 0:
        raise ValueError(f'Arbitrageur lost {profit} USD')
    # pu.plot(events, pool='Omnipool', prop='liquidity')
    # er = 1
