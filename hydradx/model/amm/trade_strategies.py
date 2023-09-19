import math
import copy
from .global_state import GlobalState
from .agents import Agent
from .amm import AMM
from .basilisk_amm import ConstantProductPoolState
from .omnipool_amm import OmnipoolState
from . import omnipool_amm as oamm
from .stableswap_amm import StableSwapPoolState
from typing import Callable
import random
# from numbers import Number

import numpy as np


class TradeStrategy:
    def __init__(self, strategy_function: Callable[[GlobalState, str], GlobalState], name: str, run_once: bool = False):
        self.function = strategy_function
        self.run_once = run_once
        self.done = False
        self.name = name

    def execute(self, state: GlobalState, agent_id: str) -> GlobalState:
        if self.done:
            return state
        elif self.run_once:
            self.done = True
        return_val = self.function(state, agent_id)
        if return_val != state:
            raise AssertionError('TradeStrategy function returned a different state object.')
        return return_val

    def __add__(self, other):
        assert isinstance(other, TradeStrategy)

        def combo_function(state, agent_id) -> GlobalState:
            new_state = self.execute(state, agent_id)
            return other.execute(new_state, agent_id)

        return TradeStrategy(combo_function, name='\n'.join([self.name, other.name]))


def random_swaps(
    pool_id: str,
    amount: dict[str: float] or float,
    randomize_amount: bool = True,
) -> TradeStrategy:
    """
    amount should be a dict in the form of:
    {
        token_name: sell_quantity
    }
    """

    def strategy(state: GlobalState, agent_id: str):

        buy_asset = random.choice(list(amount.keys()))
        sell_asset = random.choice(list(amount.keys()))
        sell_quantity = (
            amount[sell_asset] / (state.price(sell_asset) or 1) * (random.random() if randomize_amount else 1)
        ) or 1
        if buy_asset == sell_asset:
            return state
        else:
            pool = state.pools[pool_id]
            pool.swap(
                agent=state.agents[agent_id],
                tkn_sell=sell_asset,
                tkn_buy=buy_asset,
                sell_quantity=sell_quantity
            )
        return state

    return TradeStrategy(strategy, name=f'random swaps ({list(amount.keys())})')


def steady_swaps(
    pool_id: str,
    usd_amount: float,
    asset_list: list = ()
) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        strategy.buy_index = getattr(strategy, 'buy_index', -1)
        strategy.buy_index += 1
        agent = state.agents[agent_id]
        assets = asset_list or list(agent.holdings.keys())
        buy_index = strategy.buy_index % len(assets)
        sell_index = (buy_index + 1) % len(assets)

        buy_asset = assets[buy_index]
        sell_asset = assets[sell_index]
        sell_quantity = usd_amount / state.price(sell_asset)

        pool = state.pools[pool_id]
        agent = state.agents[agent_id]
        pool.swap(
            agent=agent,
            tkn_sell=sell_asset,
            tkn_buy=buy_asset,
            sell_quantity=sell_quantity
        )
        return state

    return TradeStrategy(strategy, name=f'steady swaps (${usd_amount})')


def constant_swaps(
    pool_id: str,
    sell_quantity: float,
    sell_asset: str,
    buy_asset: str
) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        return state.execute_swap(
            pool_id=pool_id,
            agent_id=agent_id,
            tkn_sell=sell_asset,
            tkn_buy=buy_asset,
            sell_quantity=sell_quantity
        )

    return TradeStrategy(strategy, name=f'constant swaps (${sell_quantity})')


def back_and_forth(
    pool_id: str,
    percentage: float  # percentage of TVL to trade each block
) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):
        omnipool: OmnipoolState = state.pools[pool_id]
        agent: Agent = state.agents[agent_id]
        assets = list(set(agent.asset_list) & set(omnipool.asset_list))
        for asset in assets:
            # asset = agent.asset_list[i]
            dr = percentage / 2 * omnipool.liquidity[asset]
            lrna_init = state.agents[agent_id].holdings['LRNA']
            omnipool.swap(agent=agent, tkn_sell=asset, tkn_buy='LRNA', sell_quantity=dr, modify_imbalance=False)
            dq = state.agents[agent_id].holdings['LRNA'] - lrna_init
            omnipool.swap(agent=agent, tkn_sell='LRNA', tkn_buy=asset, sell_quantity=dq, modify_imbalance=False)

        return state

    return TradeStrategy(strategy, name=f'back and forth (${percentage})')


def invest_all(pool_id: str, assets: list or str = None, when: int = 0) -> TradeStrategy:

    if assets and not isinstance(assets, list):
        assets = [assets]

    class Strategy:
        def __init__(self, _when):
            self.when = _when
            self.done = False

        def execute(self, state: GlobalState, agent_id: str):
            if self.done or state.time_step < self.when:
                return state
            self.done = True
            agent: Agent = state.agents[agent_id]
            pool = state.pools[pool_id]

            for asset in assets or list(agent.holdings.keys()):
                if agent.holdings[asset] == 0:
                    continue
                if asset in state.pools[pool_id].asset_list:
                    pool.add_liquidity(
                        agent=agent,
                        quantity=agent.holdings[asset],
                        tkn_add=asset
                    )

            return state

    return TradeStrategy(Strategy(when).execute, name=f'invest all ({pool_id})')


def withdraw_all(when: int) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):
        if state.time_step == when:
            agent = state.agents[agent_id]
            new_state = state
            for key in agent.holdings.keys():
                # shares.keys might just be the pool name, or it might be a tuple (pool, token)
                if isinstance(key, tuple):
                    pool_id = key[0]
                    tkn = key[1]
                else:
                    pool_id = key
                    tkn = key
                if pool_id in state.pools:
                    pool: AMM = state.pools[pool_id]
                    pool.remove_liquidity(
                        agent=agent,
                        quantity=agent.holdings[key],
                        tkn_remove=tkn
                    )
            return new_state
        else:
            return state

    return TradeStrategy(strategy, name=f'withdraw all at time step {when}')


def sell_all(pool_id: str, tkn_sell: str, tkn_buy: str, when: int = -1) -> TradeStrategy:

    class Strategy:
        def __init__(self):
            self.when = when
            self.done = False

        def execute(self, state: GlobalState, agent_id: str) -> GlobalState:
            agent = state.agents[agent_id]
            if self.done or not agent.holdings[tkn_sell] or state.time_step < self.when:
                return state
            if self.when > 0:
                self.done = True
            return state.execute_swap(
                pool_id, agent_id, tkn_sell, tkn_buy, sell_quantity=agent.holdings[tkn_sell]
            )

    return TradeStrategy(Strategy().execute, name=f'sell all {tkn_sell} for {tkn_buy}')


def invest_and_withdraw(frequency: float = 0.001, pool_id: str = 'omnipool', sell_lrna: bool = False) -> TradeStrategy:
    class Strategy:
        def __init__(self):
            self.last_move = 0
            self.invested = False

        def __call__(self, state: GlobalState, agent_id: str) -> GlobalState:

            if (state.time_step - self.last_move) * frequency > random.random():
                omnipool: OmnipoolState = state.pools[pool_id]
                agent: Agent = state.agents[agent_id]
                agent_holdings = copy.copy(agent.holdings)

                if self.invested:
                    # withdraw
                    for tkn in agent_holdings:
                        if isinstance(tkn, tuple) and tkn[0] == pool_id:
                            omnipool.remove_liquidity(
                                agent=agent,
                                quantity=agent.holdings[tkn],
                                tkn_remove=tkn[1]
                            )
                        if sell_lrna:
                            omnipool.swap(
                                agent=agent,
                                tkn_sell='LRNA',
                                tkn_buy=omnipool.stablecoin,
                                sell_quantity=agent.holdings['LRNA']
                            )
                else:
                    # invest
                    for tkn in agent_holdings:
                        if tkn in state.pools[pool_id].asset_list:
                            omnipool.add_liquidity(
                                agent=agent,
                                quantity=agent.holdings[tkn],
                                tkn_add=tkn
                            )

                self.last_move = state.time_step
                self.invested = not self.invested

            return state

    return TradeStrategy(Strategy(), name=f'invest and withdraw every {frequency} time steps')


# iterative arbitrage method
def find_agent_delta_y(
    target_price: float,
    price_after_trade: Callable,
    starting_bid: float = 1,
    precision: float = 0.000000001,
    max_iterations: int = 50
):
    b = starting_bid
    previous_change = 1
    p = price_after_trade(b)
    previous_price = p
    diff = p / target_price
    i = 0
    while abs(1 - diff) > precision and i < max_iterations:
        progress = (previous_price - p) / (previous_price - target_price) or 2
        old_b = b
        b -= previous_change * (1 - 1 / progress)
        previous_price = p
        previous_change = b - old_b
        p = price_after_trade(buy_amount=b if b >= 0 else 0, sell_amount=-b if b < 0 else 0)
        diff = p / target_price
        i += 1

    return b if i < max_iterations else 0


def constant_product_arbitrage(pool_id: str, minimum_profit: float = 0, direct_calc: bool = True) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str):

        pool: ConstantProductPoolState = state.pools[pool_id]
        agent: Agent = state.agents[agent_id]
        if not(isinstance(pool, ConstantProductPoolState)):
            raise TypeError(f'{pool_id} is not compatible with constant product arbitrage.')

        x = pool.asset_list[0]
        y = pool.asset_list[1]

        if not direct_calc:
            agent_delta_y = recursive_calculation(state, x, y)
        else:
            agent_delta_y = direct_calculation(state, x, y)

        agent_delta_x = -agent_delta_y * pool.liquidity[x] / (pool.liquidity[y] - agent_delta_y)
        if agent_delta_y > 0:
            agent_delta_x /= 1 - pool.trade_fee.compute(y, abs(agent_delta_y))
        else:
            agent_delta_x *= 1 - pool.trade_fee.compute(y, abs(agent_delta_y))

        projected_profit = (
            agent_delta_y * state.price(y)
            + agent_delta_x * state.price(x)
        )

        # in case we want to graph this later
        # agent = state.agents[agent_id]
        # agent.projected_profit = projected_profit

        if projected_profit <= minimum_profit:
            # don't do it
            # agent.trade_rejected += 1
            return state

        # buy just enough of non-USD asset
        if agent_delta_y > 0 and x != 'USD' or agent_delta_y < 0 and y != 'USD':
            state.external_market_trade(
                agent_id=agent_id,
                tkn_buy=x if agent_delta_y > 0 else 'USD',
                tkn_sell=y if agent_delta_y < 0 else 'USD',
                sell_quantity=agent_delta_y if agent_delta_y < 0 else 0,
                buy_quantity=-agent_delta_x if agent_delta_y > 0 else 0
            )

        # swap
        pool.swap(agent=agent, tkn_sell=x, tkn_buy=y, buy_quantity=agent_delta_y)

        # immediately cash out everything for USD
        for tkn, quantity in agent.holdings.items():
            if agent.holdings[tkn] > 0 and tkn != 'USD':
                state.external_market_trade(
                    agent_id, tkn_buy='USD', tkn_sell=tkn, sell_quantity=quantity
                )

        return state

    def direct_calculation(state: GlobalState, tkn_sell: str, tkn_buy: str):

        pool = state.pools[pool_id]
        p = state.price(tkn_buy) / state.price(tkn_sell)
        x = pool.liquidity[tkn_sell]
        y = pool.liquidity[tkn_buy]
        f = pool.trade_fee.compute('', 0)
        if p < x/y * (1 - f):
            # agent can profit by selling y to AMM
            b = 2 * y - (f / p) * x * (1 - f)
            c = y ** 2 - x * y / p * (1 - f)
            t = math.sqrt(b ** 2 - 4 * c)
            if -b < t:
                dY = (-b + t) / 2
            else:
                dY = (-b - t) / 2
            return -dY
        elif p > x/y * (1 + f):
            # agent can profit by selling x to AMM
            b = 2 * y + (f / p) * x / (1 - f)
            c = y ** 2 - x * y / p / (1 - f)
            t = math.sqrt(b ** 2 - 4 * c)
            if -b < t:
                dY = (-b + t) / 2
            else:
                dY = (-b - t) / 2
            return -dY
        else:
            return 0

    def recursive_calculation(state: GlobalState, tkn_sell: str, tkn_buy: str):
        # an alternate way to calculate optimal trade
        # should be equivalent with a flat percentage fee, but also works with more complicated fee structures
        pool = state.pools[pool_id]
        x = pool.liquidity[tkn_sell]
        y = pool.liquidity[tkn_buy]
        # VVV this would be correct if there is no fee VVV
        agent_delta_y = (y - math.sqrt(x * y * state.price(tkn_sell) / state.price(tkn_buy)))

        def price_after_trade(buy_amount=0, sell_amount=0):
            if buy_amount:
                sell_amount = -(x - pool.invariant / (y - buy_amount)) \
                              * (1 + pool.trade_fee.compute(tkn_sell, buy_amount))
                price = (y - buy_amount) / (x + sell_amount)

            elif sell_amount:
                buy_amount = (x - pool.invariant / (y + sell_amount)) \
                             * (1 - pool.trade_fee.compute(tkn_sell, sell_amount))
                price = (y + sell_amount) / (x - buy_amount)

            else:
                raise ValueError('Must specify either buy_amount or sell_amount')

            if agent_delta_y < 0:
                price /= (1 - pool.trade_fee.compute(y, sell_amount))
            else:
                price /= (1 + pool.trade_fee.compute(y, sell_amount))

            return price

        target_price = state.price(tkn_sell) / state.price(tkn_buy)
        return find_agent_delta_y(target_price, price_after_trade, agent_delta_y)

    return TradeStrategy(strategy, name=f'constant product pool arbitrage ({pool_id})')


def omnipool_arbitrage(pool_id: str, arb_precision=1, skip_assets=None):
    if skip_assets is None:
        skip_assets = []

    def get_mat(prices: list[float], reserves: list[int], lrna: list[int], usd_index: int):
        mat = [[float(1)] * len(prices)]
        for i in range(len(prices)):
            if usd_index == i:
                continue
            row = [float(0)] * len(prices)
            row[usd_index] = math.sqrt(prices[i] * reserves[i] * lrna[i])
            row[i] = -math.sqrt(reserves[usd_index] * lrna[usd_index])
            mat.append(row)
        return mat

    def calc_new_reserve(new_reserve_b, old_reserve_a, old_reserve_b):
        return old_reserve_a * old_reserve_b / new_reserve_b

    def get_dr_list(prices, reserves, lrna, usd_index):
        # delta_ri, i.e. change in reserve of asset i

        mat = get_mat(prices, reserves, lrna, usd_index)

        A = np.array(mat)
        B_ls = [float(0)] * len(prices)
        B_ls[0] = float(sum(lrna))
        B = np.array(B_ls)
        X = np.linalg.solve(A, B)

        dr = [calc_new_reserve(X[i], reserves[i], lrna[i]) - reserves[i] for i in range(len(prices))]
        return dr

    def get_dq_list(dr, reserves: list, lrna: list):
        # delta_qi, i.e. the change in the amount of lrna in pool i
        dq = [-lrna[i] * dr[i] / (reserves[i] + dr[i]) for i in range(len(reserves))]
        return dq

    def strategy(state: GlobalState, agent_id: str) -> GlobalState:
        omnipool: OmnipoolState = state.pools[pool_id]
        agent: Agent = state.agents[agent_id]
        if not isinstance(omnipool, OmnipoolState):
            raise AssertionError()

        reserves = []
        lrna = []
        prices = []
        asset_list = []
        asset_fees = []
        lrna_fees = []
        skip_ct = 0
        usd_index = omnipool.asset_list.index(omnipool.stablecoin)
        usd_fee = omnipool.asset_fee[omnipool.stablecoin].compute(tkn=omnipool.stablecoin)
        # usd_fee = omnipool.last_fee[omnipool.stablecoin]
        usd_LRNA_fee = omnipool.lrna_fee[omnipool.stablecoin].compute(tkn=omnipool.stablecoin)
        # usd_LRNA_fee = omnipool.last_lrna_fee[omnipool.stablecoin]

        for i in range(len(omnipool.asset_list)):
            asset = omnipool.asset_list[i]

            if asset in skip_assets:  # we may not want to arb all assets
                skip_ct += 1
                if i < usd_index:
                    usd_index -= 1
                continue
            if asset == omnipool.stablecoin:
                usd_index = i - skip_ct

            asset_fee = omnipool.asset_fee[asset].compute(tkn=asset)
            # asset_fee = omnipool.last_fee[asset]
            asset_LRNA_fee = omnipool.lrna_fee[asset].compute(tkn=asset)
            # asset_LRNA_fee = omnipool.last_lrna_fee[asset]
            if arb_precision < 2:
                low_price = (1 - usd_fee) * (1 - asset_LRNA_fee) * oamm.usd_price(omnipool, tkn=asset)
                high_price = 1 / (1 - asset_fee) / (1 - usd_LRNA_fee) * oamm.usd_price(omnipool, tkn=asset)

                if asset != omnipool.stablecoin and low_price <= state.external_market[asset] <= high_price:
                    skip_ct += 1
                    if i < usd_index:
                        usd_index -= 1
                    continue

            reserves.append(omnipool.liquidity[asset])
            lrna.append(omnipool.lrna[asset])
            prices.append(state.external_market[asset])
            asset_list.append(asset)
            asset_fees.append(asset_fee)
            lrna_fees.append(asset_LRNA_fee)

        dr = get_dr_list(prices, reserves, lrna, usd_index)
        dq = get_dq_list(dr, reserves, lrna)

        r = omnipool.liquidity
        q = omnipool.lrna

        for j in range(arb_precision):
            dr = [0]*len(dq)
            for i in range(len(prices)):
                asset = asset_list[i]
                delta_Qi = dq[i] * (j+1) / arb_precision
                if delta_Qi > 0:
                    dr[i] = r[asset] * delta_Qi / (q[asset] + delta_Qi) * (1 - asset_fees[i])
                else:
                    delta_Qi_fee_adj = delta_Qi / (1 - lrna_fees[i])
                    dr[i] = r[asset] * delta_Qi_fee_adj / (q[asset] + delta_Qi_fee_adj)
            profit = sum([dr[i] * prices[i] for i in range(len(prices))])
            if profit < 0:
                if j > 0:
                    for i in range(len(asset_list)):
                        if dq[i] > 0:
                            omnipool.swap(
                                agent=agent, tkn_sell="LRNA", tkn_buy=asset_list[i],
                                sell_quantity=dq[i] * j/arb_precision, modify_imbalance=False)
                        else:
                            omnipool.swap(
                                agent=agent, tkn_sell=asset_list[i], tkn_buy="LRNA",
                                buy_quantity=-dq[i] * j/arb_precision, modify_imbalance=False)
                break
            elif j == arb_precision - 1:
                for i in range(len(asset_list)):
                    if dq[i] > 0:
                        omnipool.swap(
                            agent=agent, tkn_sell="LRNA", tkn_buy=asset_list[i],
                            sell_quantity=dq[i], modify_imbalance=False)
                    else:
                        omnipool.swap(
                            agent=agent, tkn_sell=asset_list[i], tkn_buy="LRNA",
                            buy_quantity=-dq[i], modify_imbalance=False)
                break  # technically unnecessary

        return state

    return TradeStrategy(strategy, name='omnipool arbitrage')


def stableswap_arbitrage(pool_id: str, minimum_profit: float = 1, precision: float = 0.00001):

    def strategy(state: GlobalState, agent_id: str) -> GlobalState:

        stable_pool: StableSwapPoolState = state.pools[pool_id]
        if not isinstance(stable_pool, StableSwapPoolState):
            raise AssertionError()
        sorted_assets = sorted(list(
            stable_pool.liquidity.keys()), key=lambda k: state.external_market[k] / stable_pool.liquidity.get(k)
        )
        buy_asset = sorted_assets[0]
        sell_asset = sorted_assets[-1]
        target_price = state.external_market[buy_asset] / state.external_market[sell_asset]

        d = stable_pool.calculate_d()

        def price_after_trade(buy_amount: float = 0, sell_amount: float = 0):
            buy_amount = buy_amount or sell_amount
            balance_out = stable_pool.liquidity[buy_asset] - buy_amount
            balance_in = stable_pool.calculate_y(
                stable_pool.modified_balances(delta={buy_asset: -buy_amount}, omit=[sell_asset]), d
            )
            return stable_pool.price_at_balance([balance_in, balance_out], d)

        delta_y = find_agent_delta_y(target_price, price_after_trade, precision=precision)
        delta_x = (
            stable_pool.liquidity[sell_asset]
            - stable_pool.calculate_y(stable_pool.modified_balances(delta={buy_asset: -delta_y}, omit=[sell_asset]), d)
        ) * (1 + stable_pool.trade_fee)

        projected_profit = (
            delta_y * state.price(buy_asset)
            + delta_x * state.price(sell_asset)
        )

        if projected_profit <= minimum_profit:
            # don't do it
            # agent.trade_rejected += 1
            return state

        new_state = state.execute_swap(pool_id, agent_id, sell_asset, buy_asset, buy_quantity=delta_y)
        return new_state

    return TradeStrategy(strategy, name='stableswap arbitrage')


def toxic_asset_attack(pool_id: str, asset_name: str, trade_size: float, start_timestep: int = 0) -> TradeStrategy:

    def strategy(state: GlobalState, agent_id: str) -> GlobalState:
        if state.time_step < start_timestep:
            return state

        state.external_market[asset_name] = 0

        pool = state.pools[pool_id]
        agent = state.agents[agent_id]
        current_price = oamm.lrna_price(pool, asset_name)
        if current_price <= 0:
            return state
        usd_price = oamm.lrna_price(pool, pool.stablecoin) / current_price
        if usd_price <= 0:
            return state
        quantity = (
            (pool.lrna_total - pool.lrna[asset_name])
            * pool.weight_cap[asset_name] / (1 - pool.weight_cap[asset_name])
            - pool.lrna[asset_name]
        ) / current_price - 0.001  # because rounding errors

        pool.simulate_add_liquidity(
            pool_id, agent_id,
            quantity=quantity,
            tkn_add=asset_name
        )
        sell_quantity = trade_size * usd_price
        if state.pools[pool_id].liquidity[asset_name] + sell_quantity > 10 ** 12:
            # go right up to the maximum
            sell_quantity = max(10 ** 12 - state.pools[pool_id].liquidity[asset_name] - 0.0000001, 0)
            if sell_quantity == 0:
                # pool is maxed
                return state
        pool.swap(
            agent=agent,
            tkn_sell=asset_name,
            tkn_buy='USD',
            sell_quantity=sell_quantity
        )
        return state

    return TradeStrategy(strategy, name=f'toxic asset attack (asset={asset_name}, trade_size={trade_size})')


def price_manipulation(
        pool_id: str, asset1: str, asset2: str, max_fraction: float = float('inf'), interval: int = 1
):
    def price_manipulation_strategy(state: GlobalState, agent_id: str):
        if state.time_step % interval != 0:
            return state
        omnipool: OmnipoolState = state.pools[pool_id]
        agent: Agent = state.agents[agent_id]
        agent_holdings = copy.copy(agent.holdings)
        omnipool.swap(
            agent=agent,
            tkn_sell=asset1, tkn_buy=asset2,
            sell_quantity=min(agent.holdings[asset1], omnipool.liquidity[asset1] * max_fraction)
        )
        omnipool.add_liquidity(
            agent=agent,
            quantity=min(agent.holdings[asset2] / 2, omnipool.liquidity[asset2] * max_fraction),
            # quantity=agent.holdings[asset2] / 2,
            tkn_add=asset2
        )
        # here we want to bring the prices back in line with the market
        delta_r = (math.sqrt((
            omnipool.lrna[asset2] * omnipool.lrna[asset1] * omnipool.liquidity[asset2] * omnipool.liquidity[asset1]
        ) / (state.external_market[asset2] / state.external_market[asset1])) - (
            omnipool.lrna[asset1] * omnipool.liquidity[asset2]
        )) / (omnipool.lrna[asset2] + omnipool.lrna[asset1])
        omnipool.swap(
            agent=agent,
            tkn_sell=asset2, tkn_buy=asset1,
            sell_quantity=min(delta_r, agent.holdings[asset2], omnipool.liquidity[asset2] * max_fraction)
        )
        omnipool.remove_liquidity(
            agent=agent,
            quantity=agent.holdings[(omnipool.unique_id, asset2)],
            tkn_remove=asset2
        )
        # back the other way
        omnipool.swap(
            agent=agent,
            tkn_sell=asset2, tkn_buy=asset1,
            sell_quantity=min(agent.holdings[asset2], omnipool.liquidity[asset2] * max_fraction)
        )
        omnipool.add_liquidity(
            agent=agent,
            # quantity=min(agent.holdings[asset1] / 2, omnipool.liquidity[asset1] / 3),
            quantity=min(agent.holdings[asset1] / 2, omnipool.liquidity[asset1] * max_fraction),
            tkn_add=asset1
        )
        delta_r = (math.sqrt((
            omnipool.lrna[asset1] * omnipool.lrna[asset2] * omnipool.liquidity[asset1] * omnipool.liquidity[asset2]
        ) / (state.external_market[asset1] / state.external_market[asset2])) - (
            omnipool.lrna[asset2] * omnipool.liquidity[asset1]
        )) / (omnipool.lrna[asset1] + omnipool.lrna[asset2])
        omnipool.swap(
            agent=agent,
            tkn_sell=asset1, tkn_buy=asset2,
            sell_quantity=min(delta_r, agent.holdings[asset1], omnipool.liquidity[asset1] * max_fraction)
        )
        omnipool.remove_liquidity(
            agent=agent,
            quantity=agent.holdings[(omnipool.unique_id, asset1)],
            tkn_remove=asset1
        )
        omnipool.swap(
            agent=agent,
            tkn_sell='LRNA', tkn_buy=asset2,
            sell_quantity=agent.holdings['LRNA']
        )
        profit = sum(agent.holdings.values()) - sum(agent_holdings.values())
        return state

    return TradeStrategy(
        strategy_function=price_manipulation_strategy,
        name='price_manipulation',
    )


def price_manipulation_multiple_blocks(
        pool_id: str):
    class PriceManipulationStrategy:
        def __init__(self):
            # self.trade_asset = asset2
            # self.attack_asset = asset1
            self.asset_sell_target = 0
            self.asset_sold = 0
            self.add_liquidity_target = 0
            self.liquidity_added = 0
            self.arb_trade_target = 0
            self.remove_liquidity_target = 0
            self.liquidity_removed = 0
            self.attack_asset = ''
            self.trade_asset = ''
            self.asset_pairs = []

        def execute(self, state: GlobalState, agent_id: str):
            omnipool: OmnipoolState = state.pools[pool_id]
            agent: Agent = state.agents[agent_id]

            if self.attack_asset == '':
                self.attack_asset = omnipool.asset_list[0]
                self.trade_asset = omnipool.asset_list[1]

            if (omnipool.unique_id, self.attack_asset) not in agent.holdings:
                agent.holdings[(omnipool.unique_id, self.attack_asset)] = 0

            if (
                    self.asset_sell_target == 0
                    and agent.holdings[(omnipool.unique_id, self.attack_asset)] == 0
            ):
                # the cycle begins
                self.remove_liquidity_target = 0

                # cycle between all assets
                trade_asset_index = omnipool.asset_list.index(self.trade_asset)
                attack_asset_index = omnipool.asset_list.index(self.attack_asset)
                self.trade_asset = omnipool.asset_list[(trade_asset_index + 1) % len(omnipool.asset_list)]
                if self.trade_asset == self.attack_asset:
                    self.attack_asset = (omnipool.asset_list[
                        (attack_asset_index + len(omnipool.asset_list) - 1)
                        % len(omnipool.asset_list)
                    ])
                self.asset_pairs.append({
                    'attack asset': self.attack_asset,
                    'trade asset': self.trade_asset,
                    'time step': state.time_step
                })
                if self.asset_pairs[-1]['attack asset'] == 'HDX' and self.asset_pairs[-1]['trade asset'] == 'DOT':
                    er = 1

                # target the maximum liquidity that will be allowed by the weight cap
                max_liquidity = (
                        (omnipool.weight_cap[self.attack_asset] * omnipool.lrna_total
                         - omnipool.lrna[self.attack_asset])
                        / (1 - omnipool.weight_cap[self.attack_asset])
                        / oamm.lrna_price(omnipool, self.trade_asset)
                ) if omnipool.weight_cap[self.attack_asset] < 1 else float('inf')

                # choose the largest trade size that will be allowed by the per block trade limit
                self.asset_sell_target = min(
                    agent.holdings[self.trade_asset],
                    omnipool.liquidity[self.trade_asset] / 2,
                    max_liquidity / 2
                )
                self.asset_pairs[-1]['sell target'] = self.asset_sell_target
                self.asset_sold = 0

            if self.asset_sell_target > 0:
                # make sure we don't go over any limits and get rejected
                sell_quantity = min(
                    omnipool.liquidity[self.trade_asset] * omnipool.trade_limit_per_block * .9999,
                    omnipool.calculate_sell_from_buy(
                        tkn_sell=self.trade_asset, tkn_buy=self.attack_asset,
                        buy_quantity=omnipool.liquidity[self.attack_asset] * omnipool.trade_limit_per_block
                    ) * .9999,
                    self.asset_sell_target - self.asset_sold
                )
                omnipool.swap(
                    agent=agent,
                    tkn_sell=self.trade_asset, tkn_buy=self.attack_asset,
                    sell_quantity=sell_quantity
                )
                self.asset_sold += sell_quantity
                if self.asset_sold >= self.asset_sell_target:
                    # we've sold enough - stop trying to sell
                    # switch gears and start adding liquidity
                    self.asset_sell_target = 0
                    # find the maximum liquidity we can actually add
                    max_liquidity = (
                            (omnipool.weight_cap[self.attack_asset] * omnipool.lrna_total
                             - omnipool.lrna[self.attack_asset])
                            / (1 - omnipool.weight_cap[self.attack_asset])
                            / oamm.lrna_price(omnipool, self.attack_asset)
                    ) if omnipool.weight_cap[self.attack_asset] < 1 else float('inf')
                    self.add_liquidity_target = min(
                        agent.holdings[self.attack_asset] / 2,
                        omnipool.liquidity[self.attack_asset] * 100,
                        max_liquidity * .9999
                    )
                    self.asset_pairs[-1]['add target'] = self.add_liquidity_target
                    self.liquidity_added = 0
                else:
                    # if we haven't sold enough, try again next block
                    return state

            if self.add_liquidity_target > 0:
                add_quantity = min(
                    self.add_liquidity_target,
                    omnipool.liquidity[self.attack_asset] * omnipool.trade_limit_per_block,
                    self.add_liquidity_target - self.liquidity_added
                )

                omnipool.add_liquidity(
                    agent=agent,
                    quantity=add_quantity,
                    tkn_add=self.attack_asset
                )

                self.liquidity_added += add_quantity
                if self.liquidity_added >= self.add_liquidity_target:
                    # we've added enough liquidity - stop trying to add liquidity
                    # switch gears and start arbitraging
                    self.add_liquidity_target = 0
                    self.arb_trade_target = 1
                    self.asset_pairs[-1]['arb target'] = self.arb_trade_target

            if self.arb_trade_target > 0:
                # here we want to bring the prices back in line with the market
                tkn_buy = self.trade_asset
                tkn_sell = self.attack_asset
                delta_r = (math.sqrt((
                    omnipool.lrna[tkn_sell] * omnipool.lrna[tkn_buy]
                    * omnipool.liquidity[tkn_sell] * omnipool.liquidity[tkn_buy]
                ) / (state.external_market[tkn_sell] / state.external_market[tkn_buy])) - (
                    omnipool.lrna[tkn_buy] * omnipool.liquidity[tkn_sell]
                )) / (omnipool.lrna[tkn_sell] + omnipool.lrna[tkn_buy])

                # find the max that we can (or want to) actually sell
                sell_quantity = min(
                    omnipool.liquidity[self.attack_asset] * omnipool.trade_limit_per_block,
                    omnipool.calculate_sell_from_buy(
                        tkn_sell=tkn_sell, tkn_buy=tkn_buy,
                        buy_quantity=omnipool.liquidity[self.trade_asset] * omnipool.trade_limit_per_block
                    ) * .9999,
                    delta_r,
                    agent.holdings[self.attack_asset]
                )

                omnipool.swap(
                    agent=agent,
                    tkn_sell=self.attack_asset, tkn_buy=self.trade_asset,
                    sell_quantity=sell_quantity
                )
                if state.time_step > 85:
                    er = 1
                if (
                        oamm.usd_price(omnipool, self.attack_asset) / oamm.usd_price(omnipool, self.trade_asset)
                        <= state.external_market[self.attack_asset] / state.external_market[self.trade_asset] * 1.00001
                        or agent.holdings[self.attack_asset] == 0
                ):
                    # we're good, stop trying to arbitrage
                    self.arb_trade_target = 0
                    # now we want to remove all the liquidity we added
                    self.remove_liquidity_target = agent.holdings[(omnipool.unique_id, self.attack_asset)]
                    self.asset_pairs[-1]['remove target'] = self.remove_liquidity_target
                    self.liquidity_removed = 0

            if self.remove_liquidity_target > 0:
                # try and get it all, although it might take multiple blocks
                remove_quantity = min(
                    omnipool.max_withdrawal_per_block * omnipool.shares[self.attack_asset],
                    agent.holdings[(omnipool.unique_id, self.attack_asset)]
                )
                omnipool.remove_liquidity(
                    agent=agent,
                    quantity=remove_quantity,
                    tkn_remove=self.attack_asset
                )
                self.liquidity_removed += remove_quantity

            agent.attack_history = self.asset_pairs
            return state

    return TradeStrategy(
        strategy_function=PriceManipulationStrategy().execute,
        name='price manipulation',
    )


def price_sensitive_trading(
        pool_id: str,
        max_volume_usd: float,
        price_sensitivity: float,
        tkn_sell: str = None,
        tkn_buy: str = None,
        trade_frequency: float = 0.1
) -> TradeStrategy:
    def strategy(state: GlobalState, agent_id: str) -> GlobalState:
        if random.random() > trade_frequency:
            return state
        agent: Agent = state.agents[agent_id]
        pool: OmnipoolState = state.pools[pool_id]
        options = list(set(agent.asset_list) & set(pool.asset_list))
        sell = tkn_sell
        buy = tkn_buy
        if tkn_sell is None:
            sell = random.choice(options)
            options.remove(sell)
        if tkn_buy is None:
            buy = random.choice(options)
        slip_rate = (
            pool.calculate_sell_from_buy(buy, sell, max_volume_usd / state.external_market[buy])
            / (state.external_market[buy] / state.external_market[sell])
            / (max_volume_usd / state.external_market[buy])
        ) - 1  # find the price of buying from the pool vs. buying from the market
        trade_volume = max(
            max_volume_usd / state.external_market[sell] * max(min(1 - price_sensitivity * slip_rate, 1), 0) ** 10,
            0
        ) / trade_frequency
        pool.swap(
            agent=agent,
            tkn_sell=sell,
            tkn_buy=buy,
            sell_quantity=trade_volume,
        )
        return state

    return TradeStrategy(strategy, name=f'price sensitive trading (sensitivity ={price_sensitivity})')
