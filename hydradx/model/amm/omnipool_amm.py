import copy
from .agents import Agent
from .amm import AMM, FeeMechanism
from .stableswap_amm import StableSwapPoolState
from mpmath import mpf, mp

mp.dps = 50


class OmnipoolState(AMM):
    unique_id: str = 'omnipool'

    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = "USD",
                 asset_fee: FeeMechanism or float = 0,
                 lrna_fee: FeeMechanism or float = 0
                 ):
        """
        tokens should be a dict in the form of [str: dict]
        the nested dict needs the following parameters:
        {
          'liquidity': float  # starting risk asset liquidity in the pool
          (
          'LRNA': float  # starting LRNA on the other side of the pool
          or
          'LRNA_price': float  # price of the asset denominated in LRNA
          )

          optional:
          'weight_cap': float  # maximum fraction of TVL that may be held in this pool
        }
        """

        super().__init__()

        if 'HDX' not in tokens:
            raise ValueError('HDX not included in tokens.')
        if preferred_stablecoin not in tokens:
            raise ValueError(f'{preferred_stablecoin} is preferred stablecoin, but not included in tokens.')

        self.asset_list: list[str] = []
        self.liquidity = {}
        self.lrna = {}
        self.shares = {}
        self.protocol_shares = {}
        self.weight_cap = {}
        for token, pool in tokens.items():
            assert pool['liquidity'], f'token {token} missing required parameter: liquidity'
            self.asset_list.append(token)
            self.liquidity[token] = mpf(pool['liquidity'])
            self.shares[token] = mpf(pool['liquidity'])
            self.protocol_shares[token] = mpf(pool['liquidity'])
            self.weight_cap[token] = mpf(pool['weight_cap'] if 'weight_cap' in pool else 1)
            if 'LRNA' in pool:
                self.lrna[token] = mpf(pool['LRNA'])
            elif 'LRNA_price' in pool:
                self.lrna[token] = mpf(pool['liquidity'] * pool['LRNA_price'])
            else:
                raise ValueError("token {name} missing required parameter: ('LRNA' or 'LRNA_price)")

        self.asset_fee: FeeMechanism = asset_fee.assign(self) if isinstance(asset_fee, FeeMechanism) \
            else self.basic_fee(asset_fee).assign(self)
        self.lrna_fee: FeeMechanism = lrna_fee.assign(self) if isinstance(lrna_fee, FeeMechanism) \
            else self.basic_fee(lrna_fee).assign(self)
        self.lrna_imbalance = mpf(0)  # AKA "L"
        self.tvl_cap = tvl_cap
        self.stablecoin = preferred_stablecoin
        self.fail = ''
        self.sub_pools = {}  # require sub_pools to be added through create_sub_pool

    def price(self, i: str, j: str = ''):
        """
        price of an asset i denominated in j, according to current market conditions in the omnipool
        """
        j = j if j in self.liquidity else self.stablecoin
        if self.liquidity[i] == 0:
            return 0
        return self.lrna[i] / self.liquidity[i] / self.lrna[j] * self.liquidity[j]

    def lrna_price(self, tkn) -> float:
        """
        price of asset i in LRNA
        """
        return self.lrna[tkn] / self.liquidity[tkn]

    @property
    def lrna_total(self):
        return sum(self.lrna.values())

    @property
    def tvl_total(self):
        # base this just on the LRNA/USD exchange rate in the pool
        return self.liquidity[self.stablecoin] * self.lrna[self.stablecoin] / self.lrna_total

    def copy(self):
        copy_state = copy.deepcopy(self)
        copy_state.fail = ''
        return copy_state

    @staticmethod
    def slip_fee(slip_factor: float, minimum_fee: float = 0) -> FeeMechanism:
        def fee_function(
                exchange: AMM, tkn: str, delta_tkn: float
        ) -> float:
            return (slip_factor * abs(delta_tkn) / (exchange.liquidity[tkn] + delta_tkn)) + minimum_fee

        return FeeMechanism(fee_function, f"Slip fee (alpha={slip_factor}, min={minimum_fee}")

    def __repr__(self):
        # don't go overboard with the precision here
        precision = 10
        lrna = {tkn: round(self.lrna[tkn], precision) for tkn in self.lrna}
        lrna_total = round(self.lrna_total, precision)
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.liquidity}
        weight_cap = {tkn: round(self.weight_cap[tkn], precision) for tkn in self.weight_cap}
        price = {tkn: round(self.price(tkn), precision) for tkn in self.asset_list}
        return (
                   f'Omnipool: {self.unique_id}\n'
                   f'********************************\n'
                   f'tvl cap: {self.tvl_cap}\n'
                   f'lrna fee: {self.lrna_fee.name}\n'
                   f'asset fee: {self.asset_fee.name}\n'
                   f'asset pools: (\n\n'
               ) + '\n'.join(
            [(
                    f'    *{token}*\n'
                    f'    asset quantity: {liquidity[token]}\n'
                    f'    lrna quantity: {lrna[token]}\n'
                    f'    USD price: {price[token]}\n' +
                    f'    tvl: {lrna[token] * liquidity[self.stablecoin] / lrna[self.stablecoin]}\n'
                    f'    weight: {lrna[token]}/{lrna_total} ({lrna[token] / lrna_total})\n'
                    f'    weight cap: {weight_cap[token]}\n'
                    f'    total shares: {self.shares[token]}\n'
                    f'    protocol shares: {self.protocol_shares[token]}\n'
            ) for token in self.asset_list]
        ) + '\n)\n' + f'sub pools: (\n\n    ' + ')\n(\n'.join(
            [
                '\n    '.join(pool_desc.split('\n'))
                for pool_desc in
                [repr(pool) for pool in self.sub_pools.values()]
            ]
        ) + '\n)'

    def calculate_sell_from_buy(
            self,
            tkn_buy: str,
            tkn_sell: str,
            buy_quantity: float
    ):
        """
        Given a buy quantity, calculate the effective price, so we can execute it as a sell
        """
        asset_fee = self.asset_fee.compute(tkn=tkn_buy, delta_tkn=-buy_quantity)
        delta_Qj = self.lrna[tkn_buy] * buy_quantity / (
                self.liquidity[tkn_buy] * (1 - asset_fee) - buy_quantity)
        lrna_fee = self.lrna_fee.compute(tkn=tkn_sell, delta_tkn=(
                self.liquidity[tkn_buy] * delta_Qj /
                (self.lrna[tkn_buy] - delta_Qj)
        ))
        delta_Qi = -delta_Qj / (1 - lrna_fee)
        delta_Ri = -self.liquidity[tkn_sell] * delta_Qi / (self.lrna[tkn_sell] + delta_Qi)
        return delta_Ri

    def get_sub_pool(self, tkn: str):
        # if asset in not in omnipool, return the ID of the sub_pool where it can be found
        if tkn in self.asset_list:
            return ''
        else:
            for pool in self.sub_pools.values():
                if tkn in pool.asset_list:
                    return pool.unique_id

    def execute_swap(
            self, agent: Agent,
            tkn_buy: str, tkn_sell: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        """
        execute swap in place (modify and return self and agent)
        """

        if tkn_buy not in self.asset_list + ['LRNA'] or tkn_sell not in self.asset_list + ['LRNA']:
            # note: this default routing behavior assumes that an asset will only exist in one place in the omnipool
            return self.execute_stable_swap(
                agent=agent,
                sub_pool_buy_id=self.get_sub_pool(tkn_buy),
                sub_pool_sell_id=self.get_sub_pool(tkn_sell),
                tkn_sell=tkn_sell, tkn_buy=tkn_buy,
                buy_quantity=buy_quantity,
                sell_quantity=sell_quantity
            )

        if tkn_sell == 'LRNA':
            return self.execute_lrna_swap(
                agent=agent,
                delta_ra=buy_quantity,
                delta_qa=-sell_quantity,
                tkn=tkn_buy
            )
        elif tkn_buy == 'LRNA':
            raise ValueError('Buying LRNA not implemented.')

        if buy_quantity:
            # back into correct delta_Ri, then execute sell
            delta_Ri = self.calculate_sell_from_buy(tkn_buy, tkn_sell, buy_quantity)
            return self.execute_swap(
                agent=agent,
                tkn_buy=tkn_buy,
                tkn_sell=tkn_sell,
                sell_quantity=delta_Ri
            )

        i = tkn_sell
        j = tkn_buy
        delta_Ri = sell_quantity
        if delta_Ri <= 0:
            return self.fail_transaction('sell amount must be greater than zero', agent)

        delta_Qi = self.lrna[i] * -delta_Ri / (self.liquidity[i] + delta_Ri)
        asset_fee = self.asset_fee.compute(tkn=tkn_sell, delta_tkn=sell_quantity)
        lrna_fee = self.lrna_fee.compute(
            tkn=tkn_buy,
            delta_tkn=self.liquidity[j] * sell_quantity / (self.lrna[j] + sell_quantity) * (1 - asset_fee)
        )

        delta_Qj = -delta_Qi * (1 - lrna_fee)
        delta_Rj = self.liquidity[j] * -delta_Qj / (self.lrna[j] + delta_Qj) * (1 - asset_fee)
        delta_L = min(-delta_Qi * lrna_fee, -self.lrna_imbalance)
        delta_QH = -lrna_fee * delta_Qi - delta_L

        if self.liquidity[i] + sell_quantity > 10 ** 12:
            return self.fail_transaction('Asset liquidity cannot exceed 10 ^ 12.', agent)

        if agent.holdings[i] < sell_quantity:
            return self.fail_transaction(f"Agent doesn't have enough {i}", agent)

        self.lrna[i] += delta_Qi
        self.lrna[j] += delta_Qj
        self.liquidity[i] += delta_Ri
        self.liquidity[j] += delta_Rj
        self.lrna['HDX'] += delta_QH
        self.lrna_imbalance += delta_L

        if j not in agent.holdings:
            agent.holdings[j] = 0
        agent.holdings[i] -= delta_Ri
        agent.holdings[j] -= delta_Rj

        return self, agent

    def execute_lrna_swap(
            self,
            agent: Agent,
            delta_ra: float = 0,
            delta_qa: float = 0,
            tkn: str = ''
    ):
        """
        Execute LRNA swap in place (modify and return)
        """
        asset_fee = self.asset_fee.compute(
            tkn=tkn, delta_tkn=delta_ra or self.liquidity[tkn] * delta_qa / (delta_qa + self.lrna[tkn])
        )

        if delta_qa < 0:
            delta_Q = -delta_qa
            delta_R = self.liquidity[tkn] * -delta_Q / (delta_Q + self.lrna[tkn]) * (1 - asset_fee)
            delta_ra = -delta_R
        elif delta_ra > 0:
            delta_R = -delta_ra
            delta_Q = self.lrna[tkn] * -delta_R / (self.liquidity[tkn] * (1 - asset_fee) + delta_R)
            delta_qa = -delta_Q
        else:
            return self.fail_transaction('Buying LRNA not implemented.', agent)

        if delta_qa + agent.holdings['LRNA'] < 0:
            return self.fail_transaction("agent doesn't have enough lrna", agent)
        elif delta_ra + agent.holdings[tkn] < 0:
            return self.fail_transaction(f"agent doesn't have enough {tkn} holdings", agent)
        elif delta_R + self.liquidity[tkn] <= 0:
            return self.fail_transaction('insufficient assets in pool', agent)
        elif delta_Q + self.lrna[tkn] <= 0:
            return self.fail_transaction('insufficient lrna in pool', agent)

        agent.holdings['LRNA'] += delta_qa
        agent.holdings[tkn] += delta_ra
        old_lrna = self.lrna[tkn]
        old_liquidity = self.liquidity[tkn]
        l = self.lrna_imbalance
        q = self.lrna_total
        self.lrna[tkn] += delta_Q
        self.liquidity[tkn] += delta_R
        self.lrna_imbalance = (
                self.lrna_total * self.liquidity[tkn] / self.lrna[tkn]
                * old_lrna / old_liquidity
                * (1 + l / q) - self.lrna_total
        )
        return self, agent

    def execute_stable_swap(
            self, agent: Agent,
            tkn_sell: str, tkn_buy: str,
            sub_pool_buy_id: str = "",
            sub_pool_sell_id: str = "",
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ) -> tuple[AMM, Agent]:

        if tkn_sell == 'LRNA':
            if buy_quantity:
                sub_pool = self.sub_pools[sub_pool_buy_id]
                # buy a specific quantity of a stableswap asset using LRNA
                shares_needed = sub_pool.calculate_withdrawal_shares(tkn_remove=tkn_buy, quantity=buy_quantity)
                self.execute_lrna_swap(agent, delta_ra=shares_needed, tkn=sub_pool.unique_id)
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail, agent)
                sub_pool.execute_withdraw_asset(agent, buy_quantity, tkn_buy)
                return self, agent
            elif sell_quantity:
                sub_pool = self.sub_pools[sub_pool_buy_id]
                agent_shares = agent.holdings[sub_pool.unique_id]
                self.execute_swap(
                    agent=agent,
                    tkn_buy=sub_pool.unique_id, tkn_sell='LRNA',
                    sell_quantity=sell_quantity
                )
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail, agent)
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                sub_pool.execute_remove_liquidity(agent, delta_shares, tkn_buy)
                return self, agent
        elif sub_pool_sell_id and tkn_buy in self.asset_list:
            sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_sell_id]
            if sell_quantity:
                # sell a stableswap asset for an omnipool asset
                agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                sub_pool.execute_add_liquidity(agent, sell_quantity, tkn_sell)
                if self.fail:
                    # the transaction failed.
                    return self.fail_transaction(self.fail, agent)
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                self.execute_swap(
                    agent=agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=sub_pool.unique_id,
                    sell_quantity=delta_shares
                )
                return self, agent
            elif buy_quantity:
                # buy an omnipool asset with a stableswap asset
                sell_shares = self.calculate_sell_from_buy(tkn_buy, sub_pool.unique_id, buy_quantity)
                if sell_shares < 0:
                    return self.fail_transaction("Not enough liquidity in the stableswap/LRNA pool.", agent)
                sub_pool.execute_buy_shares(agent, sell_shares, tkn_sell)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail, agent)
                self.execute_swap(agent, tkn_buy, sub_pool.unique_id, buy_quantity)
                return self, agent
        elif sub_pool_buy_id and tkn_sell in self.asset_list:
            sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_buy_id]
            if buy_quantity:
                # buy a stableswap asset with an omnipool asset
                shares_traded = sub_pool.calculate_withdrawal_shares(tkn_buy, buy_quantity)

                # buy shares in the subpool
                self.execute_swap(agent, tkn_buy=sub_pool.unique_id, tkn_sell=tkn_sell, buy_quantity=shares_traded)
                if self.fail:
                    # if the swap failed, the transaction failed.
                    return self.fail_transaction(self.fail, agent)
                # withdraw the shares for the desired token
                sub_pool.execute_withdraw_asset(agent, quantity=buy_quantity, tkn_remove=tkn_buy)
                if sub_pool.fail:
                    return self.fail_transaction(sub_pool.fail, agent)
                return self, agent
            elif sell_quantity:
                # sell an omnipool asset for a stableswap asset
                agent_shares = agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in agent.holdings else 0
                self.execute_swap(
                    agent=agent,
                    tkn_buy=sub_pool.unique_id,
                    tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )
                delta_shares = agent.holdings[sub_pool.unique_id] - agent_shares
                if self.fail:
                    return self.fail_transaction(self.fail, agent)
                sub_pool.execute_remove_liquidity(
                    agent=agent, shares_removed=delta_shares, tkn_remove=tkn_buy
                )
                return self, agent
        elif sub_pool_buy_id and sub_pool_sell_id:
            # trade between two subpools
            pool_buy: StableSwapPoolState = self.sub_pools[sub_pool_buy_id]
            pool_sell: StableSwapPoolState = self.sub_pools[sub_pool_sell_id]
            if buy_quantity:
                # buy enough shares of tkn_sell to afford buy_quantity worth of tkn_buy
                shares_bought = pool_buy.calculate_withdrawal_shares(tkn_buy, buy_quantity)
                if shares_bought > pool_buy.liquidity[tkn_buy]:
                    return self.fail_transaction(f'Not enough liquidity in {pool_buy.unique_id}: {tkn_buy}.', agent)
                shares_sold = self.calculate_sell_from_buy(
                    tkn_buy=pool_buy.unique_id,
                    tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                pool_sell.execute_buy_shares(
                    agent=agent, quantity=shares_sold,
                    tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail, agent)
                self.execute_swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    buy_quantity=shares_bought
                )
                if self.fail:
                    return self.fail_transaction(self.fail, agent)
                pool_buy.execute_withdraw_asset(
                    agent=agent, quantity=buy_quantity,
                    tkn_remove=tkn_buy, fail_on_overdraw=False
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail, agent)

                # if all three parts succeeded, then we're good!
                return self, agent
            elif sell_quantity:
                agent_sell_holdings = agent.holdings[sub_pool_sell_id] if sub_pool_sell_id in agent.holdings else 0
                pool_sell.execute_add_liquidity(
                    agent=agent, quantity=sell_quantity, tkn_add=tkn_sell
                )
                if pool_sell.fail:
                    return self.fail_transaction(pool_sell.fail, agent)
                delta_sell_holdings = agent.holdings[sub_pool_sell_id] - agent_sell_holdings
                agent_buy_holdings = agent.holdings[sub_pool_buy_id] if sub_pool_buy_id in agent.holdings else 0
                self.execute_swap(
                    agent=agent,
                    tkn_buy=pool_buy.unique_id, tkn_sell=pool_sell.unique_id,
                    sell_quantity=delta_sell_holdings
                )
                if self.fail:
                    return self.fail_transaction(self.fail, agent)
                delta_buy_holdings = agent.holdings[sub_pool_buy_id] - agent_buy_holdings
                pool_buy.execute_remove_liquidity(
                    agent=agent, shares_removed=delta_buy_holdings, tkn_remove=tkn_buy
                )
                if pool_buy.fail:
                    return self.fail_transaction(pool_buy.fail, agent)
                return self, agent
        else:
            raise ValueError('buy_quantity or sell_quantity must be specified.')

    def execute_create_sub_pool(
            self,
            tkns_migrate: list[str],
            sub_pool_id: str,
            amplification: float,
            trade_fee: FeeMechanism or float = 0
    ):
        new_sub_pool = StableSwapPoolState(
            tokens={tkn: self.liquidity[tkn] for tkn in tkns_migrate},
            amplification=amplification,
            unique_id=sub_pool_id,
            trade_fee=trade_fee
        )
        new_sub_pool.conversion_metrics = {
            tkn: {
                'price': self.lrna_price(tkn),
                'old_shares': self.shares[tkn],
                'omnipool_shares': self.lrna[tkn],
                'subpool_shares': self.lrna[tkn]
            } for tkn in tkns_migrate
        }
        new_sub_pool.shares = sum([self.lrna[tkn] for tkn in tkns_migrate])
        self.sub_pools[sub_pool_id] = new_sub_pool
        self.asset_list.append(sub_pool_id)
        self.liquidity[sub_pool_id] = sum([self.lrna[tkn] for tkn in tkns_migrate])
        self.shares[sub_pool_id] = sum([self.lrna[tkn] for tkn in tkns_migrate])
        self.lrna[sub_pool_id] = sum([self.lrna[tkn] for tkn in tkns_migrate])
        self.weight_cap[sub_pool_id] = 1
        self.protocol_shares[sub_pool_id] = sum([
            self.lrna[tkn] * self.protocol_shares[tkn] / self.shares[tkn] for tkn in tkns_migrate
        ])
        # remove assets from Omnipool
        for tkn in tkns_migrate:
            self.liquidity[tkn] = 0
            self.lrna[tkn] = 0
            self.asset_list.remove(tkn)
        return self

    def execute_migrate_asset(self, tkn_migrate: str, sub_pool_id: str):
        """
        Move an asset from the Omnipool into a stableswap subpool.
        """
        sub_pool: StableSwapPoolState = self.sub_pools[sub_pool_id]
        s = sub_pool.unique_id
        i = tkn_migrate
        if tkn_migrate in sub_pool.liquidity:
            raise AssertionError('Assets should only exist in one place in the Omnipool at a time.')
        sub_pool.liquidity[i] = self.liquidity[i]
        self.protocol_shares[s] += (
            self.shares[s] * self.lrna[i] / self.lrna[s] * self.protocol_shares[i] / self.shares[i]
        )

        sub_pool.conversion_metrics[i] = {
            'price': self.lrna[i] / self.lrna[s] * sub_pool.shares / self.liquidity[i],
            'old_shares': self.shares[i],
            'omnipool_shares': self.lrna[i] * self.shares[s] / self.lrna[s],
            'subpool_shares': self.lrna[i] * sub_pool.shares / self.lrna[s]
        }

        self.shares[s] += self.lrna[i] * self.shares[s] / self.lrna[s]
        self.liquidity[s] += self.lrna[i] * sub_pool.shares / self.lrna[s]
        sub_pool.shares += self.lrna[i] * sub_pool.shares / self.lrna[s]
        self.lrna[s] += self.lrna[i]

        # remove asset from omnipool and add it to subpool
        self.lrna[i] = 0
        self.liquidity[i] = 0
        self.asset_list.remove(i)
        sub_pool.asset_list.append(i)
        return self

    def execute_migrate_lp(
        self,
        agent: Agent,
        sub_pool_id: str,
        tkn_migrate: str
    ):
        sub_pool = self.sub_pools[sub_pool_id]
        conversions = sub_pool.conversion_metrics[tkn_migrate]
        old_pool_id = (self.unique_id, tkn_migrate)
        old_share_price = agent.share_prices[old_pool_id]
        # TODO: maybe this is an edge case or not allowed, but what if the agent already has a share price locked in?
        # ex., maybe they have LPed into the new subpool after their asset was migrated,
        # but before they had migrated their own position
        agent.share_prices[sub_pool_id] = old_share_price / conversions['price']
        if sub_pool_id not in agent.holdings:
            agent.holdings[sub_pool_id] = 0
        agent.holdings[sub_pool_id] += (
            agent.holdings[old_pool_id] / conversions['old_shares'] * conversions['omnipool_shares']
        )
        self.liquidity[sub_pool_id] += (
            agent.holdings[old_pool_id] / conversions['old_shares'] * conversions['subpool_shares']
        )  # frac{s_\alpha}{S_i}\Delta U_s
        agent.holdings[old_pool_id] = 0

        return self, agent


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def lrna_price(state: OmnipoolState, i: str, fee: float = 0) -> float:
    """Price of i denominated in LRNA"""
    if state.liquidity[i] == 0:
        return 0
    else:
        return (state.lrna[i] / state.liquidity[i]) * (1 - fee)


def swap_lrna(
        old_state: OmnipoolState,
        old_agent: Agent,
        delta_ra: float = 0,
        delta_qa: float = 0,
        tkn: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after LRNA swap"""

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    return new_state.execute_lrna_swap(new_agent, delta_ra, delta_qa, tkn)


def swap(
        old_state: OmnipoolState,
        old_agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
) -> tuple[OmnipoolState, Agent]:
    """
    execute swap on a copy of old_state and old_agent, and return the copies
    """
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.execute_swap(
        agent=new_agent,
        sell_quantity=sell_quantity,
        buy_quantity=buy_quantity,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
    )

    return new_state, new_agent


def migrate(
        old_state: OmnipoolState,
        tkn_migrate: str,
        sub_pool_id: str
) -> OmnipoolState:
    return old_state.copy().execute_migrate_asset(tkn_migrate, sub_pool_id)


def add_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent = None,
        quantity: float = 0,
        tkn_add: str = ''
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity addition"""

    new_state = old_state.copy()
    new_agent = old_agent.copy()

    # assert quantity > 0, f"delta_R must be positive: {quantity}"
    if tkn_add not in old_state.asset_list:
        for sub_pool in new_state.sub_pools.values():
            if tkn_add in sub_pool.asset_list:
                sub_pool.execute_add_liquidity(
                    agent=new_agent,
                    quantity=quantity,
                    tkn_add=tkn_add
                )
            # deposit into the Omnipool
            return add_liquidity(
                new_state, new_agent,
                quantity=(new_agent.holdings[sub_pool.unique_id] -
                          (old_agent.holdings[sub_pool.unique_id] if sub_pool.unique_id in old_agent.holdings else 0)),
                tkn_add=sub_pool.unique_id
            )
        raise AssertionError(f"invalid value for i: {tkn_add}")

    # Token amounts update
    new_state.liquidity[tkn_add] += quantity

    if old_agent:
        new_agent.holdings[tkn_add] -= quantity
        if new_agent.holdings[tkn_add] < 0:
            return old_state.fail_transaction('Transaction rejected because agent has insufficient funds.', old_agent)

    # Share update
    if new_state.shares[tkn_add]:
        new_state.shares[tkn_add] *= new_state.liquidity[tkn_add] / old_state.liquidity[tkn_add]
    else:
        new_state.shares[tkn_add] = new_state.liquidity[tkn_add]

    if old_agent:
        # shares go to provisioning agent
        if not (new_state.unique_id, tkn_add) in new_agent.holdings:
            new_agent.holdings[(new_state.unique_id, tkn_add)] = 0
        new_agent.holdings[(new_state.unique_id, tkn_add)] += new_state.shares[tkn_add] - old_state.shares[tkn_add]
    else:
        # shares go to protocol
        new_state.protocol_shares[tkn_add] += new_state.shares[tkn_add] - old_state.shares[tkn_add]

    # LRNA add (mint)
    delta_Q = lrna_price(old_state, tkn_add) * quantity
    new_state.lrna[tkn_add] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = (
            quantity * old_state.lrna[tkn_add] / old_state.liquidity[tkn_add]
            * old_state.lrna_imbalance / old_state.lrna_total
    )
    new_state.lrna_imbalance += delta_L

    if new_state.lrna[tkn_add] / new_state.lrna_total > new_state.weight_cap[tkn_add]:
        return old_state.fail_transaction(
            'Transaction rejected because it would exceed the weight cap in pool[{i}].', old_agent
        )

    if new_state.tvl_total > new_state.tvl_cap:
        return old_state.fail_transaction('Transaction rejected because it would exceed the TVL cap.', old_agent)

    if new_state.liquidity[tkn_add] > 10 ** 12:
        return old_state.fail_transaction('Asset liquidity cannot exceed 10 ^ 12.', old_agent)

    # set price at which liquidity was added
    if old_agent:
        new_agent.share_prices[(new_state.unique_id, tkn_add)] = new_state.lrna_price(tkn_add)

    return new_state, new_agent


def remove_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity removal"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    if quantity == 0:
        return new_state, new_agent

    if tkn_remove not in new_state.asset_list:
        for sub_pool in new_state.sub_pools.values():
            if tkn_remove in sub_pool.asset_list:
                sub_pool.execute_remove_liquidity(
                    new_agent, quantity, tkn_remove
                )
                if sub_pool.fail:
                    return old_state.fail_transaction(sub_pool.fail, old_agent)
                return new_state, new_agent

        raise AssertionError(f"invalid value for i: {tkn_remove}")

    quantity = -abs(quantity)
    assert quantity <= 0, f"delta_S cannot be positive: {quantity}"
    assert tkn_remove in old_state.asset_list, f"invalid token name: {tkn_remove}"

    # determine if they should get some LRNA back as well as the asset they invested
    piq = old_state.lrna_price(tkn_remove)
    p0 = new_agent.share_prices[(new_state.unique_id, tkn_remove)]
    mult = (piq - p0) / (piq + p0)

    # Share update
    delta_B = max(mult * quantity, 0)
    new_state.protocol_shares[tkn_remove] += delta_B
    new_state.shares[tkn_remove] += quantity + delta_B
    new_agent.holdings[(new_state.unique_id, tkn_remove)] += quantity

    # Token amounts update
    delta_R = old_state.liquidity[tkn_remove] * max((quantity + delta_B) / old_state.shares[tkn_remove], -1)
    new_state.liquidity[tkn_remove] += delta_R
    new_agent.holdings[tkn_remove] -= delta_R
    if piq >= p0:  # prevents rounding errors
        if 'LRNA' not in new_agent.holdings:
            new_agent.holdings['LRNA'] = 0
        new_agent.holdings['LRNA'] -= piq * (
                2 * piq / (piq + p0) * quantity / old_state.shares[tkn_remove]
                * old_state.liquidity[tkn_remove] - delta_R
        )

    # LRNA burn
    delta_Q = lrna_price(old_state, tkn_remove) * delta_R
    new_state.lrna[tkn_remove] += delta_Q

    # L update: LRNA fees to be burned before they will start to accumulate again
    delta_L = (
            delta_R * old_state.lrna[tkn_remove] / old_state.liquidity[tkn_remove]
            * old_state.lrna_imbalance / old_state.lrna_total
    )
    new_state.lrna_imbalance += delta_L

    return new_state, new_agent


OmnipoolState.swap = staticmethod(swap)
OmnipoolState.add_liquidity = staticmethod(add_liquidity)
OmnipoolState.remove_liquidity = staticmethod(remove_liquidity)
