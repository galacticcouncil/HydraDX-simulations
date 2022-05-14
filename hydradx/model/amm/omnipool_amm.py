import copy
import math

from . import amm
from .amm import TradeStrategy, Asset
import random


class OmnipoolRiskAssetPool(amm.RiskAssetPool):

    @property
    def name(self) -> str:
        return self.positions[0].assetName

    @property
    def lrnaQuantity(self) -> float:
        return self.positions[1].quantity

    @lrnaQuantity.setter
    def lrnaQuantity(self, value):
        self.positions[1].quantity = value

    @property
    def asset(self) -> amm.Asset:
        return self.positions[0].asset

    @property
    def assetName(self) -> str:
        return self.positions[0].asset.name

    @property
    def assetQuantity(self) -> float:
        return self.positions[0].quantity

    @assetQuantity.setter
    def assetQuantity(self, value):
        self.positions[0].quantity = value


class OmnipoolAgent(amm.Agent):

    # noinspection PyTypeChecker
    def add_liquidity(self, market: amm.Exchange, asset: str or Asset, quantity: float):
        """ add the specified quantity of asset to the exchange pool """
        if not self.position(asset):
            self.add_position(asset, quantity)
            self.add_position(market.pool(asset).shareToken.name, 0)
        market.add_liquidity(agent=self, pool=market.pool(asset), quantity=quantity)
        return self

    def remove_liquidity(self, market: amm.Exchange, asset: str or Asset, quantity: float):
        """ sell off the specified quantity of asset from the exchange """
        market.remove_liquidity(agent=self, pool=market.pool(asset.name), quantity=quantity)
        return self

    def remove_all_liquidity(self, market: amm.Exchange):
        """ sell off all assets that are in the exchange """
        for pool_asset in filter(lambda asset: self.position(asset), self.pool_asset_list):
            self.remove_liquidity(market, pool_asset, self.holdings(pool_asset))
        return self

    def value_holdings(self, market: amm.Exchange) -> float:
        """ returns the value of all this agent's combined holdings, denominated in LRNA """
        return sum(
            self.holdings(asset) * market.pool(asset).ratio
            if market.pool(asset)
            else self.holdings(asset)
            for asset in self.asset_list
        )

    def pool_asset(self, asset: str or Asset) -> str:
        asset_name = self.asset(asset).name
        for asset in self.pool_asset_list:
            if asset.name == amm.ShareToken.token_name(asset_name):
                return asset.name

    @property
    def pool_asset_list(self) -> list[amm.ShareToken]:
        return_list = []
        for asset in self.asset_list:
            if isinstance(asset, amm.ShareToken):
                return_list.append(asset)
        return return_list

    def s(self, asset: str or Asset) -> float:
        """ quantity of shares in pool[index] owned by the agent """
        return self.holdings(self.pool_asset(asset))

    def r(self, asset: str or Asset) -> float:
        """ quantity of asset[index] owned by the agent external to the omnipool """
        return self.holdings(asset)

    def add_delta_r(self, asset: str or Asset, value: float):
        """ add value to agent's holding in asset """
        self.add_position(asset, value)

    def add_delta_s(self, asset: str or Asset, value: float):
        """ change quantity of shares in pool[index] owned by the agent """
        self.position(self.pool_asset(asset)).quantity += value

    def p(self, asset: str or Asset) -> float:
        """ price at which the agent's holdings in pool[index] were acquired """
        return self.price(self.pool_asset(asset))

    def set_p(self, asset: str or Asset, value: float):
        self.position(self.pool_asset(asset)).buy_in_price = value

    @property
    def q(self):
        """ quantity of LRNA held by the agent """
        return self.holdings('LRNA')

    def add_delta_q(self, value: float):
        self.add_position('LRNA', quantity=value)


class Omnipool(amm.Exchange):
    lrnaFee = 0
    assetFee = 0
    lrnaImbalance = 0
    preferred_stablecoin = 'USD'

    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float = 0,
                 lrna_fee: float = 0,
                 preferred_stablecoin: str = 'USD'
                 ):
        """ The Omnipool class. Derives from amm.Exchange """
        super().__init__(tvl_cap_usd, asset_fee or self.assetFee)
        self.lrnaFee = lrna_fee or self.lrnaFee
        self.preferred_stablecoin = preferred_stablecoin or self.preferred_stablecoin

    def algebraic_symbols(self):
        """ usage: P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = [OmniPool].algebraic_symbols() """
        return self.P, self.Q, self.R, self.S, self.T, self.L, self.lrnaFee, self.assetFee, self.Q_total, self.T_total

    def __repr__(self):
        return (
            f'Omnipool\n'
            f'tvl cap: {self.tvlCapUSD}\n'
            f'lrna fee: {self.lrnaFee}\n'
            f'asset fee: {self.assetFee}\n'
            f'asset pools: (\n'
        ) + ')\n(\n'.join(
            [(
                f'    {pool.name}\n'
                f'    asset quantity: {pool.assetQuantity}\n'
                f'    lrna quantity: {pool.lrnaQuantity}\n'
                f'    usd price: {pool.asset.price}\n'
                f'    weight: {pool.totalValue}/{self.T_total} ({pool.totalValue / self.T_total})\n'
                f'    weight cap: {pool.weightCap}\n'
                f'    total shares: {pool.shares}\n'
                f'    protocol shares: {pool.sharesOwnedByProtocol}\n'
            ) for pool in self.pool_list]
        ) + '\n)'

    def add_lrna_pool(self,
                      risk_asset: str or Asset,
                      initial_quantity: float,
                      weight_cap: float = 1.0
                      ):
        asset = self.asset(risk_asset)

        new_pool = OmnipoolRiskAssetPool(
            positions=[
                amm.Position(asset, quantity=initial_quantity),
                amm.Position(self.lrna, quantity=initial_quantity * asset.price / self.lrna.price)
            ],
            weight_cap=weight_cap,
            unique_id=asset.name
        )
        self._asset_pools_dict[asset.name] = new_pool
        return self

    def pool(self, index: int or str or amm.Asset, *args) -> OmnipoolRiskAssetPool:
        """
        Given the name or index of an asset or a reference to that asset, returns the associated pool.
        Since an omnipool pool only has one risk asset, only the first function argument counts.
        """
        asset = self.asset(index)
        if isinstance(asset, amm.ShareToken):
            return self.pool(*asset.asset_names)
        pool_id = asset.name
        return self._asset_pools_dict[pool_id] if pool_id in self._asset_pools_dict else None

    @property
    def pool_list(self) -> list[OmnipoolRiskAssetPool]:
        """ Returns a list of all pools in the omnipool. """
        return [pool for pool in self._asset_pools_dict.values()]

    @property
    def pool_asset_list(self) -> list[Asset]:
        """ Returns a list of all assets for which there are pools """
        return [pool.asset for pool in self.pool_list]

    @property
    def lrna(self) -> amm.Asset:
        lrna = self.asset('LRNA')
        if not lrna:
            raise ValueError('LRNA not initialized in market!')
        return lrna

    def W(self, asset: str or Asset) -> float:
        """ the percentage of total LRNA contained in each asset pool """
        lrna_total = self.Q_total
        return self.pool(asset).lrnaQuantity / lrna_total

    def Q(self, asset: str or Asset) -> float:
        """ the absolute quantity of LRNA in each asset pool """
        return self.pool(asset).lrnaQuantity

    @property
    def Q_total(self) -> float:
        """ the total quantity of LRNA contained in all asset pools """
        return sum([pool.lrnaQuantity for pool in self.pool_list])

    def add_delta_Q(self, asset: str or Asset, value: float):
        self.pool(asset).lrnaQuantity += value

    def R(self, asset: str or Asset) -> float:
        """ quantity of risk asset in each asset pool """
        return self.pool(asset).assetQuantity

    def add_delta_R(self, asset: str or Asset, value: float):
        self.pool(asset).assetQuantity += value

    def B(self, asset: str or Asset) -> float:
        """ quantity of liquidity provider shares in each pool owned by the protocol """
        return self.pool(asset).sharesOwnedByProtocol

    def add_delta_B(self, asset: str or Asset, value: float):
        self.pool(asset).sharesOwnedByProtocol += value

    def S(self, asset: str or Asset):
        return self.pool(asset).shares

    def add_delta_S(self, asset: str or Asset, value: float):
        self.pool(asset).shares += value

    def T(self, asset: str or Asset) -> float:
        pool = self.pool(asset)
        if pool:
            return pool.totalValue
        else:
            return 0

    def add_delta_T(self, asset: str or Asset, value: float):
        self.pool(asset).totalValue += value

    @property
    def T_total(self):
        return sum([self.T(pool.name) for pool in self.pool_list])

    def P(self, asset: str or Asset) -> float:
        """ price of each asset denominated in LRNA """
        return self.pool(asset).ratio

    @property
    def L(self):
        return self.lrnaImbalance

    def add_delta_L(self, value):
        self.lrnaImbalance += value

    @property
    def C(self):
        """ soft cap on total value locked, denominated in preferred stablecoin """
        return self.tvlCapUSD

    # noinspection PyArgumentList
    def add_liquidity(self, agent: OmnipoolAgent, pool: OmnipoolRiskAssetPool, quantity: float):
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()
        U = self.preferred_stablecoin
        i = pool.assetName
        delta_r = quantity

        if agent.r(i) < delta_r:
            # print('Transaction rejected because agent has insufficient funds.')
            # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
            return self

        # math
        delta_q = Q(i) * delta_r / R(i)
        delta_s = S(i) * delta_r / R(i)
        delta_l = delta_r * Q(i) / R(i) * L / Q_total
        delta_t = (Q(i) + delta_q) * R(U) / Q(U) - T(i)

        if T_total + delta_t > self.tvlCapUSD:
            # print('Transaction rejected because it would exceed allowable market cap.')
            # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
            return self

        if (T(i) + delta_t) / (T_total + delta_t) > self.pool(i).weightCap:
            # print('Transaction rejected because it would exceed pool weight cap.')
            # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
            # print(repr(self))
            return self

        self.add_delta_Q(i, delta_q)
        self.add_delta_R(i, delta_r)
        self.add_delta_S(i, delta_s)
        self.add_delta_L(delta_l)

        # agent.add_delta_s(i, delta_s)
        agent.add_position(self.pool(i).shareToken.name, delta_s)
        agent.add_delta_r(i, -delta_r)
        agent.set_p(i, pool.ratio)

        # print('Liquidity provision succeeded.')
        # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
        return self

    # noinspection PyArgumentList
    def swap_assets(self, agent: OmnipoolAgent, sell_asset: str or Asset, buy_asset: str or Asset, delta_r):
        i = self.asset(sell_asset).name
        j = self.asset(buy_asset).name
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()
        # assert sell_quantity > 0, 'sell amount must be greater than zero'
        if delta_r < 0:
            # interpret this as specifying the transaction in terms of how much asset j to buy
            delta_rj = delta_r
            delta_qj = Q(j) * -delta_rj / (R(j) * (1 - Fa) + delta_rj)
            delta_qi = -delta_qj / (1 - Fp)
            delta_ri = R(i) * -delta_qi / (Q(i) + delta_qi)
        else:
            # specified in terms of how much asset i to sell
            delta_ri = delta_r
            delta_qi = Q(i) * -delta_ri / (R(i) + delta_ri)
            delta_qj = -delta_qi * (1 - Fp)
            delta_rj = R(j) * -delta_qj / (Q(j) + delta_qj) * (1 - Fa)

        delta_l = min(-delta_qi * Fp, -L)
        delta_qh = -delta_qi * Fp - delta_l

        if agent.r(i) + delta_ri / (1 - Fa) < 0:
            return self

        if Q(i) + delta_qi <= 0 or Q(j) + delta_qj <= 0:
            return self

        if R(i) / (1 - Fa) + delta_ri <= 0 or R(j) / (1 - Fa) + delta_rj <= 0:
            return self

        self.add_delta_Q(i, delta_qi)
        self.add_delta_Q(j, delta_qj)
        self.add_delta_R(i, delta_ri)
        self.add_delta_R(j, delta_rj)
        self.add_delta_Q('HDX', delta_qh)
        self.add_delta_L(delta_l)

        agent.add_delta_r(i, -delta_ri)
        agent.add_delta_r(j, -delta_rj)

        # print('Asset swap succeeded.')
        # print(f'({i} -> {j}, agent {agent.name}, quantity {sell_quantity})')
        return self

    # noinspection PyArgumentList
    def remove_liquidity(self, agent: OmnipoolAgent, pool: OmnipoolRiskAssetPool, quantity: float):
        i = pool.name
        u = self.preferred_stablecoin
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()

        if agent.s(i) < quantity or quantity < 0:
            return self

        delta_sa = -quantity
        delta_b = max((pool.ratio - agent.p(i)) / (pool.ratio + agent.p(i)) * delta_sa, 0)
        delta_si = delta_sa + delta_b
        delta_ri = R(i) / S(i) * delta_si
        delta_q = Q(i) * delta_ri / R(i)
        delta_l = delta_ri * Q(i) / R(i) * L / Q_total

        delta_ra = -delta_ri

        agent.add_delta_s(i, delta_sa)
        agent.add_delta_r(i, delta_ra)
        agent.add_delta_q(
            -pool.ratio * (2 * pool.ratio / (pool.ratio + agent.p(i)) * delta_sa / S(i) * R(i) + delta_ra)
        )

        self.add_delta_R(i, delta_ri)
        self.add_delta_Q(i, delta_q)
        self.add_delta_L(delta_l)
        self.add_delta_B(i, delta_b)
        self.add_delta_S(i, delta_si)

        delta_t = Q(i) * R(u) / Q(u) - T(i)
        self.add_delta_T(i, delta_t)

        return self

    # noinspection PyArgumentList
    def swap_lrna(self,
                  agent: OmnipoolAgent,
                  asset: str or Asset,
                  delta_r: float = 0,
                  delta_q: float = 0
                  ):
        delta_qa = -delta_q
        delta_ra = -delta_r
        delta_qi = delta_q
        delta_ri = delta_r
        i = self.asset(asset).name
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()

        if delta_ri + R(i) <= 0:
            # print("LRNA sell failed: there is not enough asset in the pool.")
            # print(f'(asset {i}, agent {agent.name}, asset bought {delta_ra}, LRNA sold {delta_qa})')
            return self
        if delta_qi + Q(i) <= 0:
            return self

        if delta_qa < 0 or delta_ra > 0:
            # agent is selling LRNA/buying asset
            if delta_qa < 0:
                # transaction specified in terms of LRNA sold
                delta_qi = -delta_qa
                delta_ri = R(i) * -delta_qi / (Q(i) * (1 - Fa) + delta_qi)
                delta_ra = -delta_ri
            elif delta_ra > 0:
                # transaction specified in terms of asset bought
                delta_ri = -delta_ra
                delta_qi = Q(i) * -delta_ri / (R(i) * (1 - Fa) + delta_ri)
                delta_qa = -delta_qi

            delta_l = 0
            delta_qh = 0

        elif delta_qa > 0 or delta_ra < 0:
            # agent is buying LRNA/selling asset
            if delta_qa > 0:
                # transaction specified in terms of LRNA bought
                delta_qi = -delta_qa
                delta_ri = R(i) * -delta_qi / (Q(i) + delta_qi)
                delta_ra = -delta_ri
            elif delta_ra < 0:
                # transaction specified in terms of asset sold
                delta_ri = -delta_ra
                delta_qi = Q(i) * -delta_ri / (R(i) + delta_ri)
                delta_qa = -delta_qi

            delta_l = min(-delta_qi * Fp, -L)
            delta_qh = -delta_qi * Fp - delta_l

        else:
            raise ValueError(
                f'Invalid transaction (asset {i}, agent {agent.name}, asset bought {delta_ra}, LRNA sold {delta_qa})'
            )

        # delta_l = delta_qi * (1 + (1 - Fa) * Q(i) / (Q(i) + delta_qi))

        if delta_ra / (1 - Fa) + agent.r(i) < 0 or delta_qa + agent.q < 0:
            # print("Agent has insufficient funds to complete transaction.")
            # print(f'(asset {i}, agent {agent.name}, delta_asset {delta_ra}, delta_LRNA {delta_qa})')
            return self
        elif delta_ri / (1 - Fa) + self.R(i) <= 0 or delta_qi + self.Q(i) <= 0:
            # print("Protocol has insufficient funds to complete transaction.")
            # print(f'(asset {i}, agent {agent.name}, delta_asset {delta_ra}, delta_LRNA {delta_qa})')
            return self

        self.add_delta_Q(i, delta_qi)
        self.add_delta_R(i, delta_ri)
        self.add_delta_Q('HDX', delta_qh)
        self.add_delta_L(delta_l)

        agent.add_delta_r(i, delta_ra)
        agent.add_delta_q(delta_qa)

        # print("Lrna sold for asset.")
        # print(f'(asset {i}, agent {agent.name}, asset bought {delta_ra}, LRNA sold {delta_qa})')

        return self


# noinspection PyArgumentList
def swap_assets(
        market_state: Omnipool,
        agents_list: list[OmnipoolAgent],
        sell_asset: int or str or amm.Asset,
        buy_asset: int or str or amm.Asset,
        trader_id: int,
        delta_r: float
        ) -> tuple[Omnipool, list[OmnipoolAgent]]:

    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).swap_assets(
        agent=agents_list[trader_id],
        sell_asset=market_state.asset(sell_asset).name,
        buy_asset=market_state.asset(buy_asset).name,
        delta_r=delta_r
    )

    return new_state, new_agents


# noinspection PyArgumentList
def add_liquidity(market_state: Omnipool,
                  agents_list: list[OmnipoolAgent],
                  agent_index: int,
                  asset: str or Asset,
                  delta_r: float
                  ) -> tuple[Omnipool, list[OmnipoolAgent]]:
    """ compute new state after agent[agent_index] adds liquidity to asset[asset_index] pool in quantity delta_r """
    if delta_r < 0:
        raise ValueError("Cannot provision negative liquidity ^_-")

    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).add_liquidity(
        agent=new_agents[agent_index],
        pool=market_state.pool(asset),
        quantity=delta_r
    )

    return new_state, new_agents


def remove_liquidity(market_state: Omnipool,
                     agents_list: list[OmnipoolAgent],
                     agent_index: int,
                     asset: str or Asset,
                     delta_r: float
                     ) -> tuple[Omnipool, list[OmnipoolAgent]]:
    """ Compute new state after agent[agent_index] removes liquidity from pool[asset_index] in quantity delta_r. """

    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).remove_liquidity(
        agent=new_agents[agent_index],
        pool=market_state.pool(asset),
        quantity=delta_r
    )

    return new_state, new_agents


def swap_lrna(market_state: Omnipool,
              agents_list: list[OmnipoolAgent],
              agent_index: int,
              asset: str or Asset,
              delta_r: float = 0,
              delta_q: float = 0
              ) -> tuple[Omnipool, list[OmnipoolAgent]]:
    """
    Compute new state after swapping LRNA for an asset. Delta_q and delta_r are from the perspective of the market.
    """
    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).swap_lrna(
        agent=new_agents[agent_index],
        asset=asset,
        delta_r=delta_r,
        delta_q=delta_q
    )
    return new_state, new_agents


class OmnipoolTradeStrategies:
    @staticmethod
    def random_swaps(amount: float = 0, percent: float = 0, randomize_amount: bool = False):
        percent = min(percent, 100)
        if percent == 100:
            randomize_amount = True

        @TradeStrategy
        def strategy(agent: OmnipoolAgent, market: Omnipool):
            buy_asset = random.choice([position.asset for position in agent.positions.values()])
            sell_asset = random.choice([position.asset for position in agent.positions.values()])
            sell_quantity = max(
                             (amount / sell_asset.price
                              or agent.holdings(sell_asset) * percent / 100
                              )
                             * (random.random() if randomize_amount else 1)
                             , agent.holdings(sell_asset)) or 1
            if buy_asset == sell_asset:
                return market, agent
            elif buy_asset == market.lrna:
                return market.swap_lrna(agent=agent, asset=sell_asset, delta_r=sell_quantity)
            elif sell_asset == market.lrna:
                return market.swap_lrna(agent=agent, asset=buy_asset, delta_q=sell_quantity)
            else:
                return market.swap_assets(
                    agent=agent,
                    sell_asset=sell_asset,
                    buy_asset=buy_asset,
                    delta_r=sell_quantity
                ), agent
        return strategy

    @staticmethod
    def arbitrage(sell_fee: float = 0):

        # noinspection PyArgumentList
        @TradeStrategy
        def strategy(agent: OmnipoolAgent, market: Omnipool):
            stablecoin = market.preferred_stablecoin
            for riskAsset in agent.asset_list:
                if not agent.holdings(riskAsset) or riskAsset == stablecoin:
                    continue
                P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = market.algebraic_symbols()
                P = (lambda x: market.pool(x).ratio * market.lrna.price)  # get exchange price in USD not LRNA
                Pe = (lambda x: market.asset(x).price * (1 - sell_fee))  # external market price, also in USD
                i = riskAsset.name
                o = stablecoin

                delta_ro = (
                            R(i) * Q(o)
                            * (-P(i) + math.sqrt(P(i) * Pe(i) * (1 - Fa) * (1 - Fp)))
                            / (Q(i) + Q(o) * (1 - Fp))
                            )

                if delta_ro > 0:
                    market.swap_assets(agent, sell_asset=o, buy_asset=i, delta_r=delta_ro)
                elif delta_ro < 0:
                    market.swap_assets(agent, sell_asset=i, buy_asset=o, delta_r=delta_ro)

        return strategy
