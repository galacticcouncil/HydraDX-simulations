import copy
from ..modular_amm import amm


class OmnipoolRiskAssetPool(amm.RiskAssetPool):
    def __init__(self, positions: list, weight_cap: float = 1):
        """
        The state of one asset pool.
        """
        super().__init__(positions, weight_cap, unique_id=positions[0].name)

    @property
    def name(self):
        return self.positions[0].name

    @property
    def lrnaQuantity(self):
        return self.positions[1].quantity

    @lrnaQuantity.setter
    def lrnaQuantity(self, value):
        self.positions[1].quantity = value

    @property
    def assetQuantity(self):
        return self.positions[0].quantity

    @assetQuantity.setter
    def assetQuantity(self, value):
        self.positions[0].quantity = value

    @property
    def assetIndex(self):
        return self.positions[0].asset.name


class OmnipoolAgent(amm.Agent):

    # noinspection PyTypeChecker
    def add_liquidity(self, market: amm.Exchange, asset: amm.Asset, quantity: float):
        if not self.position(asset):
            self.add_position(asset, quantity)
            self.add_position(market.pool(asset).shareToken, 0)
        market.add_liquidity(agent=self, pool=market.pool(asset), quantity=quantity)
        return self

    def pool_asset(self, index: int or str or amm.Asset) -> amm.ShareToken:
        for asset in filter(lambda a: isinstance(a, amm.ShareToken), self.asset_list):
            if asset.name == amm.ShareToken.token_name(index) or asset.index == index or index in asset.assets:
                return asset

    def s(self, index: int or str or amm.Asset) -> float:
        """ quantity of shares in pool[index] owned by the agent """
        return self.holdings(self.pool_asset(index))

    def r(self, index: int or str or amm.Asset) -> float:
        """ quantity of asset[index] owned by the agent external to the omnipool """
        return self.holdings(index)

    def add_delta_r(self, index: int or str or amm.Asset, value):
        """ add value to agent's holding in asset """
        self.position(index).quantity += value

    def add_delta_s(self, index: int or str or amm.Asset, value):
        """ change quantity of shares in pool[index] owned by the agent """
        self.position(self.pool_asset(index)).quantity += value

    def p(self, index: int or str or amm.Asset) -> float:
        """ price at which the agent's holdings in pool[index] were acquired """
        return self.price(self.pool_asset(index))

    def set_p(self, index: int or str or amm.Asset, value):
        self.position(self.pool_asset(index)).price = value

    @property
    def q(self):
        """ quantity of LRNA held by the agent """
        return self.holdings('LRNA')


class OmniPool(amm.Exchange):
    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float,
                 lrna_fee: float,
                 asset_list: list,
                 preferred_stablecoin: amm.Asset
                 ):
        super().__init__(tvl_cap_usd, asset_fee, asset_list, price_denominator=preferred_stablecoin)
        self.lrnaFee = lrna_fee
        self.lrna = amm.Asset('LRNA', 1)
        self.lrnaImbalance = 0
        self.stableCoin = preferred_stablecoin

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
                f'    price: {pool.price}\n'
                f'    weight: {pool.totalValue}/{self.T_total} ({pool.totalValue / self.T_total})\n'
                f'    weight cap: {pool.weightCap}\n'
                f'    total shares: {pool.shares}\n'
                f'    protocol shares: {pool.sharesOwnedByProtocol}\n'
            ) for pool in self.pool_list]
        ) + '\n)'

    def add_lrna_pool(self,
                      risk_asset: amm.Asset,
                      initial_quantity: float,
                      weight_cap: float = 1.0
                      ) -> OmnipoolRiskAssetPool:

        new_pool = OmnipoolRiskAssetPool(
            positions=[
                amm.Position(risk_asset, quantity=initial_quantity),
                amm.Position(self.lrna, quantity=initial_quantity * risk_asset.price / self.lrna.price)
            ],
            weight_cap=weight_cap
        )
        self._asset_pools_list.append(new_pool)
        self._asset_pools_dict[risk_asset] = new_pool
        return new_pool

    def pool(self, index: int or str or amm.Asset) -> OmnipoolRiskAssetPool:
        """
        given the name or index of an asset, a reference to that asset,
        or an (asset, lrna) set, returns the associated pool
        """
        asset = self.asset(index)
        return self._asset_pools_dict[asset] if asset else None

    def W(self, index: int or str) -> float:
        """ the percentage of total LRNA contained in each asset pool """
        lrna_total = self.Q_total
        return self.pool(index).lrnaQuantity / lrna_total

    def Q(self, index: int or str) -> float:
        """ the absolute quantity of LRNA in each asset pool """
        return self.pool(index).lrnaQuantity

    @property
    def Q_total(self) -> float:
        """ the total quantity of LRNA contained in all asset pools """
        return sum([pool.lrnaQuantity for pool in self.pool_list])

    def add_delta_Q(self, index: int or str, value):
        self.pool(index).lrnaQuantity += value

    def R(self, index: int or str) -> float:
        """ quantity of risk asset in each asset pool """
        return self.pool(index).assetQuantity

    def add_delta_R(self, index: int or str, value):
        self.pool(index).assetQuantity += value

    def B(self, index: int or str) -> float:
        """ quantity of liquidity provider shares in each pool owned by the protocol """
        return self.pool(index).sharesOwnedByProtocol

    def S(self, index: int or str):
        return self.pool(index).shares

    def add_delta_S(self, index: int or str, value):
        self.pool(index).shares += value

    def T(self, index: int or str) -> float:
        pool = self.pool(index)
        if pool:
            return pool.totalValue
        else:
            return 0

    def add_delta_T(self, index: int or str, value):
        self.pool(index).totalValue += value

    @property
    def T_total(self):
        return sum([self.T(index) for index in self.asset_list])

    def P(self, index: int or str) -> float:
        """ price of each asset denominated in LRNA """
        return self.pool(index).price

    @property
    def L(self):
        return self.lrnaImbalance

    def add_delta_L(self, value):
        self.lrnaImbalance += value

    @property
    def C(self):
        """ soft cap on total value locked, denominated in preferred stablecoin """
        return self.tvlCapUSD

    def algebraic_symbols(self):
        """ usage: P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = Omnipool.algebraic_symbols() """
        return self.P, self.Q, self.R, self.S, self.T, self.L, self.lrnaFee, self.assetFee, self.Q_total, self.T_total

    # noinspection PyArgumentList
    def add_liquidity(self, agent: OmnipoolAgent, pool: OmnipoolRiskAssetPool, quantity: float):
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()
        U = self.stableCoin
        i = pool.assetIndex
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

        if (T(i) + delta_t) / T_total > self.pool(i).weightCap:
            # print('Transaction rejected because it would exceed pool weight cap.')
            # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
            # print(repr(self))
            return self

        self.add_delta_Q(i, delta_q)
        self.add_delta_R(i, delta_r)
        self.add_delta_S(i, delta_s)
        self.add_delta_L(delta_l)

        # agent.add_delta_s(i, delta_s)
        agent.add_position(self.pool(i).shareToken, delta_s)
        agent.add_delta_r(i, -delta_r)
        agent.set_p(i, self.price(i))

        # print('Liquidity provision succeeded.')
        # print(f'(asset {i}, agent {agent.name}, quantity {delta_r})')
        return self

    # noinspection PyArgumentList
    def swap_assets(self, agent: OmnipoolAgent, sell_asset: amm.Asset, buy_asset: amm.Asset, sell_quantity):
        i = self.pool(sell_asset).assetIndex
        j = self.pool(buy_asset).assetIndex
        P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = self.algebraic_symbols()
        assert sell_quantity > 0, 'sell amount must be greater than zero'

        delta_Ri = sell_quantity
        delta_Qi = Q(i) * -delta_Ri / (R(i) + delta_Ri)
        delta_Qj = -delta_Qi * (1 - Fp)
        delta_Rj = R(j) * -delta_Qj / (Q(j) + delta_Qj) * (1 - Fa)
        delta_L = min(-delta_Qi * Fp, -L)
        delta_QH = -delta_Qi * Fp - delta_L

        self.add_delta_Q(i, delta_Qi)
        self.add_delta_Q(j, delta_Qj)
        self.add_delta_R(i, delta_Ri)
        self.add_delta_R(j, delta_Rj)
        self.add_delta_Q('HDX', delta_QH)
        self.add_delta_L(delta_L)

        agent.add_delta_r(i, -delta_Ri)
        agent.add_delta_r(j, -delta_Rj)

        print('Asset swap succeeded.')
        print(f'({i} -> {j}, agent {agent.name}, quantity {sell_quantity})')
        return self


# noinspection PyArgumentList
def swap_assets(
        market_state: OmniPool,
        agents_list: list[OmnipoolAgent],
        sell_index: int or str,
        buy_index: int or str,
        trader_id: int,
        sell_quantity: float
        ) -> tuple[OmniPool, list[OmnipoolAgent]]:

    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).swap_assets(
        agent=agents_list[trader_id],
        sell_asset=market_state.asset(sell_index),
        buy_asset=market_state.asset(buy_index),
        sell_quantity=sell_quantity
    )

    return new_state, new_agents


# noinspection PyArgumentList
def add_liquidity(market_state: OmniPool,
                  agents_list: list[OmnipoolAgent],
                  agent_index: int,
                  asset_index: int or str,
                  delta_r: float
                  ) -> tuple[OmniPool, list[OmnipoolAgent]]:
    """ compute new state after agent[agent_index] adds liquidity to asset[asset_index] pool in quantity delta_r """
    if delta_r < 0:
        raise ValueError("Cannot provision negative liquidity ^_-")

    new_agents = copy.deepcopy(agents_list)
    new_state = copy.deepcopy(market_state).add_liquidity(
        agent=new_agents[agent_index],
        pool=market_state.pool(asset_index),
        quantity=delta_r
    )

    return new_state, new_agents


# def add_asset(
#     state: State,
#     asset_name,
#     asset_quantity,
#     asset_price
# ) -> State:
#
#     new_state = copy.deepcopy(state)
#     return new_state
#
# def buy_lrna(
#      state: State,
#      sell_asset: int or str or amm.RiskAssetPool,
#      quantity: float
# ) -> State:
#
#     new_state = copy.deepcopy(state)
#     return new_state
#
# def sell_lrna(
#     state: State,
#     buy_asset: int or str or amm.RiskAssetPool,
#     sell_asset
# ) -> State:
#
#     new_state = copy.deepcopy(state)
#     return new_state
