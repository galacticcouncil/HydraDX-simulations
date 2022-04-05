import copy
import pytest
from ..modular_amm import amm
from ..modular_amm.amm import Market


class OmniPool(amm.Exchange):

    def set_initial_liquidity(self, asset: amm.Asset, agent: amm.Agent, quantity: float):
        super().set_initial_liquidity(asset, agent, quantity)

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

    def T(self, index: int or str):
        return self.pool(index).assetQuantity * self.pool(index).asset.price

    def T_total(self):
        return sum([self.T(index) for index in self.asset_list])

    def P(self, index: int or str) -> float:
        """ price of each asset denominated in LRNA """
        return self._asset_dict[index].lrna_price

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
        return self.P, self.Q, self.R, self.S, self.T, self.L, self.lrnaFee, self.assetFee, self.Q_total, self.T_total()


class OmnipoolAgent(amm.Agent):
    omnipool: amm.Exchange
    external_market: amm.Market

    def __init__(self,
                 name: str,
                 ):
        super().__init__(name=name)

    # noinspection PyTypeChecker
    def add_asset(self, market: Market, asset: amm.Asset, quantity: float):
        # auto-detect these parameters
        if type(market) == OmniPool:
            self.omnipool = market
            self.add_liquidity(market, asset, quantity)
        else:
            self.external_market = market
        super().add_asset(market=market, asset=asset, quantity=quantity)
        return self

    def add_liquidity(self, market: amm.Exchange, asset: amm.Asset, quantity: float):
        market.set_initial_liquidity(asset=asset, agent=self, quantity=quantity)

    def s(self, index: int or str) -> float:
        """ quantity of shares in pool[index] owned by the agent """
        return self.holdings(self.omnipool, index)

    def r(self, index: int or str) -> float:
        """ quantity of asset[index] owned by the agent external to the omnipool """
        return self.holdings(self.external_market, index)

    def add_delta_r(self, index: int or str, value):
        """ add value to agent's asset holding in external market """
        self.add_asset(self.external_market, index, value)

    def add_delta_s(self, index: int or str, value):
        self.add_asset(self.omnipool, index, value)

    def p(self, index: int or str) -> float:
        """ price at which the agent's holdings in pool[index] were acquired """
        return self.price(self.omnipool, index)

    @property
    def q(self):
        """ quantity of LRNA held by the agent """
        return self.holdings(self.omnipool, 'LRNA')


# noinspection PyArgumentList
def swap_assets(
        market_state: OmniPool,
        agents_list: list[OmnipoolAgent],
        sell_index: int or str,
        buy_index: int or str,
        trader_id: int,
        sell_quantity: float
        ) -> tuple[amm.Exchange, list[OmnipoolAgent]]:

    i = sell_index
    j = buy_index
    P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = market_state.algebraic_symbols()
    for n in [i, j]:
        if not market_state.pool(n):
            raise IndexError('asset {n} not found')
    old_state = market_state
    assert sell_quantity > 0, 'sell amount must be greater than zero'

    delta_Ri = sell_quantity
    delta_Qi = Q(i) * -delta_Ri / (R(i) + delta_Ri)
    delta_Qj = -delta_Qi * (1 - Fp)
    delta_Rj = R(j) * -delta_Qj / (Q(j) + delta_Qj) * (1 - Fa)
    delta_L = min(-delta_Qi * Fp, -L)
    delta_QH = -delta_Qi * Fp - delta_L

    new_state = copy.deepcopy(market_state)
    new_state.add_delta_Q(i, delta_Qi)
    new_state.add_delta_Q(j, delta_Qj)
    new_state.add_delta_R(i, delta_Ri)
    new_state.add_delta_R(j, delta_Rj)
    new_state.add_delta_Q('HDX', delta_QH)
    new_state.add_delta_L(delta_L)

    # do some algebraic checks
    if new_state.Q(i) * new_state.R(i) != pytest.approx(new_state.Q(i) * new_state.R(i)):
        raise ValueError('price change in asset {i}')
    if i != 0 and j != 0:
        if delta_L + delta_Qj + delta_Qi + delta_QH != pytest.approx(0, abs=1e10):
            raise ValueError('Some LRNA was lost along the way.')

    new_agents = copy.deepcopy(agents_list)
    trader = new_agents[trader_id]

    trader.add_delta_r(i, -delta_Ri)
    trader.add_delta_r(j, -delta_Rj)

    return new_state, new_agents


# noinspection PyArgumentList
def add_liquidity(market_state: OmniPool,
                  agents_list: list[OmnipoolAgent],
                  agent_index: int,
                  asset_index: int or str,
                  delta_r: float
                  ) -> tuple[OmniPool, list[OmnipoolAgent]]:
    """ compute new state after agent[agent_index] adds liquidity to asset[asset_index] in quantity delta_r """
    if delta_r < 0:
        raise ValueError("Cannot provision negative liquidity ^_-")
    if agents_list[agent_index].r(asset_index) < delta_r:
        print('Transaction rejected because agent has insufficient funds.')
        print(f'(asset {asset_index}, agent {agent_index}, quantity {delta_r})')
        return market_state, agents_list

    P, Q, R, S, T, L, Fp, Fa, Q_total, T_total = market_state.algebraic_symbols()
    U = market_state.stableCoin
    i = asset_index
    delta_q = Q(i) * delta_r / R(i)
    delta_s = S(i) * delta_r / R(i)
    delta_l = delta_r * Q(i) / R(i) * L / Q_total
    delta_t = (Q(i) + delta_q) * R(U) / Q(U) + T(i)

    if T_total + delta_t > market_state.tvlCapUSD:
        print('Transaction rejected because it would exceed allowable market cap.')
        print(f'(asset {asset_index}, agent {agent_index}), quantity {delta_r}')
        return market_state, agents_list

    elif pytest.approx(R(i) / S(i)) != (R(i) + delta_r) / (S(i) + delta_s):
        raise ValueError("Incorrect ratio of assets to shares.")

    elif pytest.approx(Q(i) / R(i)) != (Q(i) + delta_q) / (R(i) + delta_r):
        raise ValueError("Asset price should not change when liquidity is added.")

    elif pytest.approx(Q(i) / R(i) * (Q_total + L) / Q_total) != \
            (Q(i) + delta_q) / (R(i) + delta_r) * (Q_total + delta_q + L + delta_l) / (Q_total + delta_q):
        # TODO: understand better what this means.
        raise ValueError("Target price has changed.")

    new_state = copy.deepcopy(market_state)
    new_state.add_delta_Q(i, delta_q)
    new_state.add_delta_R(i, delta_r)
    new_state.add_delta_S(i, delta_s)
    new_state.add_delta_L(delta_l)

    new_agents = copy.deepcopy(agents_list)
    lp = new_agents[agent_index]
    lp.add_delta_s(delta_s)
    lp.add_delta_r(-delta_r)
    assert lp.p(i) == pytest.approx(new_state.price(i))


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
