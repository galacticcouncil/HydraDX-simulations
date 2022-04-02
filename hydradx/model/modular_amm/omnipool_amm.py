import copy
# import pytest
from ..modular_amm import amm
from ..modular_amm.amm import Market


class OmniPool(amm.Market):

    @property
    def totalQ(self) -> float:
        """ the total quantity of LRNA contained in all asset pools """
        return sum([asset.lrnaQuantity for asset in self._asset_list])

    def W(self, index: int or str) -> float:
        """ the percentage of total LRNA contained in each asset pool """
        lrna_total = self.totalQ
        return self._asset_dict[index].lrnaQuantity / lrna_total

    def Q(self, index: int or str) -> float:
        """ the absolute quantity of LRNA in each asset pool """
        return self._asset_dict[index].lrnaQuantity

    def add_delta_Q(self, index: int or str, value):
        self._asset_dict[index].lrnaQuantity += value

    def R(self, index: int or str) -> float:
        """ quantity of risk asset in each asset pool """
        return self._asset_dict[index].assetQuantity

    def add_delta_R(self, index: int or str, value):
        self._asset_dict[index].assetQuantity += value

    def B(self, index: int or str) -> float:
        """ quantity of liquidity provider shares in each pool owned by the protocol """
        return self._asset_dict[index].sharesOwnedByProtocol

    def P(self, index: int or str) -> float:
        """ price of each asset denominated in LRNA """
        return self._asset_dict[index].price

    @property
    def L(self):
        return self.lrnaImbalance

    @L.setter
    def L(self, value):
        self.lrnaImbalance = value

    @property
    def C(self):
        """ soft cap on total value locked, denominated in preferred stablecoin """
        return self.tvlCapUSD


class Agent(amm.Agent):
    omnipool: amm.Exchange
    external_market: amm.Market

    def __init__(self,
                 name: str,
                 positions: dict[Market: dict[str: float]],
                 ):
        super().__init__(name=name, positions=positions)
        for market in positions:
            if type(market) == OmniPool:
                self.omnipool = market
            else:
                self.external_market = market

    def s(self, index: int or str) -> float:
        """ quantity of shares in pool[index] owned by the agent """
        return self.holdings(self.omnipool, index)

    def r(self, index: int or str) -> float:
        """ quantity of asset[index] owned by the agent external to the omnipool """
        return self.holdings(self.external_market, index)

    def add_delta_r(self, index: int or str, value):
        """ add value to agent's asset holding in pool[index] """
        self.add_asset(self.omnipool, index, value)

    def p(self, index: int or str) -> float:
        """ price at which the agent's holdings in pool[index] were acquired """
        return self.asset[index].price if index in self.poolAssets else 0

    @property
    def q(self):
        """ quantity of LRNA held by the agent """
        return self.outsideAssets['LRNA'] if 'LRNA' in self.outsideAssets else 0


def swap_assets(
        state: (amm.Exchange, list[Agent]),
        sell_index: int,
        buy_index: int,
        trader_id: str or int,
        sell_quantity: float
) -> tuple[amm.Exchange, list[Agent]]:

    i = sell_index
    j = buy_index
    omni_pool = state.exchange
    assert sell_quantity > 0, 'sell amount must be greater than zero'

    delta_Qi = omni_pool.Q(i) * -sell_quantity / (omni_pool.R(i) + sell_quantity)
    delta_Qj = -delta_Qi * (1 - omni_pool.lrnaFee)
    delta_Rj = omni_pool.R(j) * -delta_Qj / (omni_pool.Q(j) + delta_Qj) * (1 - omni_pool.assetFee)
    delta_L = min(-delta_Qi * omni_pool.lrnaFee, -omni_pool.L)
    delta_QH = -omni_pool.lrnaFee * delta_Qi - delta_L
    delta_Ri = sell_quantity

    new_state = copy.deepcopy(state)
    omni_pool = new_state.exchange
    omni_pool.add_delta_Q(i, delta_Qi)
    omni_pool.add_delta_Q(j, delta_Qj)
    omni_pool.add_delta_R(i, delta_Ri)
    omni_pool.add_delta_R(j, delta_Rj)
    omni_pool.add_delta_Q('HDX', delta_QH)
    omni_pool.L += delta_L

    # do some algebraic checks
    # if omni_pool.Q(i) * omni_pool.R(i) != pytest.approx(omni_pool.Q(i) * omni_pool.R(i)):
    #     raise f'price change in asset {i}'

    trader: Agent = new_state.agents[trader_id]

    trader.add_delta_r(i, -delta_Ri)
    trader.add_delta_r(j, -delta_Rj)

    return new_state

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
