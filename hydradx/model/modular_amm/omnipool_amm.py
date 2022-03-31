import copy
import pytest
import amm
from amm import State


class OmniPool:
    @staticmethod
    def swap_assets(
            state: State,
            sell_index: int,
            buy_index: int,
            trader_id: str or int,
            sell_quantity: float
    ) -> State:

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
        new_state.Q(i) += delta_Qi
        new_state['Q'][j] += delta_Qj
        new_state['R'][i] += delta_Ri
        new_state['R'][j] += delta_Rj
        new_state['Q'][new_state['token_list'].index('HDX')] += delta_QH
        new_state['L'] += delta_L

        # do some algebraic checks
        if omni_pool.Q(i) * omni_pool.R[i] != pytest.approx(new_state['Q'][i] * new_state['R'][i]):
            raise f'price change in asset {i}'

        trader: amm.Agent
        if type(trader_id) == int:
            trader = new_state.agents[trader_id]
        elif type(trader_id) == str:
            trader = new_state.agents[[agent.name for agent in new_state.agents].index(trader_id)]

        trader.r[i] -= delta_Ri
        trader.r[j] -= delta_Rj

        return new_state

    @staticmethod
    def add_asset(
        state: State,
        asset_name,
        asset_quantity,
        asset_price
    ) -> State:

        new_state = copy.deepcopy(state)
        return new_state

    @staticmethod
    def buy_lrna(
         state: State,
         sell_asset: int or str or amm.RiskAssetPool,
         quantity: float
    ) -> State:

        new_state = copy.deepcopy(state)
        return new_state

    @staticmethod
    def sell_lrna(
        state: State,
        buy_asset: int or str or amm.RiskAssetPool,
        sell_asset
    ) -> State:

        new_state = copy.deepcopy(state)
        return new_state
