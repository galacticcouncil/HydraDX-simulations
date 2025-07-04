import copy, math, numpy as np

from hydradx.model.amm.exchange import Exchange
from hydradx.model.amm.xyk_amm import ConstantProductPoolState


class AmmConstraints:
    def __init__(self, amm: Exchange):
        self.asset_list = amm.asset_list
        self.tkn_shares = amm.unique_id

    def get_indicator_matrices(self, global_asset_list: list[str]):
        asset_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        share_mat = np.zeros((len(global_asset_list), len(self.asset_list) + 1))
        for i, asset in enumerate(global_asset_list):
            if asset == self.tkn_shares:
                share_mat[i, 0] = 1
            if asset in self.asset_list:
                asset_mat[i, self.asset_list.index(asset) + 1] = 1
        return share_mat, asset_mat

class XykConstraints(AmmConstraints):
    def __init__(self, amm: ConstantProductPoolState):
        super().__init__(amm)

class StableswapConstraints(AmmConstraints):
    def __init__(self, amm: Exchange):
        super().__init__(amm)
