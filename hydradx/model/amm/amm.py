import copy
import typing


class Asset:
    _ids = 0
    dict = {}
    list = []

    def __init__(self, name: str, price: float):
        """ price is denominated in USD, probably ¯\_(ツ)_/¯ """
        self.name = name
        self.price = price
        self.index = Asset._ids
        Asset._ids += 1
        Asset.dict[name] = self
        Asset.list.append(self)
        
    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.name == other.name
        else:
            return self.name == other

    @staticmethod
    def clear():
        Asset._ids = 0
        Asset.dict = {}
        Asset.list = []


class AssetHolder:
    def __init__(self):
        self._asset_dict = Asset.dict

    @property
    def asset_list(self) -> list[Asset]:
        return [self.asset(asset) for asset in self._asset_dict.values()]

    @asset_list.setter
    def asset_list(self, value: list[Asset]):
        self._asset_dict = {asset.name: asset for asset in sorted(value, key=lambda x: x.index)}

    def asset(self, asset: str or int or Asset) -> Asset:
        if isinstance(asset, Asset):
            return self._asset_dict[asset.name] if asset.name in self._asset_dict else None
        elif type(asset) == int:
            return self.asset_list[asset]
        return self._asset_dict[asset] if asset in self._asset_dict else None


class Position:
    def __init__(self,
                 asset: Asset,
                 quantity: float = 0,
                 buy_in_price: float = 0):

        self.quantity = quantity
        self.buy_in_price = buy_in_price or asset.price
        self.asset = asset

    @property
    def assetName(self):
        """ The name of the asset. """
        return self.asset.name


class ShareToken(Asset):
    def __init__(self, name: str, price: float, asset_names: list):
        super().__init__(name, price)
        self.asset_names = asset_names

    @staticmethod
    def token_name(asset_name: str):
        return f'pool shares ({asset_name})'


class RiskAssetPool:
    index: int

    def __init__(self,
                 positions: list[Position],  # of positions
                 weight_cap: float = 1,
                 unique_id: str = ""
                 ):
        """
        The state of one asset pool.
        """
        self.positions = positions
        self.weightCap = weight_cap
        self.shares = self.positions[0].quantity
        self.sharesOwnedByProtocol = self.shares
        self.totalValue = sum([position.quantity * position.buy_in_price for position in positions])
        self.unique_id = unique_id

        self.shareToken = ShareToken(
            name=ShareToken.token_name(self.unique_id),
            price=self.ratio,
            asset_names=[position.assetName for position in self.positions]
        )
        # Market.add_asset(self.shareToken)

    def position(self, asset_name: str) -> Position:
        for position in self.positions:
            if position.assetName == asset_name:
                return position

    @property
    def ratio(self) -> float:
        return self.positions[1].quantity / self.positions[0].quantity


class TradeStrategy:
    def __init__(self, strategy_function: typing.Callable):
        self.function = strategy_function

    def execute(self, agent, market) -> tuple:
        return self.function(agent, market)


class Agent(AssetHolder):
    def __init__(self, name: str, trade_strategy: TradeStrategy = None):
        super().__init__()
        self.name = name
        self.tradeStrategy = trade_strategy
        self.positions = {}

    def erase_external_holdings(self):
        """ agent loses all their money that is not in an exchange """
        for asset in self.asset_list:
            if type(asset) == Asset and self.position(asset):
                self.position(asset.name).quantity = 0
        return self

    def position(self, asset: str or Asset) -> Position:
        """ Given the name of (or a reference to) an asset, returns the agent's position as regards that asset. """
        asset_name = self.asset(asset).name
        return self.positions[asset_name] if asset_name in self.positions else None

    def holdings(self, asset: str or Asset) -> float:
        """ get this agent's holdings in the specified asset """
        return self.position(asset).quantity if self.position(asset) else 0

    def price(self, asset: str or Asset) -> float:
        """ Returns this agent's buy-in price for the specified asset. """
        return self.position(asset).buy_in_price if self.position(asset) else None

    def add_position(self, asset: str or Asset, quantity: float, price: float = 0):
        """ add specified quantity to the agent's asset holdings in market """
        asset = self.asset(asset)
        if self.position(asset):
            self.position(asset).quantity += quantity
            return self

        # if (market, asset) position was not found, create one
        newPosition = Position(
            asset=asset,
            quantity=quantity,
            buy_in_price=price
        )
        self.positions[asset.name] = newPosition
        return self


class Exchange(AssetHolder):
    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float,
                 ):
        super().__init__()
        self._asset_pools_dict = dict()

        self.tvlCapUSD = tvl_cap_usd
        self.assetFee = asset_fee

    def add_pool(self, positions: list, weight_cap: float = 0):
        """ add an asset pool to the exchange """
        if not self.pool(positions):
            new_pool = RiskAssetPool(
                positions=positions,
                weight_cap=weight_cap,
                unique_id=self.poolName([position.assetName for position in positions])
            )
            # maintain a reference to both name and index in _asset_pools_dict, for convenience
            self._asset_pools_dict[new_pool.unique_id] = new_pool
        return self

    def pool(self, *asset_names) -> RiskAssetPool:
        """ get the pool containing the specified assets """
        index = self.poolName(list(asset_names))
        return self._asset_pools_dict[index] if index in self._asset_pools_dict else None

    @property
    def pool_list(self) -> list[RiskAssetPool]:
        return [pool for pool in self._asset_pools_dict.values()]

    @staticmethod
    def poolName(asset_names: list) -> str:
        return ", ".join(sorted(asset_names))

    def add_liquidity(self, agent: Agent, pool: RiskAssetPool, quantity: float):
        """ very naive default implementation of add_liquidity. override. """
        for position in pool.positions:
            position.quantity += quantity

    def remove_liquidity(self, agent: Agent, pool: RiskAssetPool, quantity: float):
        """ very naive default implementation of add_liquidity. override. """
        for position in pool.positions:
            if position.quantity < quantity:
                return

        for position in pool.positions:
            position.quantity -= quantity

    def swap_assets(self, agent: Agent, sell_asset: str or Asset, buy_asset: str or Asset, sell_quantity):
        """ very naive default implementation of swap_assets. override. """
        pool = self.pool([buy_asset, sell_asset])
        if pool.position(sell_asset).quantity < sell_quantity:
            return
        pool.position(sell_asset).quantity -= sell_quantity
        pool.position(sell_asset).quantity += sell_quantity


class WorldState:
    def __init__(self, exchange: Exchange, agents: dict[str: Agent]):
        self.asset_list = exchange.asset_list
        self.exchange = exchange
        self.agents = agents
        self.syncAssets()

    def copy(self):
        """ Returns a copy of the world state with assets synced to the common list. Use instead of copy.deepcopy. """
        returnCopy = copy.deepcopy(self)
        returnCopy.syncAssets()
        return returnCopy

    def syncAssets(self):
        # make sure everyone is working from the same list of assets in an object-identity sense
        for holder in [self.exchange] + list(self.agents.values()):
            holder.asset_list = self.asset_list
        for pool in self.exchange.pool_list:
            for position in pool.positions:
                position.asset = self.exchange.asset(position.assetName)
        for agent in self.agents.values():
            for position in agent.positions.values():
                position.asset = agent.asset(position.assetName)

