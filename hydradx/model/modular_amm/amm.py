class Asset:
    _ids = 0

    def __init__(self, name: str, price: float):
        """ price is denominated in USD, probably ¯\_(ツ)_/¯ """
        self.name = name
        self.price = price
        self.index = Asset._ids
        Asset._ids += 1

    @classmethod
    def clear(cls):
        cls._ids = 0


class Market:
    _asset_dict = dict()

    def initializeAssetList(self, assets: list):
        """
        Reset and initialize a list of assets. The list is initialized on a Market class itself,
        so it is universal among this and all other instances of Market or Agent.
        """
        Market.reset()
        for asset in assets:
            Market.add_asset(asset)
        return self

    @staticmethod
    def reset():
        Market._asset_dict = dict()
        Asset.clear()

    @staticmethod
    def add_asset(new_asset: Asset):
        """ Add an asset to the Market class. It will be visible to all instances of Market or Agent. """
        # no two assets should have the same name, so check that
        if not Market.asset(new_asset.name):
            Market._asset_dict[new_asset.name] = new_asset
        else:
            Market.asset(new_asset.name).price = new_asset.price

    @staticmethod
    def asset(asset_name: str) -> Asset:
        if isinstance(asset_name, Asset):
            asset_name = asset_name.name
        return Market._asset_dict[asset_name] if asset_name in Market._asset_dict else None

    @staticmethod
    def price(asset_index: int or str):
        return Market.asset(asset_index).price


class Position:
    def __init__(self,
                 asset_name: int or str,
                 quantity: float = 0,
                 price: float = 0):

        asset = Market.asset(asset_name)
        self.quantity = quantity
        self.price = price or asset.price
        self._asset = asset

    @property
    def asset(self):
        return self._asset

    @property
    def assetName(self):
        return self._asset.name

    @property
    def index(self):
        return self.asset.index


class Agent:
    def __init__(self, name: str):

        self.name = name
        self._positions = {}

    def position(self, asset_name: str) -> Position:
        return self._positions[asset_name] if asset_name in self._positions else None

    def asset(self, asset_name: str) -> Asset:
        return self.position(asset_name).asset

    def holdings(self, asset_name: str) -> float:
        """ get this agent's holdings in the specified market, asset pair """
        return self.position(asset_name).quantity if self.position(asset_name) else 0

    def price(self, asset_name: str) -> float:
        """ get this agent's buy-in price for the specified market, asset pair """
        return self.position(asset_name).price if self.position(asset_name) else None

    @property
    def asset_list(self) -> list:
        return [Market.asset(asset) for asset in self._positions.keys()]

    def add_position(self, asset_name: str, quantity: float, price: float = 0):
        """ add specified quantity to the agent's asset holdings in market """
        asset_name = Market.asset(asset_name).name
        if self.position(asset_name):
            self.position(asset_name).quantity += quantity
            return self

        # if (market, asset) position was not found, create one
        newPosition = Position(
            asset_name=asset_name,
            quantity=quantity,
            price=price
        )
        self._positions[asset_name] = newPosition
        return self


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
                 positions: list,  # of positions
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
        self.totalValue = sum([position.quantity * position.price for position in positions])
        self.unique_id = unique_id

        self.shareToken = ShareToken(
            name=ShareToken.token_name(self.unique_id),
            price=self.ratio,
            asset_names=[position.assetName for position in self.positions]
        )
        Market.add_asset(self.shareToken)

    @property
    def ratio(self) -> float:
        return self.positions[1].quantity / self.positions[0].quantity


class Exchange(Market):
    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float,
                 ):

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

    def pool(self, asset_names: list) -> RiskAssetPool:
        """ get the pool containing the specified assets """
        index = self.poolName(asset_names)
        return self._asset_pools_dict[index] if index in self._asset_pools_dict else None

    @property
    def pool_list(self):
        return [pool for pool in self._asset_pools_dict.values()]

    @staticmethod
    def poolName(asset_names: list) -> str:
        return ", ".join(sorted(asset_names))

    def add_liquidity(self, agent: Agent, pool: RiskAssetPool, quantity: float):

        for position in pool.positions:
            position.quantity += quantity
