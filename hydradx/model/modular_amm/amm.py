class Asset:
    _ids = 0

    def __init__(self, name: str, price: float):
        """ price is denominated in USD, probably ¯\_(ツ)_/¯ """
        self.name = name
        self.price = price
        self.index = Asset._ids
        Asset._ids += 1


class Market:
    def __init__(self, assets: list, price_denominator: Asset or None = None):
        """ Base class of all markets. Accepts a list of assets and a price denominator """
        self._asset_list = list()
        self._asset_dict = dict()
        for asset in assets:
            self.add_asset(asset)
        if not price_denominator:
            # choose whichever asset has a price of 1
            self.priceDenomination = list(filter(lambda a: a.price == 1, self.asset_list))[0]
        else:
            self.priceDenomination = self.asset(price_denominator)

    def add_asset(self, new_asset: Asset):
        """ add an asset to the market """
        if not self.asset(new_asset):
            new_asset.index = len(self._asset_list)
            self._asset_list.append(new_asset)
            # maintain a reference to both name and index in asset_dict, for convenience
            self._asset_dict[new_asset.index] = new_asset
            self._asset_dict[new_asset.name] = new_asset
            self._asset_dict[new_asset] = new_asset

    def asset(self, index: int or str or Asset) -> Asset:
        return self._asset_dict[index] if index in self._asset_dict else None

    @property
    def asset_list(self) -> list[Asset]:
        return self._asset_list

    def price(self, asset: Asset):
        asset = self.asset(asset)
        return asset.price / self.priceDenomination.price


class Position:
    def __init__(self,
                 asset: Asset,
                 quantity: float = 0,
                 price: float = 0):

        self.quantity = quantity
        self.price = price or asset.price
        self.asset = asset

    @property
    def name(self):
        return self.asset.name

    @property
    def index(self):
        return self.asset.index


class Agent:
    def __init__(self, name: str):

        self.name = name
        self._positions = {}

    def position(self, index: int or str or Asset) -> Position:
        return self._positions[index] if index in self._positions else None

    def asset(self, index: int or str or Asset) -> Asset:
        return self._position[index].asset if index in self._positions else None

    def holdings(self, index: int or str or Asset) -> float:
        """ get this agent's holdings in the specified market, asset pair """
        return self.position(index).quantity if self.position(index) else 0

    def price(self, index: int or str or Asset) -> float:
        """ get this agent's buy-in price for the specified market, asset pair """
        return self.position(index).price if self.position(index) else None

    @property
    def asset_list(self):
        return list(filter(lambda key: isinstance(key, Asset), self._positions))

    def add_position(self, asset: Asset, quantity: float, price: float = 0):
        """ add specified quantity to the agent's asset holdings in market """
        if asset in self._positions:
            self._positions[asset].quantity += quantity
            return self

        # if (market, asset) position was not found, create one
        newPosition = Position(
            asset=asset,
            quantity=quantity,
            price=price
        )
        self._positions[asset] = newPosition
        self._positions[asset.name] = newPosition
        self._positions[asset.index] = newPosition
        return self


class ShareToken(Asset):
    def __init__(self, name: str, price: float, assets: list):
        super().__init__(name, price)
        self.assets = assets

    @staticmethod
    def token_name(asset_str: str):
        return f'pool shares ({asset_str})'


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
        self.unique_id = unique_id or self.poolName([position.asset for position in self.positions])

        self.shareToken = ShareToken(
            name=ShareToken.token_name(self.unique_id),
            price=self.price,
            assets=[position.asset for position in self.positions]
        )

    @property
    def price(self) -> float:
        return self.positions[1].quantity / self.positions[0].quantity

    @staticmethod
    def poolName(assets: list) -> str:
        return ", ".join([asset.name for asset in sorted(assets, key=lambda asset: asset.name)])


class Exchange(Market):
    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float,
                 assets: list,
                 price_denominator: Asset
                 ):

        super().__init__(assets, price_denominator=price_denominator)

        self._asset_pools_list = list()
        self._asset_pools_dict = dict()

        self.tvlCapUSD = tvl_cap_usd
        self.assetFee = asset_fee

    def add_pool(self, positions: list, weight_cap: float = 0):
        """ add an asset pool to the exchange """
        for asset in positions:
            if not self.asset(asset):
                self.add_asset(asset.asset)

        if not self.pool(positions):
            new_pool = RiskAssetPool(
                positions=positions,
                weight_cap=weight_cap
            )
            self._asset_pools_list.append(new_pool)
            # maintain a reference to both name and index in _asset_pools_dict, for convenience
            self._asset_pools_dict[new_pool.unique_id] = new_pool

    def pool(self, assets: list) -> RiskAssetPool:
        """ get the pool containing the specified assets """
        index = RiskAssetPool.poolName(assets)
        return self._asset_pools_dict[index] if index in self._asset_pools_dict else None

    @property
    def pool_list(self) -> list:
        return self._asset_pools_list

    def add_liquidity(self, agent: Agent, pool: RiskAssetPool, quantity: float):

        for position in pool.positions:
            position.quantity += quantity

    def price(self, index: int or str):
        return self.pool(index).price
