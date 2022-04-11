class Asset:
    index: int

    def __init__(self, name: str, price: float):
        """ price is denominated in USD, probably ¯\_(ツ)_/¯ """
        self.name = name
        self.price = price


class Market:
    def __init__(self, assets: list, price_denomination: Asset or None = None):
        """ Base class of all markets. Accepts a list of assets and a price denominator """
        self._asset_list = list()
        self._asset_dict = dict()
        for asset in assets:
            self.add_asset(asset)
        if not price_denomination:
            # choose whichever asset has a price of 1
            self.priceDenomination = list(filter(lambda a: a.price == 1, self.asset_list))[0]
        else:
            self.priceDenomination = self.asset(price_denomination)

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

    def holdings(self, index: int or str or Asset) -> float:
        """ get this agent's holdings in the specified market, asset pair """
        return self._positions[index]

    def price(self, index: int or str or Asset) -> float:
        """ get this agent's buy-in price for the specified market, asset pair """
        return self._positions[index].price

    @property
    def asset_list(self):
        return [position.asset for position in self._positions]

    def add_position(self, asset: Asset, quantity: float, price: float = 0):
        """ add specified quantity to the agent's asset holdings in market """
        if self._positions[asset]:
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


class RiskAssetPool:
    index: int

    def __init__(self,
                 positions: list,  # of positions
                 weight_cap: float = 1
                 ):
        """
        The state of one asset pool.
        """
        self.positions = positions
        self.weightCap = weight_cap
        self.shares = self.positions[0].quantity
        self.sharesOwnedByProtocol = self.shares
        self.totalValue = 0

    @property
    def price(self) -> float:
        return self.positions[0].quantity / self.positions[1].quantity


class Exchange(Market):
    def __init__(self,
                 tvl_cap_usd: float,
                 asset_fee: float,
                 assets: list = []
                 ):

        super().__init__(assets)

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
            self._asset_pools_dict[set(position.asset for position in positions)] = new_pool

    def pool(self, *args) -> RiskAssetPool:
        index = set(args)
        return self._asset_pools_dict[index] if index in self._asset_pools_dict else None

    @property
    def pool_list(self) -> list:
        return self._asset_pools_list

    # def set_initial_liquidity(self, asset: Asset, agent: Agent or None, quantity: float):
    #     # (naively) add asset to pool
    #     if not self.pool(asset):
    #         self.add_pool(asset, quantity)
    #     else:
    #         delta_s = self.pool(asset).shares * quantity / self.pool(asset).assetQuantity \
    #             if self.pool(asset).assetQuantity > 0 else quantity
    #         self.pool(asset).shares += delta_s
    #         if not agent:
    #             self.pool(asset).sharesOwnedByProtocol += delta_s
    #         self.pool(asset).assetQuantity += quantity
    #         self.pool(asset).lrnaQuantity += quantity * asset.price / self.lrna.price

    def price(self, index: int or str):
        return self.pool(index).price
