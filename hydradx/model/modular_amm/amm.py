class Asset:
    index: int

    def __init__(self, name: str, price: float):
        """ price is denominated in USD, probably ¯\_(ツ)_/¯ """
        self.name = name
        self.price = price


class Market:
    def __init__(self, assets: list[Asset], price_denomination: Asset or None = None):
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
                 market: Market,
                 quantity: float = 0,
                 price: float = 0):

        self.quantity = quantity
        self.price = price or market.price(asset)
        self.market = market
        self.asset = asset


class Agent:
    def __init__(self, name: str):

        self.name = name
        self._positions = {}

    def holdings(self, market: Market, index: int or str) -> float:
        """ get this agent's holdings in the specified market, asset pair """
        for (p_market, asset), position in self._positions.items():
            if p_market == market and market.asset(index) == asset:
                return self._positions[(market, asset)].quantity
        return 0

    def price(self, market: Market, index: int or str) -> float:
        """ get this agent's buy-in price for the specified market, asset pair """
        for (p_market, asset), position in self._positions.items():
            if p_market == market and market.asset(index) == asset:
                return self._positions[(market, asset)].price
        return 0

    def position(self, market, asset: Asset or int or str) -> Position:
        if not type(asset) == Asset:
            asset = market.asset(asset)
        return self._positions[(market, asset)]

    def add_asset(self, market: Market, asset: Asset, quantity: float):
        """ add specified quantity to the agent's asset holdings in market """
        for (p_market, p_asset), position in self._positions.items():
            if p_market == market and p_asset == asset:
                self._positions[(market, asset)].quantity += quantity
                return
        # if (market, asset) position was not found, create one
        self._positions[(market, asset)] = Position(
            market=market,
            asset=asset,
            price=market.price(asset),
            quantity=quantity
        )
        return self


class RiskAssetPool:
    index: int

    def __init__(self,
                 asset: Asset,
                 quantity: float,
                 lrna_quantity: float = 0,
                 weight_cap: float = 0
                 ):
        """
        The state of one asset pool.
        """
        self.asset = asset
        self.assetQuantity = quantity
        self.lrnaQuantity = lrna_quantity
        self.shares = self.assetQuantity
        self.weightCap = weight_cap
        self.sharesOwnedByProtocol = self.shares

    @property
    def lrna_price(self):
        return self.lrnaQuantity / self.assetQuantity


class Exchange(Market):
    def __init__(self,
                 lrna: Asset,
                 tvl_cap_usd: float,
                 lrna_fee: float,
                 asset_fee: float,
                 preferred_stablecoin: Asset,
                 initial_liquidity: float = 0,
                 ):

        super().__init__([preferred_stablecoin])

        self._asset_pools_list = list()
        self._asset_pools_dict = dict()

        self.lrna = lrna
        self.stableCoin = preferred_stablecoin
        self.add_pool(preferred_stablecoin, initial_liquidity)

        self.tvlCapUSD = tvl_cap_usd
        self.lrnaFee = lrna_fee
        self.assetFee = asset_fee
        self.lrnaImbalance = 0

    def add_pool(self, new_asset: Asset, quantity: float, weight_cap: float = 0):
        """ add an asset pool to the exchange """
        if not self.asset(new_asset):
            self.add_asset(new_asset)
        if not self.pool(new_asset):
            new_pool = RiskAssetPool(
                asset=new_asset,
                quantity=quantity,
                lrna_quantity=quantity * new_asset.price / self.lrna.price,
                weight_cap=weight_cap
            )
            self._asset_pools_list.append(new_pool)
            # maintain a reference to both name and index in _asset_pools_dict, for convenience
            self._asset_pools_dict[new_pool.asset.index] = new_pool
            self._asset_pools_dict[new_pool.asset.name] = new_pool
            self._asset_pools_dict[new_pool.asset] = new_pool
        else:
            self.set_initial_liquidity(new_asset, agent=None, quantity=quantity)

    def pool(self, index: int or str or Asset) -> RiskAssetPool:
        return self._asset_pools_dict[index] if index in self._asset_pools_dict else None

    @property
    def pool_list(self) -> list[RiskAssetPool]:
        return self._asset_pools_list

    def set_initial_liquidity(self, asset: Asset, agent: Agent or None, quantity: float):
        # (naively) add asset to pool
        if not self.pool(asset):
            self.add_pool(asset, quantity)
        else:
            delta_s = self.pool(asset).shares * quantity / self.pool(asset).assetQuantity \
                if self.pool(asset).assetQuantity > 0 else quantity
            self.pool(asset).shares += delta_s
            if not agent:
                self.pool(asset).sharesOwnedByProtocol += delta_s
            self.pool(asset).assetQuantity += quantity
            self.pool(asset).lrnaQuantity += quantity * asset.price / self.lrna.price

    def price(self, index: int or str):
        """ Exchange prices are measured in LRNA. """
        return self.pool(index).lrna_price
