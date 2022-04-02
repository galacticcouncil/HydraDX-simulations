class Asset:
    def __init__(self, name: str, index: int, market_price_usd: float):
        self.name = name
        self.index = index
        self.price = market_price_usd


class Market:
    def __init__(self, assets: list[Asset]):
        self._asset_list = list()
        self._asset_dict = dict()
        for asset in assets:
            self.add_asset(asset)

    def add_asset(self, new_asset: Asset):
        """ add an asset to the exchange """
        new_asset.index = len(self._asset_list)
        self._asset_list.append(new_asset)
        # maintain a reference to both name and index in asset_dict, for convenience
        self._asset_dict[new_asset.index] = new_asset
        self._asset_dict[new_asset.name] = new_asset

    def asset(self, index) -> Asset:
        return self._asset_dict[index]

    @property
    def asset_list(self) -> list[Asset]:
        return self._asset_list

    def price(self, index):
        return self._asset_dict[index].price

    @staticmethod
    def relativePrice(asset: Asset, denominator: Asset):
        return asset.price / denominator.price


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
    def __init__(self,
                 name: str,
                 positions: dict[Market: dict[str: float]],
                 ):

        self.name = name

        self._positions: dict[tuple[Market, Asset]: Position] = {}
        for market in positions:
            for asset in market:
                self._positions[(market, asset)] = Position(
                    market=market,
                    asset=market.asset(asset),
                    price=market.price(asset),
                    quantity=positions[market][asset]
                )

    def holdings(self, market: Market, index: int or str) -> float:
        """ get this agent's holdings in the specified market, asset pair """
        for (p_market, asset), position in self._positions.items():
            if p_market == market and market.asset(index) == asset:
                return self._positions[(market, asset)].quantity
        return 0

    def add_asset(self, market: Market, index: int or str, delta: float):
        """ add delta to the agent's asset holdings in market """
        for (p_market, asset), position in self._positions.items():
            if p_market == market and market.asset(index) == asset:
                self._positions[(market, asset)].quantity += delta
                return
        # if (market, asset) position was not found, create one
        asset = market.asset(index)
        self._positions[(market, asset)] = Position(
            market=market,
            asset=asset,
            price=market.price(asset),
            quantity=delta
        )


class RiskAssetPool:
    index: int

    def __init__(self,
                 asset: Asset,
                 quantity: float,
                 lrna_price: float = 1.0,
                 lrna_quantity: float = 0,
                 shares: float = 1.0,
                 weight_cap: float = 0
                 ):
        """
        The state of one asset pool.
        """
        self.asset = asset
        self.assetQuantity = quantity
        if lrna_quantity and lrna_price:
            raise "Specify either LRNA quantity or price, but not both."
        if not (lrna_quantity or lrna_price):
            raise "Either LRNA quantity or price must be specified."
        self.lrnaQuantity = lrna_quantity or quantity * lrna_price
        self.shares = shares
        self.weightCap = weight_cap
        self.sharesOwnedByProtocol = self.shares

    @property
    def price(self):
        return self.lrnaQuantity / self.assetQuantity


class Exchange(Market):
    def __init__(self,
                 risk_assets: list[RiskAssetPool],
                 tvl_cap_usd: float,
                 lrna_fee: float,
                 asset_fee: float,
                 preferred_stablecoin: str
                 ):

        self.asset_pools = {}
        super().__init__([])
        for pool in risk_assets:
            self.add_asset(pool.asset)

        self.tvlCapUSD = tvl_cap_usd
        self.lrnaFee = lrna_fee
        self.assetFee = asset_fee
        self.lrnaImbalance = 0
        self.stableCoin = risk_assets[[pool.asset.name for pool in risk_assets].index(preferred_stablecoin)]
