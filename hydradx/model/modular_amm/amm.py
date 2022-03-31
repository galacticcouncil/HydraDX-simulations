class RiskAssetPool:
    index: int

    def __init__(self,
                 name: str,
                 quantity: float,
                 lrna_price: float = 1.0,
                 lrna_quantity: float = 0,
                 shares: float = 1.0,
                 weight_cap: float = 0
                 ):
        """
        The state of one asset pool.
        """
        self.name = name
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


class Position:
    def __init__(self, pool: RiskAssetPool, quantity, price: float = 0):
        self.pool = pool
        self.shares = pool.shares * pool.assetQuantity / quantity
        self.price = price or pool.price


class Agent:
    def __init__(self,
                 pool_assets: dict[RiskAssetPool: float],
                 outside_assets: dict[RiskAssetPool, float],
                 lrna: float = 0.0
                 ):
        self.poolAssets = {
            pool: Position(pool, quantity)
            for pool, quantity in pool_assets.items()
        }
        self.outsideAssets = outside_assets
        self.lrna = lrna

    @property
    def s(self) -> dict:
        return_dict = {pool.index: position.shares for pool, position in self.poolAssets.items()}
        return_dict.update({pool.name: position.shares for pool, position in self.poolAssets.items()})
        return return_dict

    @property
    def r(self):
        return_dict = {pool.index: quantity for pool, quantity in self.outsideAssets.items()}
        return_dict.update({pool.name: quantity for pool, quantity in self.outsideAssets.items()})
        return return_dict

    @property
    def p(self):
        return_dict = {pool.index: position.price for pool, position in self.poolAssets.items()}
        return_dict.update({pool.name: position.price for pool, position in self.poolAssets.items()})
        return return_dict

    @property
    def q(self):
        return self.lrna


class Exchange:
    def __init__(self,
                 risk_assets: list[RiskAssetPool],
                 tvl_cap_usd: float,
                 lrna_fee: float,
                 asset_fee: float,
                 preferred_stablecoin: str
                 ):
        self.riskAssets = risk_assets
        self.tvlCapUSD = tvl_cap_usd
        self.lrnaFee = lrna_fee
        self.assetFee = asset_fee
        self.lrnaImbalance = 0
        self.stableCoin = risk_assets[[pool.name for pool in risk_assets].index(preferred_stablecoin)]
        for i, pool in enumerate(self.riskAssets):
            pool.index = i

    def agent(self,
              pool_assets: dict[str: float],
              outside_assets: dict[str: float]
              ) -> Agent:

        return Agent(
            pool_assets={
                self.riskAssets[[pool.name for pool in self.riskAssets].index(name)]: value
                for name, value in pool_assets.items()
            },
            outside_assets={
                self.riskAssets[[pool.name for pool in self.riskAssets].index(name)]: value
                for name, value in outside_assets.items()
            }
        )

    @property
    def totalQ(self) -> float:
        """ the total quantity of LRNA contained in all asset pools """
        return sum([asset.lrnaQuantity for asset in self.riskAssets])

    @property
    def W(self) -> dict:
        """ the percentage of total LRNA contained in each asset pool """
        lrna_total = self.totalQ
        return_dict = {pool.index: pool.lrnaQuantity / lrna_total for pool in self.riskAssets}
        return_dict.update({pool.name: pool.lrnaQuantity / lrna_total for pool in self.riskAssets})
        return return_dict

    @property
    def Q(self) -> dict:
        """ the absolute quantity of LRNA in each asset pool """
        return_dict = {pool.index: pool.lrnaQuantity for pool in self.riskAssets}
        return_dict.update({pool.name: pool.lrnaQuantity for pool in self.riskAssets})
        return return_dict

    @property
    def R(self) -> dict:
        """ quantity of risk asset in each asset pool """
        return_dict = {pool.index: pool.assetQuantity for pool in self.riskAssets}
        return_dict.update({pool.name: pool.assetQuantity for pool in self.riskAssets})
        return return_dict

    @property
    def B(self) -> dict:
        """ quantity of liquidity provider shares in each pool owned by the protocol """
        return_dict = {pool.index: pool.sharesOwnedByProtocol for pool in self.riskAssets}
        return_dict.update({pool.name: pool.sharesOwnedByProtocol for pool in self.riskAssets})
        return return_dict

    @property
    def P(self) -> dict:
        """ price of each asset denominated in LRNA """
        return_dict = {pool.index: pool.price for pool in self.riskAssets}
        return_dict.update({pool.name: pool.price for pool in self.riskAssets})
        return return_dict

    @property
    def L(self):
        return self.lrnaImbalance

    @property
    def C(self):
        """ soft cap on total value locked, denominated in preferred stablecoin """
        return self.tvlCapUSD

    @L.setter
    def L(self, value):
        self.lrnaImbalance = value
