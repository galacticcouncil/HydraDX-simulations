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
    def __init__(self, pool: RiskAssetPool, shares: float, price: float):
        self.pool_index = pool.index
        self.shares = shares
        self.price = price
        self.pool = pool


class Agent:
    def __init__(self,
                 name: str,
                 pool_assets: dict[RiskAssetPool: float],
                 outside_assets: dict[RiskAssetPool, float],
                 lrna: float = 0.0
                 ):
        self.poolAssets: dict[int: Position] = {
            pool.index: Position(pool, quantity)
            for pool, quantity in pool_assets.items()
        }
        self.poolAssets.update({
            pool.name: Position(pool, quantity)
            for pool, quantity in pool_assets.items()
        })
        self.outsideAssets = {
            pool.index: value for pool, value in outside_assets.items()
        }
        self.outsideAssets.update({
            pool.name: value for pool, value in outside_assets.items()
        })
        self.lrna = lrna
        self.name = name

    def s(self, index: int or str) -> float:
        return self.poolAssets[index].shares

    def r(self, index: int or str) -> float:
        return self.outsideAssets[index]

    def p(self, index: int or str) -> float:
        return self.poolAssets[index].price

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
        self.riskAssets: dict[int or str: RiskAssetPool]
        self.riskAssets = {i: pool for i, pool in enumerate(risk_assets)}
        self.riskAssets.update({pool.name: pool for pool in risk_assets})
        self.tvlCapUSD = tvl_cap_usd
        self.lrnaFee = lrna_fee
        self.assetFee = asset_fee
        self.lrnaImbalance = 0
        self.stableCoin = risk_assets[[pool.name for pool in risk_assets].index(preferred_stablecoin)]
        for i, pool in enumerate(self.riskAssets):
            pool.index = i

    def agent(self,
              pool_assets: dict[str: float],
              outside_assets: dict[str: float],
              name: str
              ) -> Agent:

        return Agent(
            name=name,
            pool_assets={
                self.riskAssets[[pool.name for pool in self.riskAssets].index(name)]: value
                for name, value in pool_assets.items()
            },
            outside_assets={
                self.riskAssets[[pool.name for pool in self.riskAssets].index(name)]: value
                for name, value in outside_assets.items()
            },
            lrna=outside_assets['LRNA'] if 'LRNA' in outside_assets else 0
        )

    @property
    def totalQ(self) -> float:
        """ the total quantity of LRNA contained in all asset pools """
        return sum([asset.lrnaQuantity for asset in self.riskAssets])

    def W(self, index: int or str) -> float:
        """ the percentage of total LRNA contained in each asset pool """
        lrna_total = self.totalQ
        return self.riskAssets[index].lrnaQuantity / lrna_total

    def Q(self, index: int or str) -> float:
        """ the absolute quantity of LRNA in each asset pool """
        return self.riskAssets[index].lrnaQuantity

    @Q.setter
    def Q(self, index: int or str, value):
        self.riskAssets[index].lrnaQuantity = value

    def R(self, index: int or str) -> float:
        """ quantity of risk asset in each asset pool """
        return self.riskAssets[index].assetQuantity

    def B(self, index: int or str) -> float:
        """ quantity of liquidity provider shares in each pool owned by the protocol """
        return self.riskAssets[index].sharesOwnedByProtocol

    def P(self, index: int or str) -> float:
        """ price of each asset denominated in LRNA """
        return self.riskAssets[index].price

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


class State:
    exchange: Exchange
    agents: list[Agent]
