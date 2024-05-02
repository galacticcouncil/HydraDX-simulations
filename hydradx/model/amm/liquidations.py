from hydradx.model.amm.agents import Agent


class money_market:
    def __init__(
            self,
            liquidity: dict[str: float],
            oracles: dict,
            liquidation_threshold: float or dict[str: float] = 0.7,
            min_ltv: float or dict[str: float] = 0.6,
            liquidation_penalty: float or dict[str: float] = 0.01,
    ):
        self.liquidity = liquidity
        if isinstance(liquidation_threshold, dict):
            self.liquidation_threshold = liquidation_threshold
        else:
            self.liquidation_threshold = {asset: liquidation_threshold for asset in liquidity}
        self.min_ltv = min_ltv
        if isinstance(liquidation_penalty, dict):
            self.liquidation_penalty = liquidation_penalty
        else:
            self.liquidation_penalty = {asset: liquidation_penalty for asset in liquidity}

        self.oracles = oracles  # MoneyMarket should never mutate the oracles
        self.cdps = []
        self.borrowed = {asset: 0 for asset in liquidity}
        if not self.validate():
            raise ValueError("money_market initialization failed.")

    def __repr__(self):
        return f"money_market({self.liquidity}, {self.liquidation_threshold}, {self.liquidation_penalty})"

    def borrow(self, agent: Agent, borrow_asset: str, collateral_asset: str, borrow_amt: float, collateral_amt: float) -> None:
        assert borrow_asset != collateral_asset
        assert borrow_asset in self.liquidity
        assert borrow_amt <= self.liquidity[borrow_asset]
        assert agent.is_holding(collateral_asset, collateral_amt)
        if (borrow_asset, collateral_asset) in self.oracles:
            price = 1 / self.oracles[(borrow_asset, collateral_asset)]
        elif (collateral_asset, borrow_asset) in self.oracles:
            price = self.oracles[(collateral_asset, borrow_asset)]
        else:
            raise ValueError("Oracle price not found.")
        assert price * collateral_amt * self.min_ltv >= borrow_amt
        self.borrowed[borrow_asset] += borrow_amt
        agent.holdings[borrow_asset] += borrow_amt
        agent.holdings[collateral_asset] -= collateral_amt
        cdp = CDP(borrow_asset, collateral_asset, borrow_amt, collateral_amt)
        self.cdps.append((agent.unique_id, cdp))

    def repay(self, agent: Agent, cdp: CDP) -> None:
        assert agent.is_holding(cdp.debt_asset, cdp.debt_amt)


    def validate(self):
        cdp_borrowed = {asset: 0 for asset in self.liquidity}
        for (agent, cdp) in self.cdps:
            cdp_borrowed[cdp.debt_asset] += cdp.debt_amt
        for asset in self.liquidity:
            if asset not in self.borrowed:
                return False
            if asset in cdp_borrowed and self.borrowed[asset] != cdp_borrowed[asset]:
                return False
        return len(self.liquidity) == len(self.borrowed) == len(cdp_borrowed) == len(self.liquidation_threshold) == len(self.liquidation_penalty)


# Dummy CDP class, in which CDP positions are determined to be in liquidation via a boolean
class CDP:
    def __init__(
            self,
            debt_asset,
            collateral_asset,
            debt_amt,
            collateral_amt,
            # in_liquidation: bool = False,
            # oracle_price: float = 0.0,
            # liquidation_threshold: float = 0.7,
            # liquidation_penalty: float = 0.0
    ):
        self.debt_asset = debt_asset
        self.collateral_asset = collateral_asset
        self.debt_amt = debt_amt
        self.collateral_amt = collateral_amt
        # self.in_liquidation = in_liquidation
        self.liquidation_threshold = liquidation_threshold
        self.liquidation_penalty = liquidation_penalty
        # self.oracle_price = oracle_price  # collateral asset denominated in debt asset

        # if oracle_price != 0:
        #     self.in_liquidation = self.is_liquidatable()

    def __repr__(self):
        return f"CDP({self.debt_asset}, {self.collateral_asset}, {self.debt_amt}, {self.collateral_amt}, {self.liquidation_threshold}, {self.liquidation_penalty})"

    def copy(self):
        return CDP(self.debt_asset, self.collateral_asset, self.debt_amt, self.collateral_amt, self.liquidation_threshold, self.liquidation_penalty)

    def liquidate(self, agent: Agent, debt_amt: float, oracle_price: float) -> None:
        if self.is_liquidatable(oracle_price):
            collateral_amt = debt_amt / oracle_price * (1 + self.liquidation_penalty)
            self.liquidate_cdp(agent, debt_amt, collateral_amt)

    def liquidate_cdp(self, agent: Agent, debt_amt: float, collateral_amt: float) -> None:
        if debt_amt > self.debt_amt or debt_amt > agent.holdings[self.debt_asset]:
            raise ValueError("Debt amount exceeds CDP debt or agent holdings.")
        if collateral_amt > self.collateral_amt:
            raise ValueError("Collateral amount exceeds CDP collateral.")
        self.debt_amt -= debt_amt
        agent.holdings[self.debt_asset] -= debt_amt
        self.collateral_amt -= collateral_amt
        if self.collateral_asset not in agent.holdings:
            agent.holdings[self.collateral_asset] = 0
        agent.holdings[self.collateral_asset] += collateral_amt

    def is_liquidatable(self, oracle_price: float) -> bool:
        return self.debt_amt / (self.collateral_amt * oracle_price) > self.liquidation_threshold

    def validate(self) -> bool:
        if self.debt_amt < 0 or self.collateral_amt < 0:
            return False
        if self.debt_asset == self.collateral_asset:
            return False
        return True
