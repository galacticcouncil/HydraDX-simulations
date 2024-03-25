from hydradx.model.amm.agents import Agent


# Dummy CDP class, in which CDP positions are determined to be in liquidation via a boolean
class CDP:
    def __init__(self, debt_asset, collateral_asset, debt_amt, collateral_amt, in_liquidation):
        self.debt_asset = debt_asset
        self.collateral_asset = collateral_asset
        self.debt_amt = debt_amt
        self.collateral_amt = collateral_amt
        self.in_liquidation = in_liquidation

    def __repr__(self):
        return f"CDP({self.debt_asset}, {self.collateral_asset}, {self.debt_amt}, {self.collateral_amt}, {self.in_liquidation})"

    def copy(self):
        return CDP(self.debt_asset, self.collateral_asset, self.debt_amt, self.collateral_amt, self.in_liquidation)

    def liquidate_cdp(self, agent: Agent, debt_amt: float, collateral_amt: float) -> None:
        if debt_amt > self.debt_amt or debt_amt > agent.holdings[self.debt_asset]:
            raise
        if collateral_amt > self.collateral_amt:
            raise
        self.debt_amt -= debt_amt
        agent.holdings[self.debt_asset] -= debt_amt
        self.collateral_amt -= collateral_amt
        agent.holdings[self.collateral_asset] += collateral_amt

    def validate(self) -> bool:
        if self.debt_amt < 0 or self.collateral_amt < 0:
            return False
        if self.debt_asset == self.collateral_asset:
            return False
        return True
