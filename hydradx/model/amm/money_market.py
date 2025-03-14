import copy
from .agents import Agent


class CDP:
    def __init__(
            self,
            debt: dict[str: float],
            collateral: dict[str: float],
            liquidation_threshold: float = None,
            agent=None
    ):
        self.debt: dict[str: float] = debt
        self.collateral: dict[str: float] = collateral
        self.asset_list = list(debt.keys() | collateral.keys())
        self.liquidation_threshold = liquidation_threshold
        if agent is not None:
            self.agent = agent
        else:
            self.agent = Agent()

    def __repr__(self):
        newline = '\n'
        return (
            f"CDP("
            f"    debt:"
            f"{newline.join([f"        {tkn}: {self.debt[tkn]}" for tkn in self.debt.keys()])}\n\n"
            f"    collateral:"
            f"{newline.join([f"        {tkn}: {self.collateral[tkn]}" for tkn in self.collateral.keys()])}\n\n"
        )

    def copy(self):
        return CDP(
            self.debt,
            self.collateral,
            self.liquidation_threshold
        )

    def validate(self) -> bool:
        if min(self.debt.values) < 0 or min(self.collateral.values()) < 0:
            return False
        # if sorted(list(self.debt_assets.keys())) == sorted(list(self.collateral_assets.keys)):
        #     return False
        return True


class MoneyMarketAsset:
    def __init__(
            self,
            name: str,
            price: float,
            liquidity: float,
            liquidation_bonus: float,
            liquidation_threshold: float,
            ltv: float,
            emode_liquidation_bonus: float = 0,
            emode_liquidation_threshold: float = 0,
            emode_ltv: float = 0,
            emode_label: str = '',
    ):
        self.name = name
        self.price = price
        self.liquidity = liquidity
        self.liquidation_bonus = liquidation_bonus
        self.liquidation_threshold = liquidation_threshold
        self.ltv = ltv
        self.emode_liquidation_bonus = emode_liquidation_bonus
        self.emode_liquidation_threshold = emode_liquidation_threshold
        self.emode_ltv = emode_ltv
        self.emode_label = emode_label


class MoneyMarket:
    def __init__(
            self,
            assets: list[MoneyMarketAsset],
            cdps: list = None,
            full_liquidation_threshold: float = 0.95,  # this should maybe be per-asset or per-asset-pair
            close_factor: float = 0.5
    ):
        self.liquidity = {
            asset.name: asset.liquidity
            for asset in assets
        }
        self.borrowed = {asset: 0 for asset in self.liquidity}
        self.asset_list = list(self.liquidity.keys())
        self.prices = {
            asset.name: asset.price
            for asset in assets
        }  # MoneyMarket should never mutate the oracles

        self.ltv = {
            (asset1.name, asset2.name):
            asset1.emode_ltv if asset1.emode_label and asset1.emode_label == asset2.emode_label else asset1.emode_ltv
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }
        self.liquidation_bonus = {
            (asset1.name, asset2.name):
            asset1.emode_liquidation_bonus if asset1.emode_label and asset1.emode_label == asset2.emode_label else asset1.liquidation_bonus
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }
        self.liquidation_threshold = {
            (asset1.name, asset2.name):
            asset1.emode_liquidation_threshold if asset1.emode_label and asset1.emode_label == asset2.emode_label else asset1.liquidation_threshold
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }

        self.cdps: list[CDP] = [] if cdps is None else cdps
        for cdp in self.cdps:
            if not cdp.liquidation_threshold:
                cdp.liquidation_threshold = self.cdp_liquidation_threshold(cdp)

        for cdp in self.cdps:
            for tkn in cdp.debt:
                self.borrowed[tkn] += cdp.debt[tkn]

        self.partial_liquidation_pct = close_factor
        self.full_liquidation_threshold = full_liquidation_threshold
        if not self.validate():
            raise ValueError("money_market initialization failed.")
        self.fail = ''

    def __repr__(self):
        return f"money_market({self.liquidity}, {self.liquidation_threshold}, {self.liquidation_bonus})"

    def copy(self):
        return copy.deepcopy(self)

    def fail_transaction(self, fail: str):
        self.fail = fail
        return self

    def get_oracle_price(self, tkn: str):
        if self.prices[tkn] is not None:
            return self.prices[tkn]
        else:
            raise ValueError("Oracle price not found.")

    def get_cdps(self, collateral_tkn: str = None, debt_tkn: str = None):
        return [cdp for cdp in self.cdps if (
                (collateral_tkn is None or collateral_tkn in cdp.collateral) and
                (debt_tkn is None or debt_tkn in cdp.debt))]

    def get_health_factor(self, cdp: CDP) -> float:
        prices = {tkn: self.get_oracle_price(tkn) for tkn in cdp.collateral.keys() | cdp.debt.keys()}
        health_factor = (
            sum([
                cdp.collateral[c_tkn] * prices[c_tkn]
                for c_tkn in cdp.collateral
            ]) * self.cdp_liquidation_threshold(cdp)
            / sum([
            cdp.debt[d_tkn] * prices[d_tkn]
                for d_tkn in cdp.debt
            ])
        )
        return health_factor

    def get_ltv(self, collateral_tkn: str, debt_tkn: str) -> float:
        return self.ltv[(collateral_tkn, debt_tkn)] if (collateral_tkn, debt_tkn) in self.ltv else 0

    def get_liquidation_bonus(self, collateral_tkn: str, debt_tkn: str) -> float:
        return self.liquidation_bonus[(collateral_tkn, debt_tkn)] if (collateral_tkn, debt_tkn) in self.liquidation_bonus else 0

    def is_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < 1

    def is_fully_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < self.full_liquidation_threshold

    def get_liquidation_threshold(self, collateral_asset, debt_asset) -> float:
        """Get the liquidation threshold for a collateral-debt asset pair."""
        return self.liquidation_threshold.get((collateral_asset, debt_asset), 0)

        def borrow(self, borrow_asset: str, collateral_asset: str, borrow_amt: float,
               collateral_amt: float) -> None:
        assert borrow_asset != collateral_asset
        assert borrow_asset in self.liquidity
        assert borrow_amt <= self.liquidity[borrow_asset] - self.borrowed[borrow_asset]
        # assert agent.is_holding(collateral_asset, collateral_amt)
        price = self.get_oracle_price(collateral_asset) / self.get_oracle_price(borrow_asset)
        assert price * collateral_amt * self.get_ltv(collateral_asset, borrow_asset) >= borrow_amt
        self.borrowed[borrow_asset] += borrow_amt
        # if borrow_asset not in agent.holdings:
        #     agent.holdings[borrow_asset] = 0
        # agent.holdings[borrow_asset] += borrow_amt
        # agent.holdings[collateral_asset] -= collateral_amt
        cdp = CDP({borrow_asset: borrow_amt}, {collateral_asset: collateral_amt})
        self.cdps.append(cdp)

    def repay(self, cdp_i: int) -> None:
        cdp = self.cdps[cdp_i]
        # agent = self.cdps[cdp_i].agent
        # assert agent.is_holding(cdp.debt_assets, cdp.debt_amt)
        # agent.holdings[cdp.debt_assets] -= cdp.debt_amt
        for tkn in cdp.debt:
            self.borrowed[tkn] -= cdp.debt[tkn]
        # if cdp.collateral_assets not in agent.holdings:
        #     agent.holdings[cdp.collateral_assets] = 0
        # agent.holdings[cdp.collateral_assets] += cdp.collateral_amt
        self.cdps.pop(cdp_i)

    def get_maximum_repayment(self, cdp: CDP, debt_asset: str) -> float:
        assert debt_asset in cdp.debt
        if self.is_fully_liquidatable(cdp):
            return cdp.debt[debt_asset]
        else:
            debt_value = cdp.debt[debt_asset] * self.get_oracle_price(debt_asset)
            return min(
                cdp.debt[debt_asset],
                debt_value * self.partial_liquidation_pct / self.get_oracle_price(debt_asset)
            )

    def calculate_liquidation(self, cdp: CDP, debt_asset: str, delta_debt: float = -1) -> dict[str: float]:
        assert debt_asset in cdp.debt
        if delta_debt < 0:
            delta_debt = self.get_maximum_repayment(cdp, debt_asset)
        debt_value = delta_debt * self.get_oracle_price(debt_asset)
        returns = {}
        for collateral_asset in sorted(
                cdp.collateral.keys(),
                key=lambda x: self.get_liquidation_bonus(x, debt_asset),
                reverse=True
        ):
            collateral_value = (
                cdp.collateral[collateral_asset] * self.get_oracle_price(collateral_asset)
                / (1 + self.get_liquidation_bonus(collateral_asset, debt_asset))
            )
            repay_amount = (  # amount of debt repayment that can be covered by this collateral
                delta_debt if debt_value < collateral_value
                else delta_debt * collateral_value / debt_value
            )
            returns[collateral_asset] = (
                cdp.collateral[collateral_asset] if repay_amount < delta_debt
                else min(
                    repay_amount * self.get_oracle_price(debt_asset) / self.get_oracle_price(collateral_asset)
                    * (1 + self.get_liquidation_bonus(collateral_asset, debt_asset)),
                    cdp.collateral[collateral_asset]
                )
            )
        return returns

    def liquidate(self, cdp: CDP, debt_asset: str, agent: Agent):
        if not self.is_liquidatable(cdp):
            return
        repay_amount = (
            cdp.debt[debt_asset] if self.is_fully_liquidatable(cdp)
            else min(
                cdp.debt[debt_asset],
                cdp.debt[debt_asset] * self.partial_liquidation_pct
            )
        )

        collateral = self.calculate_liquidation(cdp, debt_asset, repay_amount)
        for collateral_asset in collateral:
            if not agent.validate_holdings(collateral_asset, collateral[collateral_asset]):
                return self.fail_transaction(f"Agent doesn't have enough {collateral_asset}.")
        for collateral_asset in collateral:
            agent.add(collateral_asset, collateral[collateral_asset])
            cdp.collateral[collateral_asset] -= collateral[collateral_asset]

        agent.remove(debt_asset, repay_amount)
        cdp.debt[debt_asset] -= repay_amount
        cdp.liquidation_threshold = self.cdp_liquidation_threshold(cdp)
        return self

    def validate(self):
        cdp_borrowed = {asset: 0 for asset in self.liquidity}
        if self.partial_liquidation_pct < 0 or self.partial_liquidation_pct > 1:
            return False
        for cdp in self.cdps:
            for tkn in cdp.debt:
                if tkn not in self.liquidity:
                    return False
                if cdp.debt[tkn] < 0:
                    return False
                if cdp.debt[tkn] > self.liquidity[tkn]:
                    return False
                cdp_borrowed[tkn] += cdp.debt[tkn]
        for asset in self.liquidity:
            if asset not in self.borrowed:
                return False
            if asset in cdp_borrowed and self.borrowed[asset] != cdp_borrowed[asset]:
                return False
            for asset2 in [asset2 for asset2 in self.liquidity if asset2 != asset]:
                if (asset, asset2) not in self.ltv:
                    return False
                if (asset, asset2) not in self.liquidation_bonus:
                    return False
                if (asset, asset2) not in self.liquidation_threshold:
                    return False
        for k in self.liquidation_threshold:
            # if k not in self.full_liquidation_threshold:
            #     return False
            if self.liquidation_threshold[k] > self.full_liquidation_threshold:
                return False
        if not (len(self.liquidity) == len(self.borrowed) == len(cdp_borrowed) ):
            return False
        if not(len(self.liquidation_threshold) == len(self.liquidation_bonus) == len(self.ltv)):
            return False
        return True
