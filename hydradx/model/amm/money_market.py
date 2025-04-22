import copy
from .agents import Agent


class CDP:
    def __init__(
            self,
            debt: dict[str: float],
            collateral: dict[str: float],
            liquidation_threshold: float = None,
            health_factor: float = None,
            e_mode: int = None
    ):
        self.debt: dict[str: float] = {tkn: debt[tkn] for tkn in debt}
        self.collateral: dict[str: float] = {tkn: collateral[tkn] for tkn in collateral}
        self.asset_list = list(debt.keys() | collateral.keys())
        self.liquidation_threshold = liquidation_threshold
        self.health_factor = health_factor
        self.fix_liquidation_threshold = True if liquidation_threshold is not None else False
        self.e_mode = e_mode

    def __repr__(self):
        newline = '\n'
        return (
            f"CDP:\n"
            f"    debt:\n"
            f"{newline.join([f'        {tkn}: {self.debt[tkn]}' for tkn in self.debt.keys()])}\n"
            f"    collateral:\n"
            f"{newline.join([f'        {tkn}: {self.collateral[tkn]}' for tkn in self.collateral.keys()])}\n"
        )

    def copy(self):
        return CDP(
            self.debt,
            self.collateral,
            liquidation_threshold=self.liquidation_threshold if self.fix_liquidation_threshold else None,
            health_factor=self.health_factor
        )

    def validate(self) -> bool:
        if min(self.debt.values()) < 0 or min(self.collateral.values()) < 0:
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
            emode_liquidation_bonus: float = None,
            emode_liquidation_threshold: float = None,
            emode_ltv: float = None,
            emode_label: str = '',
    ):
        self.name = name
        self.price = price
        self.liquidity = liquidity
        self.liquidation_bonus = liquidation_bonus
        self.liquidation_threshold = liquidation_threshold
        self.ltv = ltv
        self.emode_liquidation_bonus = emode_liquidation_bonus or liquidation_bonus
        self.emode_liquidation_threshold = emode_liquidation_threshold or liquidation_threshold
        self.emode_ltv = emode_ltv or ltv
        self.emode_label = emode_label


class MoneyMarket:
    def __init__(
            self,
            assets: list[MoneyMarketAsset],
            cdps: list = None,
            full_liquidation_threshold: float = 0.95,
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
        }
        self.assets = {asset.name: asset for asset in assets}

        self.cdps: list[CDP] = [] if cdps is None else cdps
        for cdp in self.cdps:
            if not cdp.liquidation_threshold:
                cdp.liquidation_threshold = self.cdp_liquidation_threshold(cdp)
            if cdp.health_factor is None:
                cdp.health_factor = self.get_health_factor(cdp)

            for tkn in cdp.debt:
                self.borrowed[tkn] += cdp.debt[tkn]

        self.partial_liquidation_pct = close_factor
        self.full_liquidation_threshold = full_liquidation_threshold
        if not self.validate():
            raise ValueError("money_market initialization failed.")
        self.fail = ''

    def __repr__(self):
        total_collateral = {tkn: sum([cdp.collateral[tkn] if tkn in cdp.collateral else 0 for cdp in self.cdps])
            for tkn in self.asset_list
        }
        return (
            f"money_market("
            f"    liquidity: {self.liquidity}\n"
            f"    liquidation threshold: {self.liquidation_threshold}\n"
            f"    liquidation bonus: {self.liquidation_bonus})\n"
            f"    total borrowed: {self.borrowed}\n"
            f"    total collateral: {total_collateral}\n"
        )

    def copy(self):
        return copy.deepcopy(self)

    def add_new_asset(self, new_asset: MoneyMarketAsset):
        self.liquidity[new_asset.name] = new_asset.liquidity
        self.prices[new_asset.name] = new_asset.price
        self.borrowed[new_asset.name] = 0
        self.asset_list.append(new_asset.name)
        self.assets[new_asset.name] = new_asset

    def fail_transaction(self, fail: str):
        self.fail = fail
        return self

    def price(self, tkn: str, numeraire: str = None):
        if numeraire is None:
            return self.prices[tkn]
        else:
            return self.prices[tkn] / self.prices[numeraire]

    def get_cdps(self, collateral_tkn: str = None, debt_tkn: str = None):
        return [cdp for cdp in self.cdps if (
                (collateral_tkn is None or collateral_tkn in cdp.collateral) and
                (debt_tkn is None or debt_tkn in cdp.debt))]

    def get_health_factor(self, cdp: CDP) -> float:
        prices = {tkn: self.price(tkn) for tkn in cdp.collateral.keys() | cdp.debt.keys()}
        debt_total = sum([
            cdp.debt[d_tkn] * prices[d_tkn]
                for d_tkn in cdp.debt
        ])
        if debt_total == 0:
            return 0
        health_factor = (
            sum([
                cdp.collateral[c_tkn] * prices[c_tkn]
                for c_tkn in cdp.collateral
            ]) * self.cdp_liquidation_threshold(cdp)
            / debt_total
        )
        return health_factor

    def get_ltv(self, collateral_tkn: str, debt_tkn: str, e_mode: str = 'None') -> float:
        if self.assets[collateral_tkn].emode_label == self.assets[debt_tkn].emode_label == e_mode:
            return self.assets[collateral_tkn].emode_ltv
        else:
            return self.assets[collateral_tkn].ltv

    def get_liquidation_bonus(self, collateral_tkn: str, debt_tkn: str, e_mode: str = 'None') -> float:
        if self.assets[collateral_tkn].emode_label == self.assets[debt_tkn].emode_label == e_mode:
            return self.assets[collateral_tkn].emode_liquidation_bonus
        else:
            return self.assets[collateral_tkn].liquidation_bonus

    def get_liquidation_threshold(self, collateral_tkn, debt_tkn, e_mode: str = 'None') -> float:
        """Get the liquidation threshold for a collateral-debt asset pair."""
        if self.assets[collateral_tkn].emode_label == self.assets[debt_tkn].emode_label == e_mode:
            return self.assets[collateral_tkn].emode_liquidation_threshold
        else:
            return self.assets[collateral_tkn].liquidation_threshold

    def is_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral.values()) == 0 or sum(cdp.debt.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < 1

    def is_fully_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral.values()) == 0 or sum(cdp.debt.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < self.full_liquidation_threshold

    def is_toxic(self, cdp: CDP) -> bool:
        return self.value_assets(cdp.collateral) < self.value_assets(cdp.debt)

    def cdp_liquidation_threshold(self, cdp: CDP) -> float:
        """
        Calculate the liquidation threshold for a CDP with multiple assets.

        Args:
            cdp: A CDP instance with collateral and debt assets

        Returns:
            float: The calculated liquidation threshold as a decimal (0.0-1.0)
        """

        threshold = {tkn: 0 for tkn in cdp.collateral}
        debt_values = {tkn: cdp.debt[tkn] * self.price(tkn) for tkn in cdp.debt}
        collateral_values = {tkn: cdp.collateral[tkn] * self.price(tkn) for tkn in cdp.collateral}
        total_debt = sum(debt_values.values())
        total_collateral = sum(collateral_values.values())
        if total_collateral == 0 or total_debt == 0:
            return 0
        for tkn in threshold:
            threshold[tkn] = sum(
                [value * self.get_liquidation_threshold(tkn, debt_tkn, cdp.e_mode) for debt_tkn, value in
                 debt_values.items()]) / total_debt
        threshold_weights = {tkn: value / total_collateral for tkn, value in collateral_values.items()}
        return sum([value * threshold[tkn] for tkn, value in collateral_values.items()]) / total_collateral

    def borrow(self, agent: Agent, borrow_asset: str, collateral_asset: str, borrow_amt: float,
               collateral_amt: float) -> CDP:
        assert borrow_asset != collateral_asset
        assert borrow_asset in self.liquidity
        assert borrow_amt <= self.liquidity[borrow_asset] - self.borrowed[borrow_asset]
        assert agent.validate_holdings(collateral_asset, collateral_amt)
        price = self.price(collateral_asset) / self.price(borrow_asset)
        if price * collateral_amt * self.get_ltv(collateral_asset, borrow_asset, self.assets[collateral_asset].emode_label) < borrow_amt:
            return self.fail_transaction(f"Tried to borrow more than allowed by LTV ({borrow_asset, collateral_asset}")
        self.borrowed[borrow_asset] += borrow_amt
        # if borrow_asset not in agent.holdings:
        #     agent.holdings[borrow_asset] = 0
        # agent.holdings[borrow_asset] += borrow_amt
        # agent.holdings[collateral_asset] -= collateral_amt
        cdp = CDP({borrow_asset: borrow_amt}, {collateral_asset: collateral_amt})
        self.cdps.append(cdp)
        agent.add(borrow_asset, borrow_amt)
        agent.remove(collateral_asset, collateral_amt)
        return cdp

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
        health_factor = self.get_health_factor(cdp)
        if health_factor < self.full_liquidation_threshold:
            # fully liquidatable
            return cdp.debt[debt_asset]
        elif health_factor < 1:
            # partially liquidatable
            return cdp.debt[debt_asset] * self.partial_liquidation_pct
        else:
            return 0

    def calculate_liquidation(
            self,
            cdp: CDP,
            collateral_asset: str,
            debt_asset: str,
            repay_amount: float = -1
    ) -> tuple[float, float]:
        """
        return the amount of debt repaid and the amount of collateral liquidated
        as a tuple: (collateral_liquidated, debt_repaid)
        """
        assert debt_asset in cdp.debt
        if repay_amount < 0:
            repay_amount = self.get_maximum_repayment(cdp, debt_asset)
        else:
            repay_amount = min(repay_amount, self.get_maximum_repayment(cdp, debt_asset))
        if repay_amount == 0:
            return 0, 0
        debt_value = repay_amount * self.price(debt_asset)
        collateral_value = (
            cdp.collateral[collateral_asset] * self.price(collateral_asset)
            / (1 + self.get_liquidation_bonus(collateral_asset, debt_asset, cdp.e_mode))
        )
        debt_repaid = (  # amount of debt repayment that can be covered by this collateral
            repay_amount if debt_value < collateral_value
            else repay_amount * collateral_value / debt_value
        )
        payout = (
            cdp.collateral[collateral_asset] if collateral_value <= debt_value
            else debt_repaid * self.price(debt_asset) / self.price(collateral_asset)
                * (1 + self.get_liquidation_bonus(collateral_asset, debt_asset, cdp.e_mode))
        )
        return payout, debt_repaid

    def liquidate(
            self,
            cdp: CDP,
            agent: Agent,
            debt_asset: str,
            collateral_asset: str,
            repay_amount: float = -1
    ):
        if repay_amount > cdp.debt[debt_asset]:
            raise ValueError("Repay amount exceeds CDP debt.")
        payout, debt_repaid = self.calculate_liquidation(
            cdp=cdp,
            collateral_asset=collateral_asset,
            debt_asset=debt_asset,
            repay_amount=repay_amount
        )
        if payout == 0:
            return self.fail_transaction("CDP cannot be liquidated.")
        if not agent.validate_holdings(debt_asset, debt_repaid):
            return self.fail_transaction(f"Agent doesn't have enough {debt_asset}.")

        agent.add(collateral_asset, payout)
        cdp.collateral[collateral_asset] -= payout

        agent.remove(debt_asset, debt_repaid)
        cdp.debt[debt_asset] -= debt_repaid
        self.borrowed[debt_asset] -= debt_repaid

        if not cdp.fix_liquidation_threshold:
            cdp.liquidation_threshold = self.cdp_liquidation_threshold(cdp)
        return self

    def value_assets(self, assets: dict[str: float]) -> float:
        return sum([assets[tkn] * self.price(tkn) for tkn in assets])

    def update(self):
        pass

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
        if not (len(self.liquidity) == len(self.borrowed) == len(cdp_borrowed) ):
            return False
        return True
