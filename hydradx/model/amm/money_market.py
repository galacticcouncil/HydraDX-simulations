import copy
from .agents import Agent


class CDP:
    def __init__(
            self,
            debt: dict[str: float],
            collateral: dict[str: float],
            liquidation_threshold: float = None,
            health_factor: float = 0,
            agent=None
    ):
        self.debt: dict[str: float] = debt
        self.collateral: dict[str: float] = collateral
        self.asset_list = list(debt.keys() | collateral.keys())
        self.liquidation_threshold = liquidation_threshold
        self.health_factor = health_factor
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
            if not cdp.health_factor:
                cdp.health_factor = self.get_health_factor(cdp)

        for cdp in self.cdps:
            for tkn in cdp.debt:
                self.borrowed[tkn] += cdp.debt[tkn]

        self.partial_liquidation_pct = close_factor
        self.full_liquidation_threshold = full_liquidation_threshold
        if not self.validate():
            raise ValueError("money_market initialization failed.")
        self.fail = ''
        self.assets = assets

    def __repr__(self):
        return (
            f"money_market("
            f"    liquidity: {self.liquidity}\n"
            f"    liquidation threshold: {self.liquidation_threshold}\n"
            f"    liquidation bonus: {self.liquidation_bonus})\n"
            f"    total borrowed: {self.borrowed}\n"
            f"    total collateral: {dict(
                [(tkn, sum([cdp.collateral[tkn] if tkn in cdp.collateral else 0 for cdp in self.cdps])) 
                 for tkn in self.asset_list]
            )}\n"
        )

    def copy(self):
        return copy.deepcopy(self)

    def add_new_asset(self, new_asset: MoneyMarketAsset):
        for existing_asset in self.assets:
            self.liquidation_bonus[(existing_asset.name, new_asset.name)] = (
                existing_asset.emode_liquidation_bonus if existing_asset.emode_label == new_asset.emode_label
                else existing_asset.liquidation_bonus
            )
            self.liquidation_bonus[(new_asset.name, existing_asset.name)] = (
                new_asset.emode_liquidation_bonus if existing_asset.emode_label == new_asset.emode_label
                else new_asset.liquidation_bonus
            )
            self.ltv[(existing_asset.name, new_asset.name)] = (
                existing_asset.emode_ltv if existing_asset.emode_label == new_asset.emode_label
                else existing_asset.ltv
            )
            self.ltv[(new_asset.name, existing_asset.name)] = (
                new_asset.emode_ltv if existing_asset.emode_label == new_asset.emode_label
                else new_asset.ltv
            )
            self.liquidation_threshold[(existing_asset.name, new_asset.name)] = (
                existing_asset.emode_liquidation_threshold if existing_asset.emode_label == new_asset.emode_label
                else existing_asset.liquidation_threshold
            )
            self.liquidation_threshold[(new_asset.name, existing_asset.name)] = (
                new_asset.emode_liquidation_threshold if existing_asset.emode_label == new_asset.emode_label
                else new_asset.liquidation_threshold
            )
        self.liquidity[new_asset.name] = new_asset.liquidity
        self.prices[new_asset.name] = new_asset.price
        self.borrowed[new_asset.name] = 0
        self.asset_list.append(new_asset.name)

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
        if sum(cdp.collateral.values()) == 0 or sum(cdp.debt.values()) == 0:
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

    def cdp_liquidation_threshold(self, cdp: CDP) -> float:
        """
        Calculate the liquidation threshold for a CDP with multiple assets.

        Args:
            cdp: A CDP instance with collateral and debt assets

        Returns:
            float: The calculated liquidation threshold as a decimal (0.0-1.0)
        """
        # Calculate all asset values in market reference currency
        collateral_values = {}
        debt_values = {}

        # Process collateral assets
        for asset_name, balance in cdp.collateral.items():
            if balance <= 0:
                continue

            price = self.prices[asset_name]
            value = balance * price
            collateral_values[asset_name] = {"amount": balance, "value": value}

        # Process debt assets
        for asset_name, debt in cdp.debt.items():
            if debt <= 0:
                continue

            price = self.prices[asset_name]
            value = debt * price
            debt_values[asset_name] = {"amount": debt, "value": value}

        # Check for self-borrowing (same asset used as both collateral and debt)
        self_borrowing_assets = [
            asset for asset in cdp.asset_list
            if asset in cdp.collateral and asset in cdp.debt
               and cdp.collateral.get(asset, 0) > 0 and cdp.debt.get(asset, 0) > 0
        ]

        if self_borrowing_assets:
            # For self-borrowing, use any threshold involving this asset as collateral
            asset = self_borrowing_assets[0]
            for other_asset in self.asset_list:
                if other_asset != asset:
                    threshold = self.get_liquidation_threshold(asset, other_asset)
                    if threshold > 0:
                        return threshold
            return 0.7  # Default if not found

        # Match each debt asset with the best collateral asset (highest liquidation threshold)
        matches = []
        remaining_collateral = {k: {"amount": v["amount"], "value": v["value"]} for k, v in collateral_values.items()}

        # Process each debt asset
        for debt_asset, debt_info in debt_values.items():
            debt_value = debt_info["value"]

            # Find the best collateral assets for this debt based on liquidation threshold
            collateral_options = []

            for collateral_asset in remaining_collateral:
                if remaining_collateral[collateral_asset]["value"] > 0:
                    # Get the threshold between these specific assets
                    threshold = self.get_liquidation_threshold(collateral_asset, debt_asset)

                    collateral_options.append({
                        "asset": collateral_asset,
                        "threshold": threshold
                    })

            # Sort collateral options by threshold (highest first)
            collateral_options.sort(key=lambda x: x["threshold"], reverse=True)

            # Match debt with best collateral options
            debt_remaining = debt_value

            for option in collateral_options:
                collateral_asset = option["asset"]
                threshold = option["threshold"]

                available_collateral = remaining_collateral[collateral_asset]["value"]
                match_amount = min(available_collateral, debt_remaining)

                if match_amount > 0:
                    matches.append({
                        "value": match_amount,
                        "threshold": threshold
                    })

                    remaining_collateral[collateral_asset]["value"] -= match_amount
                    debt_remaining -= match_amount

                    if debt_remaining <= 0:
                        break

        # Calculate weighted average of thresholds
        total_matched_value = sum(match["value"] for match in matches)

        if total_matched_value == 0:
            return 0  # Edge case: no collateral or no debt

        weighted_sum = sum(match["value"] * match["threshold"] for match in matches)
        weighted_threshold = weighted_sum / total_matched_value

        return weighted_threshold

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
            return min(
                cdp.debt[debt_asset],
                cdp.debt[debt_asset] * self.partial_liquidation_pct
            )

    def calculate_liquidation(self, cdp: CDP, debt_asset: str, repay_amount: float = -1) -> dict[str: float]:
        assert debt_asset in cdp.debt
        if repay_amount < 0:
            repay_amount = self.get_maximum_repayment(cdp, debt_asset)
        debt_value = repay_amount * self.get_oracle_price(debt_asset)
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
                repay_amount if debt_value < collateral_value
                else repay_amount * collateral_value / debt_value
            )
            returns[collateral_asset] = (
                cdp.collateral[collateral_asset] if repay_amount < repay_amount
                else min(
                    repay_amount * self.get_oracle_price(debt_asset) / self.get_oracle_price(collateral_asset)
                    * (1 + self.get_liquidation_bonus(collateral_asset, debt_asset)),
                    cdp.collateral[collateral_asset]
                )
            )
        return returns

    def liquidate(self, cdp: CDP, debt_asset: str, agent: Agent, repay_amount: float = -1):
        if not self.is_liquidatable(cdp):
            return
        if repay_amount < 0:
            repay_amount = self.get_maximum_repayment(cdp, debt_asset)

        collateral = self.calculate_liquidation(cdp, debt_asset, repay_amount)
        if not agent.validate_holdings(debt_asset, repay_amount):
            return self.fail_transaction(f"Agent doesn't have enough {debt_asset}.")

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
