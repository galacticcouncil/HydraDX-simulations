import copy

from sympy import andre

from .agents import Agent
from .oracle import Oracle


class CDP:
    def __init__(
            self,
            debt_assets: dict[str: float],
            collateral_assets: dict[str: float],
            liquidation_threshold: float = None,
            agent=None
    ):
        self.debt = debt_assets
        self.collateral_assets = collateral_assets
        self.asset_list = list(debt_assets.keys() | collateral_assets.keys())
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
            f"{newline.join([f"        {tkn}: {self.collateral_assets[tkn]}" for tkn in self.collateral_assets.keys()])}\n\n"
        )

    def copy(self):
        return CDP(
            self.debt,
            self.collateral_assets,
            self.liquidation_threshold
        )

    def validate(self) -> bool:
        if min(self.debt.values) < 0 or min(self.collateral_assets.values()) < 0:
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
            asset1.emode_ltv if asset1.emode_label == asset2.emode_label else asset1.emode_ltv
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }
        self.liquidation_bonus = {
            (asset1.name, asset2.name):
            asset1.emode_liquidation_bonus if asset1.emode_label == asset2.emode_label else asset1.liquidation_bonus
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }
        self.liquidation_threshold = {
            (asset1.name, asset2.name):
            asset1.emode_liquidation_threshold if asset1.emode_label == asset2.emode_label else asset1.liquidation_threshold
            for asset1 in assets
            for asset2 in assets if asset2 != asset1
        }

        self.cdps: list[CDP] = [] if cdps is None else cdps
        for cdp in self.cdps:
            if not cdp.liquidation_threshold:
                cdp.liquidation_threshold = self.get_liquidation_threshold(cdp)

        for cdp in self.cdps:
            for tkn in cdp.debt:
                self.borrowed[tkn] += cdp.debt[tkn]

        self.partial_liquidation_pct = close_factor
        if not self.validate():
            raise ValueError("money_market initialization failed.")
        self.full_liquidation_threshold = full_liquidation_threshold
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
                (collateral_tkn is None or collateral_tkn in cdp.collateral_assets) and
                (debt_tkn is None or debt_tkn in cdp.debt))]

    def get_health_factor(self, cdp: CDP) -> float:
        prices = {tkn: self.get_oracle_price(tkn) for tkn in cdp.collateral_assets.keys() | cdp.debt.keys()}
        health_factor = (
            sum([
                cdp.collateral_assets[c_tkn] * prices[c_tkn]
                for c_tkn in cdp.collateral_assets
            ]) * self.get_liquidation_threshold(cdp)
            / sum([
            cdp.debt[d_tkn] * prices[d_tkn]
                for d_tkn in cdp.debt
            ])
        )
        return health_factor

    def get_ltv(self, collateral_tkn: str, debt_tkn: str) -> float:
        return self.ltv[(collateral_tkn, debt_tkn)] if (collateral_tkn, debt_tkn) in self.ltv else 0

    def get_liquidation_threshold(self, cdp: CDP) -> float:
        return self.liquidation_threshold[cdp.collateral_assets]

    def get_liquidation_bonus(self, collateral_tkn: str, debt_tkn: str) -> float:
        return self.liquidation_bonus[(collateral_tkn, debt_tkn)] if (collateral_tkn, debt_tkn) in self.liquidation_bonus else 0

    def is_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral_assets.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < 1

    def is_fully_liquidatable(self, cdp: CDP) -> bool:
        if sum(cdp.collateral_assets.values()) == 0:
            return False
        health_factor = self.get_health_factor(cdp)
        return health_factor < self.full_liquidation_threshold

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

    def liquidate(self, cdp: CDP, agent: Agent) -> None:
        if not self.is_liquidatable(cdp):
            return

        prices = {tkn: self.get_oracle_price(tkn) for tkn in cdp.collateral_assets.keys() | cdp.debt.keys()}

        total_debt_value = sum([
            prices[d_tkn] * cdp.debt[d_tkn]
            for d_tkn in cdp.debt
        ])

        if self.is_fully_liquidatable(cdp):
            liquidation_amount = total_debt_value
        else:
            liquidation_amount = self.partial_liquidation_pct * total_debt_value

        liquidation_amount = min(liquidation_amount, total_debt_value)

        # Calculate debt repayment amounts proportionally
        debt_repayments = {}
        remaining_liquidation_amount = liquidation_amount
        for d_tkn, d_amt in cdp.debt.items():
            repayment_pct = min(1.0, remaining_liquidation_amount / total_debt_value)
            repayment_amt = d_amt * repayment_pct

            debt_repayments[d_tkn] = repayment_amt
            remaining_liquidation_amount -= prices[d_tkn] * repayment_amt

        # Prioritize collateral with the highest liquidation bonuses
        collateral_claims = {}
        remaining_collateral_value = liquidation_amount

        for c_tkn, c_amt in cdp.collateral_assets.items():
            if remaining_collateral_value <= 0:
                break

            best_bonus = 0
            best_debt_tkn = None
            for d_tkn in cdp.debt.keys():
                bonus = self.get_liquidation_bonus(c_tkn, d_tkn)
                if bonus > best_bonus:
                    best_bonus = bonus
                    best_debt_tkn = d_tkn

            if best_debt_tkn is None:
                continue

            claimable_value = prices[c_tkn] * c_amt
            claim_value = min(claimable_value, remaining_collateral_value * (1 + best_bonus))
            claim_amt = claim_value / prices[c_tkn]
            remaining_collateral_value -= claim_value / (1 + best_bonus)

            collateral_claims[c_tkn] = claim_amt

        # Execute liquidation
        for d_tkn, d_repay in debt_repayments.items():
            cdp.debt[d_tkn] -= d_repay
            agent.holdings[d_tkn] = agent.holdings.get(d_tkn, 0) - d_repay
            self.liquidity[d_tkn] += d_repay

        for c_tkn, c_claim in collateral_claims.items():
            cdp.collateral_assets[c_tkn] -= c_claim
            agent.holdings[c_tkn] = agent.holdings.get(c_tkn, 0) + c_claim
            self.liquidity[c_tkn] -= c_claim

        # Remove cdp if all debt is repaid
        if sum(cdp.debt.values()) <= 0:
            self.cdps.remove(cdp)

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
