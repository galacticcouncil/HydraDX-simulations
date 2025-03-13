from .amm.agents import Agent
from .amm.stableswap_amm import StableSwapPoolState


class StabilityModule:
    def __init__(
            self,
            liquidity: dict[str: float],  # initial liquidity of stability module
            buyback_speed: list[float] or float,  # paramater controlling how quickly Hollar is bought back
            pools: list[StableSwapPoolState],  # pools that can be used to mint Hollar
            sell_price: list[float] or float = 1,  # price at which stability module sells Hollar
            max_buy_price: list[float] or float = 1,  # maximum price at which stability module buys Hollar
            buy_fee: list[float] or float = 0.0001,  # fee paid to arbitrage for assisting buying back of Hollar
            native_stable: str = 'HOLLAR'  # native stablecoin name
    ):
        assert native_stable not in liquidity  # Hollar cannot back itself
        self.liquidity = {k: v for k, v in liquidity.items()}
        self.asset_list = list(liquidity.keys())
        self.time_step = 0
        self.native_stable_bought = 0

        if isinstance(buyback_speed, list):
            if len(buyback_speed) != len(self.asset_list):
                raise ValueError("buyback_speed must have same length as asset_list")
            self.buyback_speed = {tkn: speed for tkn, speed in zip(self.asset_list, buyback_speed)}
        else:
            self.buyback_speed = {tkn: buyback_speed for tkn in self.asset_list}
        if min(self.buyback_speed.values()) < 0:  # buyback_speed = 0 turns buybacks off for an asset
            raise ValueError("buyback_speed must be non-negative")
        if max(self.buyback_speed.values()) > 1:
            raise ValueError("buyback_speed must be less than or equal to 1")

        if len(pools) != len(self.asset_list):
            raise ValueError("pools must have same length as asset_list")
        if len({id(pool) for pool in pools}) != len(pools):
            raise ValueError("pools must be unique")
        self.pools = {}  # references to designated stableswap pools
        self._pool_states = {}  # copy of pool states used for calculations
        for i, tkn in enumerate(self.asset_list):
            if tkn not in pools[i].asset_list or native_stable not in pools[i].asset_list:
                raise ValueError("pool does not have required assets")
            self.pools[tkn] = pools[i]
            self._pool_states[tkn] = pools[i].copy()

        if isinstance(sell_price, list):
            self.sell_price = {tkn: price for tkn, price in zip(self.asset_list, sell_price)}
        else:
            self.sell_price = {tkn: sell_price for tkn in self.asset_list}
        if min(self.sell_price.values()) < 1:
            raise ValueError("sell_price must be greater than or equal to 1")

        if isinstance(max_buy_price, list):
            self.max_buy_price = {tkn: price for tkn, price in zip(self.asset_list, max_buy_price)}
        else:
            self.max_buy_price = {tkn: max_buy_price for tkn in self.asset_list}
        if max(self.max_buy_price.values()) > 1:
            raise ValueError("max_buy_price must be less than or equal to 1")
        if min(self.max_buy_price.values()) <= 0:
            raise ValueError("max_buy_price must be greater than 0")

        if isinstance(buy_fee, list):
            self.buy_fee = {tkn: fee for tkn, fee in zip(self.asset_list, buy_fee)}
        else:
            self.buy_fee = {tkn: buy_fee for tkn in self.asset_list}
        if min(self.buy_fee.values()) < 0:
            raise ValueError("buy_fee must be non-negative")
        if max(self.buy_fee.values()) > 1:
            raise ValueError("buy_fee must be less than or equal to 1")

        self.native_stable = native_stable
        self.fail = ''

    def fail_transaction(self, error: str):
        self.fail = error
        return self

    def update(self):
        self.time_step += 1
        for tkn, pool in self.pools.items():
            self._pool_states[tkn] = pool.copy()
        self.native_stable_bought = 0

    def get_buy_params(self, tkn: str) -> tuple:
        pool = self._pool_states[tkn]
        imbalance = (pool.liquidity[self.native_stable] - pool.liquidity[tkn]) / 2
        max_buy_amt = max([self.buyback_speed[tkn] * imbalance, 0])
        sell_amt = pool.calculate_sell_from_buy(tkn_buy=self.native_stable, tkn_sell=tkn, buy_quantity=max_buy_amt)
        exec_price = sell_amt / max_buy_amt if max_buy_amt > 0 else 0
        buy_price = exec_price / (1 - self.buy_fee[tkn])
        if buy_price > self.max_buy_price[tkn] or buy_price == 0:
            return 0, 0
        max_buy_amt = min(max_buy_amt, self.liquidity[tkn] / buy_price)
        return max(max_buy_amt - self.native_stable_bought, 0), buy_price

    def swap(
            self,
            agent: Agent,
            tkn_buy: str = None,
            tkn_sell: str = None,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        if buy_quantity < 0:
            return self.fail_transaction("Negative buy quantity")
        if sell_quantity < 0:
            return self.fail_transaction("Negative sell quantity")
        if tkn_buy is None and tkn_sell is None:
            return self.fail_transaction("No tokens specified")

        if tkn_buy == self.native_stable:  # trader buying HOLLAR
            if tkn_sell not in self.asset_list:
                return self.fail_transaction("Token not supported by stability module")
            if buy_quantity == 0:
                buy_quantity = sell_quantity / self.sell_price[tkn_sell]
            elif sell_quantity == 0:
                sell_quantity = buy_quantity * self.sell_price[tkn_sell]
        else:
            if tkn_sell != self.native_stable:
                return self.fail_transaction("Swap must involve native stablecoin")
            if tkn_buy not in self.asset_list:
                return self.fail_transaction("Token not supported by stability module")
            max_buy_amt, buy_price = self.get_buy_params(tkn_buy)
            if max_buy_amt == 0:
                return self.fail_transaction("stability module cannot buy Hollar")
            elif buy_price == 0:  # should never be true
                raise ValueError("Buy price shouldn't be 0 when max_buy_amt is nonzero")
            if buy_quantity == 0:
                buy_quantity = sell_quantity * buy_price
            elif sell_quantity == 0:
                sell_quantity = buy_quantity / buy_price
            if sell_quantity > max_buy_amt:
                return self.fail_transaction("Max buy amount exceeded")
            if buy_quantity > self.liquidity[tkn_buy]:
                return self.fail_transaction("Insufficient liquidity in stability module")

        if not agent.validate_holdings(tkn_sell, sell_quantity):
            return self.fail_transaction("Agent does not have enough tokens to sell")
        # checks passed, proceed with swap
        if tkn_sell == self.native_stable:  # trader selling HOLLAR to HSM
            self.native_stable_bought += sell_quantity
        agent.remove(tkn_sell, sell_quantity)
        agent.add(tkn_buy, buy_quantity)
        if tkn_buy != self.native_stable:
            self.liquidity[tkn_buy] -= buy_quantity
        else:
            self.liquidity[tkn_sell] += sell_quantity

    def arb(self, agent: Agent, tkn: str) -> None:
        if tkn not in self.asset_list:
            raise ValueError("Token not supported by stability module")
        max_buy_amt, buy_price = self.get_buy_params(tkn)
        if max_buy_amt == 0:
            return
        agent.add(self.native_stable, max_buy_amt)  # flash mint Hollar for arb
        self.swap(agent, tkn_buy=tkn, tkn_sell=self.native_stable, sell_quantity=max_buy_amt)
        self.pools[tkn].swap(agent, tkn_buy=self.native_stable, tkn_sell=tkn, buy_quantity=max_buy_amt)
        agent.remove(self.native_stable, max_buy_amt)  # burn Hollar that was minted
