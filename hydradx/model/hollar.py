from .amm.agents import Agent
from .amm.stableswap_amm import StableSwapPoolState


class StabilityModule:
    def __init__(
            self,
            liquidity: dict[str: float],  # initial liquidity of stability module
            buyback_speed: list[float] or float,  # paramater controlling how quickly Hollar is bought back
            pools: list[StableSwapPoolState],  # pools that can be used to mint Hollar
            sell_price_fee: list[float] or float = 0.0001,  # fee paid by traders buying Hollar
            max_buy_price_coef: list[float] or float = 1,  # maximum price at which stability module buys Hollar
            buy_fee: list[float] or float = 0.0001,  # fee paid to arbitrage for assisting buying back of Hollar
            native_stable: str = 'HOLLAR',  # native stablecoin name,
            max_liquidity: dict[str: float] = None  # maximum liquidity of stability module
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

        if isinstance(sell_price_fee, list):
            self.sell_price_fee = {tkn: fee for tkn, fee in zip(self.asset_list, sell_price_fee)}
        else:
            self.sell_price_fee = {tkn: sell_price_fee for tkn in self.asset_list}
        if min(self.sell_price_fee.values()) < 0:
            raise ValueError("sell_price_fee must be non-negative")

        if isinstance(max_buy_price_coef, list):
            self.max_buy_price_coef = {tkn: price for tkn, price in zip(self.asset_list, max_buy_price_coef)}
        else:
            self.max_buy_price_coef = {tkn: max_buy_price_coef for tkn in self.asset_list}
        if max(self.max_buy_price_coef.values()) > 1:
            raise ValueError("max_buy_price must be less than or equal to 1")
        if min(self.max_buy_price_coef.values()) <= 0:
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

        self.max_liquidity = {}
        for tkn in self.liquidity:
            if max_liquidity is not None and tkn in max_liquidity:
                if self.liquidity[tkn] > max_liquidity[tkn]:
                    raise ValueError(f"Initial liquidity of {tkn} exceeds maximum liquidity")
                self.max_liquidity[tkn] = max_liquidity[tkn]
            else:
                self.max_liquidity[tkn] = float('inf')

    def fail_transaction(self, error: str):
        self.fail = error
        return self

    def update(self):
        self.time_step += 1
        for tkn, pool in self.pools.items():
            self._pool_states[tkn] = pool.copy()
        self.native_stable_bought = 0

    def get_peg(self, tkn: str) -> float:
        pool = self._pool_states[tkn]
        i_tkn = pool.asset_list.index(tkn)
        i_native_stable = pool.asset_list.index(self.native_stable)
        return pool.peg[i_tkn] / pool.peg[i_native_stable]

    def _get_max_buy_amount(self, tkn: str) -> float:  # note this ignores self.max_buy_price_coef
        imbalance = (self._pool_states[tkn].liquidity[self.native_stable]
                     - self.get_peg(tkn) * self._pool_states[tkn].liquidity[tkn]) / 2
        return max([self.buyback_speed[tkn] * imbalance, 0])

    def get_buy_params(self, tkn: str) -> tuple:
        pool = self._pool_states[tkn]
        peg = self.get_peg(tkn)
        # imbalance = (pool.liquidity[self.native_stable] - peg * pool.liquidity[tkn]) / 2
        # max_buy_amt = max([self.buyback_speed[tkn] * imbalance, 0])
        max_buy_amt = self._get_max_buy_amount(tkn)
        sell_amt = pool.calculate_sell_from_buy(tkn_buy=self.native_stable, tkn_sell=tkn, buy_quantity=max_buy_amt)
        exec_price = sell_amt / max_buy_amt if max_buy_amt > 0 else 0
        buy_price = exec_price / (1 - self.buy_fee[tkn])

        if buy_price > self.max_buy_price_coef[tkn] / peg or buy_price == 0:
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
            peg = self.get_peg(tkn_sell)
            sell_price = (1 + self.sell_price_fee[tkn_sell]) / peg
            if buy_quantity == 0:
                buy_quantity = sell_quantity / sell_price
            elif sell_quantity == 0:
                sell_quantity = buy_quantity * sell_price
            if sell_quantity + self.liquidity[tkn_sell] > self.max_liquidity[tkn_sell]:
                return self.fail_transaction("HSM max liquidity exceeded")
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


def fast_hollar_arb_and_dump(
        hsm: StabilityModule,
        agent: Agent,
        sell_amt: float,  # sell amount for this particular block
        tkn_buy: str,
        record: list = None  # allows user to request intermediate data to be recorded and returned
) -> dict:
    """
    This function simulates an agent arbitraging the HSM against its stableswap pool, and then dumping Hollar
    into the indicated pool. This simulation can be done more efficiently using this function than by calling
    the arb and swap functions of the StabilityModule and StableSwapPoolState classes separately.
    """
    if record is None:
        record = []
    data = {k: None for k in record}
    ss = hsm.pools[tkn_buy]
    max_buy_amt = hsm._get_max_buy_amount(tkn_buy)  # note this ignores self.max_buy_price_coef
    if 'max_buy_amt' in data:
        data['max_buy_amt'] = max_buy_amt
    hollar_buy_amt = max_buy_amt - sell_amt

    agent.add(hsm.native_stable, max_buy_amt)  # flash mint Hollar for arb
    hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=hsm.native_stable, sell_quantity=max_buy_amt)
    if hollar_buy_amt > 0:
        ss.swap(agent, tkn_buy=hsm.native_stable, tkn_sell=tkn_buy, buy_quantity=hollar_buy_amt)
    elif hollar_buy_amt < 0:
        ss.swap(agent, tkn_buy=tkn_buy, tkn_sell=hsm.native_stable, sell_quantity=-hollar_buy_amt)
    agent.remove(hsm.native_stable, max_buy_amt)  # burn Hollar that was minted
    return data
