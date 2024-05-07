from math import sqrt as sqrt
import math
from .agents import Agent
from .amm import AMM
# from mpmath import mp, mpf
# mp.dps = 50

tick_increment = 1 + 1e-4
def tick_to_price(tick: int):
    return tick_increment ** tick

def price_to_tick(price: float, tick_spacing: int = 1):
    raw_tick = math.log(price) / math.log(tick_increment)
    nearest_valid_tick = round(raw_tick / tick_spacing) * tick_spacing
    return nearest_valid_tick

import math

class ConcentratedLiquidityState(AMM):
    def __init__(
            self,
            assets: dict,
            min_tick: int = None,
            max_tick: int = None,
            tick_spacing: int = 10,
            fee: float = 0.0
    ):
        super().__init__()
        self.asset_list = list(assets.keys())
        if len(self.asset_list) != 2:
            raise ValueError("Expected 2 assets.")
        self.tick_spacing = tick_spacing
        self.fee = fee
        self.liquidity = {tkn: assets[tkn] for tkn in self.asset_list}
        self.asset_x = self.asset_list[0]
        self.asset_y = self.asset_list[1]
        x = self.liquidity[self.asset_x]
        y = self.liquidity[self.asset_y]
        price = y / x
        price_tick = price_to_tick(price, tick_spacing)

        if min_tick is not None and max_tick is not None:
            raise ValueError("Only one of min_tick or max_tick should be provided.")
        if min_tick is not None:
            max_tick = 2 * price_tick - min_tick
        elif max_tick is not None:
            min_tick = 2 * price_tick - max_tick
        else:
            raise ValueError("Either min_tick or max_tick must be provided.")
        self.min_price = tick_to_price(min_tick)
        self.max_price = tick_to_price(max_tick)
        self.min_tick = min_tick
        self.max_tick = max_tick
        k = (x * sqrt(y / x) * sqrt(self.max_price) / (sqrt(self.max_price) - sqrt(y / x))) ** 2
        a = sqrt(k * x / y) - x
        b = sqrt(k * y / x) - y
        self.x_offset = a
        self.y_offset = b
        self.invariant = k

        if not min_tick <= price_tick <= max_tick:
            raise ValueError("Initial price is outside the tick range.")
        if min_tick % self.tick_spacing != 0 or max_tick % self.tick_spacing != 0:
            raise ValueError(f"Tick values must be multiples of the tick spacing ({self.tick_spacing}).")
        self.fees_accrued = {tkn: 0 for tkn in self.asset_list}

    def swap(self, agent: Agent, tkn_buy: str, tkn_sell: str, buy_quantity: float = 0, sell_quantity: float = 0):
        if buy_quantity > 0 and sell_quantity > 0:
            raise ValueError("Only one of buy_quantity or sell_quantity should be provided.")

        if buy_quantity == 0 and sell_quantity == 0:
            raise ValueError("Either buy_quantity or sell_quantity must be provided.")

        if tkn_buy not in self.asset_list or tkn_sell not in self.asset_list:
            raise ValueError(f"Invalid token symbols. Token symbols must be {' or '.join(self.asset_list)}.")

        if tkn_buy == tkn_sell:
            raise ValueError("Cannot buy and sell the same token.")

        if buy_quantity > 0:
            sell_quantity = self.calculate_sell_from_buy(tkn_sell, tkn_buy, buy_quantity)
        elif sell_quantity > 0:
            buy_quantity = self.calculate_buy_from_sell(tkn_buy, tkn_sell, sell_quantity)

        if agent.holdings[tkn_sell] < sell_quantity:
            return self.fail_transaction(f"Agent doesn't have enough {tkn_sell}", agent)

        self.liquidity[tkn_sell] += sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity

        if tkn_buy not in agent.holdings:
            agent.holdings[tkn_buy] = 0
        agent.holdings[tkn_sell] -= sell_quantity
        agent.holdings[tkn_buy] += buy_quantity

        return self

    def calculate_buy_from_sell(self, tkn_buy: str, tkn_sell: str, sell_quantity: float) -> float:
        x_virtual, y_virtual = self.get_virtual_reserves()
        sell_quantity *= (1 - self.fee)
        if tkn_sell == self.asset_x:
            buy_quantity = sell_quantity * y_virtual / (x_virtual + sell_quantity)
        else:
            buy_quantity = sell_quantity * x_virtual / (y_virtual + sell_quantity)

        return buy_quantity

    def calculate_sell_from_buy(self, tkn_sell: str, tkn_buy: str, buy_quantity: float) -> float:
        x_virtual, y_virtual = self.get_virtual_reserves()

        if tkn_buy == self.asset_x:
            sell_quantity = buy_quantity * y_virtual / (x_virtual - buy_quantity)
        else:
            sell_quantity = buy_quantity * x_virtual / (y_virtual - buy_quantity)

        return sell_quantity / (1 - self.fee)

    def get_virtual_reserves(self):
        x_virtual = self.liquidity[self.asset_x] + self.x_offset
        y_virtual = self.liquidity[self.asset_y] + self.y_offset
        return x_virtual, y_virtual

    def price(self, tkn: str, denomination: str = '') -> float:
        if tkn not in self.asset_list:
            raise ValueError(f"Invalid token symbol. Token symbol must be {' or '.join(self.asset_list)}.")
        if denomination and denomination not in self.asset_list:
            raise ValueError(f"Invalid denomination symbol. Denomination symbol must be {' or '.join(self.asset_list)}.")
        if tkn == denomination:
            return 1

        x_virtual, y_virtual = self.get_virtual_reserves()

        if tkn == self.asset_x:
            return y_virtual / x_virtual
        else:
            return x_virtual / y_virtual

    def copy(self):
        return ConcentratedLiquidityState(
            assets=self.liquidity.copy(),
            min_tick=self.min_tick,
            tick_spacing=self.tick_spacing,
            fee=self.fee
        )

    def __str__(self):
        return f"""
        assets: {', '.join(self.asset_list)}
        min_tick: {self.min_tick} ({tick_to_price(self.min_tick)}), 
        max_tick: {self.max_tick} ({tick_to_price(self.max_tick)})
        """
