from math import sqrt as sqrt
import math

import mpmath

from .agents import Agent
from .amm import AMM
from mpmath import mp, mpf
mp.dps = 50

TICK_INCREMENT = 1 + 1e-4
MIN_TICK = -887272
MAX_TICK = -MIN_TICK
def tick_to_price(tick: int or float):
    return TICK_INCREMENT ** tick

def price_to_tick(price: float, tick_spacing: int = 0):
    raw_tick = math.log(price) / math.log(TICK_INCREMENT)
    if tick_spacing == 0:
        return raw_tick
    nearest_valid_tick = int(raw_tick / tick_spacing) * tick_spacing
    return nearest_valid_tick

# def get_sqrt_ratio_at_tick(tick: int):
#     absTick = abs(tick)
#     ratio = 0xfffcb933bd6fad37aa2d162d1a594001 if absTick & 0x1 != 0 else 0x100000000000000000000000000000000
#     if absTick & 0x2 != 0:
#         ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128
#     if absTick & 0x4 != 0:
#         ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128
#     if absTick & 0x8 != 0:
#         ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128
#     if absTick & 0x10 != 0:
#         ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128
#     if absTick & 0x20 != 0:
#         ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128
#     if absTick & 0x40 != 0:
#         ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128
#     if absTick & 0x80 != 0:
#         ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128
#     if absTick & 0x100 != 0:
#         ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128
#     if absTick & 0x200 != 0:
#         ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128
#     if absTick & 0x400 != 0:
#         ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128
#     if absTick & 0x800 != 0:
#         ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128
#     if absTick & 0x1000 != 0:
#         ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128
#     if absTick & 0x2000 != 0:
#         ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128
#     if absTick & 0x4000 != 0:
#         ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128
#     if absTick & 0x8000 != 0:
#         ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128
#     if absTick & 0x10000 != 0:
#         ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128
#     if absTick & 0x20000 != 0:
#         ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128
#     if absTick & 0x40000 != 0:
#         ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128
#     if absTick & 0x80000 != 0:
#         ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128
#
#     if tick > 0:
#         ratio = 2**256 // ratio
#
#     sqrt_price = (ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1)
#     return sqrt_price


def get_sqrt_ratio_at_tick(tick: int) -> int:
    abs_tick = abs(tick)
    if abs_tick & 1:
        ratio = 0xfffcb933bd6fad37aa2d162d1a594001
    else:
        ratio = 0x100000000000000000000000000000000
    # ratio = 0xfffcb933bd6fad37aa2d162d1a594001 if abs_tick & 0x1 else 0x100000000000000000000000000000000
    # ratio2 = mpf(1.0001) ** 0.5 if abs_tick & 0x1 else 1.0

    if abs_tick & 2**1:
        ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128
    if abs_tick & 2**2:
        ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128
    if abs_tick & 2**3:
        ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128
    if abs_tick & 2**4:
        ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128
    if abs_tick & 2**5:
        ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128
    if abs_tick & 2**6:
        ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128
    if abs_tick & 2**7:
        ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128
    if abs_tick & 2**8:
        ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128
    if abs_tick & 2**9:
        ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128
    if abs_tick & 2**10:
        ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128
    if abs_tick & 2**11:
        ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128
    if abs_tick & 2**12:
        ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128
    if abs_tick & 2**13:
        ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128
    if abs_tick & 2**14:
        ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128
    if abs_tick & 2**15:
        ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128
    if abs_tick & 2**16:
        ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128
    if abs_tick & 2**17:
        ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128
    if abs_tick & 2**18:
        ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128
    if abs_tick & 2**19:
        ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128

    if tick > 0:
        ratio = (1 << 256) // ratio

    sqrt_price = (ratio >> 32) + (1 if ratio % (1 << 32) != 0 else 0)
    return sqrt_price



# function getSqrtRatioAtTick(int24 tick) internal pure returns (uint160 sqrtPriceX96) {
#     uint256 absTick = tick < 0 ? uint256(-int256(tick)) : uint256(int256(tick));
#     require(absTick <= uint256(MAX_TICK), 'T');
#
#     uint256 ratio = absTick & 0x1 != 0 ? 0xfffcb933bd6fad37aa2d162d1a594001 : 0x100000000000000000000000000000000;
#     if (absTick & 0x2 != 0) ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128;
#     if (absTick & 0x4 != 0) ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128;
#     if (absTick & 0x8 != 0) ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128;
#     if (absTick & 0x10 != 0) ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128;
#     if (absTick & 0x20 != 0) ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128;
#     if (absTick & 0x40 != 0) ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128;
#     if (absTick & 0x80 != 0) ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128;
#     if (absTick & 0x100 != 0) ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128;
#     if (absTick & 0x200 != 0) ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128;
#     if (absTick & 0x400 != 0) ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128;
#     if (absTick & 0x800 != 0) ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128;
#     if (absTick & 0x1000 != 0) ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128;
#     if (absTick & 0x2000 != 0) ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128;
#     if (absTick & 0x4000 != 0) ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128;
#     if (absTick & 0x8000 != 0) ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128;
#     if (absTick & 0x10000 != 0) ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128;
#     if (absTick & 0x20000 != 0) ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128;
#     if (absTick & 0x40000 != 0) ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128;
#     if (absTick & 0x80000 != 0) ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128;
#
#     if (tick > 0) ratio = type(uint256).max / ratio;
#
#     // this divides by 1<<32 rounding up to go from a Q128.128 to a Q128.96.
#     // we then downcast because we know the result always fits within 160 bits due to our tick input constraint
#     // we round up in the division so getTickAtSqrtRatio of the output price is always consistent
#     sqrtPriceX96 = uint160((ratio >> 32) + (ratio % (1 << 32) == 0 ? 0 : 1));
# }


# def get_tick_at_sqrt_ratio(sqrt_price_x96):
#     MIN_SQRT_RATIO = 4295128739
#     MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342
#
#     if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 >= MAX_SQRT_RATIO:
#         raise ValueError("R")
#
#     ratio = sqrt_price_x96 << 32
#     r = ratio
#     msb = 0
#
#     if r > 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:
#         msb += 128
#         r >>= 128
#     if r > 0xFFFFFFFFFFFFFFFF:
#         msb += 64
#         r >>= 64
#     if r > 0xFFFFFFFF:
#         msb += 32
#         r >>= 32
#     if r > 0xFFFF:
#         msb += 16
#         r >>= 16
#     if r > 0xFF:
#         msb += 8
#         r >>= 8
#     if r > 0xF:
#         msb += 4
#         r >>= 4
#     if r > 0x3:
#         msb += 2
#         r >>= 2
#     if r > 0x1:
#         msb += 1
#
#     if msb >= 128:
#         r = ratio >> (msb - 127)
#     else:
#         r = ratio << (127 - msb)
#
#     log_2 = (msb - 128) << 64
#
#     for _ in range(50, 63):
#         r = (r * r) >> 127
#         f = r >> 128
#         log_2 |= f << (62 - _)
#         r >>= f
#
#     log_sqrt10001 = log_2 * 255738958999603826347141  # 128.128 number
#
#     tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
#     tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128
#
#     return tick_low if tick_low == tick_high else tick_high if get_sqrt_ratio_at_tick(tick_high) <= sqrt_price_x96 else tick_low


def get_tick_at_sqrt_ratio2(sqrt_price_x96: int) -> int:
    tick_float = 2 * math.log(sqrt_price_x96 / 2**96) / math.log(1.0001)
    tick_int = int(tick_float)
    while get_sqrt_ratio_at_tick(tick_int) <= sqrt_price_x96:
        tick_int += 1
    while get_sqrt_ratio_at_tick(tick_int) > sqrt_price_x96:
        tick_int -= 1
    return tick_int


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    MIN_SQRT_RATIO = 4295128739
    MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 >= MAX_SQRT_RATIO:
        raise ValueError("R")

    ratio = sqrt_price_x96 << 32
    r = ratio
    msb = 0

    if r > 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:
        msb += 128
        r >>= 128
    if r > 0xFFFFFFFFFFFFFFFF:
        msb += 64
        r >>= 64
    if r > 0xFFFFFFFF:
        msb += 32
        r >>= 32
    if r > 0xFFFF:
        msb += 16
        r >>= 16
    if r > 0xFF:
        msb += 8
        r >>= 8
    if r > 0xF:
        msb += 4
        r >>= 4
    if r > 0x3:
        msb += 2
        r >>= 2
    if r > 0x1:
        msb += 1

    if msb >= 128:
        r = ratio >> (msb - 127)
    else:
        r = ratio << (127 - msb)

    log_2 = (msb - 128) << 64

    for i in range(62, 50, -1):
        r = (r * r) >> 127
        f = r >> 128
        log_2 |= f << i
        r >>= f

    log_sqrt10001 = log_2 * 255738958999603826347141

    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    return tick_low if tick_low == tick_high else tick_high if get_sqrt_ratio_at_tick(tick_high) <= sqrt_price_x96 else tick_low




class ConcentratedLiquidityPosition(AMM):
    def __init__(
            self,
            assets: dict,
            min_tick: int = None,
            max_tick: int = None,
            tick_spacing: int = 10,
            fee: float = 0.0,
            protocol_fee: float = 0.0
    ):
        super().__init__()
        self.asset_list = list(assets.keys())
        if len(self.asset_list) != 2:
            raise ValueError("Expected 2 assets.")
        self.tick_spacing = tick_spacing
        self.fee = fee
        self.protocol_fee = protocol_fee
        self.liquidity = {tkn: assets[tkn] for tkn in self.asset_list}
        self.asset_x = self.asset_list[0]
        self.asset_y = self.asset_list[1]
        x = self.liquidity[self.asset_x]
        y = self.liquidity[self.asset_y]
        price = y / x
        price_tick = price_to_tick(price)

        if min_tick is not None and max_tick is not None:
            # raise ValueError("Only one of min_tick or max_tick should be provided.")
            pass
        elif min_tick is not None and max_tick is None:
            max_tick = round(2 * price_tick - min_tick)
        elif max_tick is not None and min_tick is None:
            min_tick = round(2 * price_tick - max_tick)
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
            # vvv what happens to the fee is TBD ^^^
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

        return sell_quantity * (1 + self.fee)

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

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        if fee is None:
            fee = self.fee
        return self.price(tkn_buy) * (1 + fee)

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        if fee is None:
            fee = self.fee
        return self.price(tkn_sell) * (1 - fee)

    def copy(self):
        return ConcentratedLiquidityPosition(
            assets=self.liquidity.copy(),
            min_tick=self.min_tick,
            tick_spacing=self.tick_spacing,
            fee=self.fee
        )

    @property
    def invariant(self):
        return sqrt((self.liquidity[self.asset_x] + self.x_offset) * (self.liquidity[self.asset_y] + self.y_offset))

    def __str__(self):
        return f"""
        assets: {', '.join(self.asset_list)}
        min_tick: {self.min_tick} ({tick_to_price(self.min_tick)}), 
        max_tick: {self.max_tick} ({tick_to_price(self.max_tick)})
        """


class Tick:
    def __init__(self, liquidity_net: int, sqrt_price_x96: int, index: int):
        self.liquidityNet = liquidity_net
        self.sqrt_price_x96 = sqrt_price_x96
        self.index = index
        self.initialized = True


class ConcentratedLiquidityPoolState(AMM):
    ticks: dict[int, Tick]
    def __init__(
            self,
            asset_list: list[str],
            sqrt_price_x96: int,
            liquidity: int,
            tick_spacing: int = 10,
            fee: float = 0.0,
            protocol_fee: float = 0.0,
            ticks: dict[int, Tick] = None
    ):
        super().__init__()
        self.asset_list = asset_list
        self.tick_spacing = tick_spacing
        self.fee = fee
        self.protocol_fee = protocol_fee
        self.fees_accrued = {tkn: 0 for tkn in self.asset_list}
        self.ticks: dict[int, Tick] = ticks or {}
        self.sqrt_price_x96 = sqrt_price_x96
        self.liquidity = liquidity

    def initialize_tick(self, tick: int, liquidity_net: int):
        if tick % self.tick_spacing != 0:
            raise ValueError(f"Tick values must be multiples of the tick spacing ({self.tick_spacing}).")
        sqrt_price_x96 = get_sqrt_ratio_at_tick(tick)
        new_tick = Tick(
            liquidity_net=liquidity_net,
            sqrt_price_x96=sqrt_price_x96,
            index=tick
        )
        self.ticks[tick] = new_tick
        return self

    def initialize_ticks(self, ticks: dict[int, float]):
        for tick, liquidity_net in ticks.items():
            self.initialize_tick(tick, liquidity_net)
        return self

    @property
    def current_tick(self):
        raw_tick = get_tick_at_sqrt_ratio2(self.sqrt_price_x96)
        if self.tick_spacing == 0:
            return raw_tick
        nearest_valid_tick = int(raw_tick / self.tick_spacing) * self.tick_spacing
        return nearest_valid_tick

    def next_initialized_tick(self, zero_for_one):
        search_direction = -1 if zero_for_one else 1
        current_tick = self.current_tick + search_direction * self.tick_spacing

        while not current_tick in self.ticks and MAX_TICK > current_tick > MIN_TICK:
            current_tick += search_direction * self.tick_spacing

        return self.ticks[current_tick] if current_tick in self.ticks else None

    def getAmount0Delta(self, sqrt_ratio_a: int, sqrt_ratio_b: int) -> int:
        if sqrt_ratio_a > sqrt_ratio_b:
            sqrt_ratio_a, sqrt_ratio_b = sqrt_ratio_b, sqrt_ratio_a
        return (self.liquidity << 96) * (sqrt_ratio_b - sqrt_ratio_a) // sqrt_ratio_b // sqrt_ratio_a

    def getAmount1Delta(self, sqrt_ratio_a: int, sqrt_ratio_b: int) -> int:
        if sqrt_ratio_a > sqrt_ratio_b:
            sqrt_ratio_a, sqrt_ratio_b = sqrt_ratio_b, sqrt_ratio_a
        return (self.liquidity * (sqrt_ratio_b - sqrt_ratio_a)) >> 96

    def swap(
            self,
            agent: Agent,
            tkn_buy: str,
            tkn_sell: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0,
            price_limit: float = None
    ):
        exact_input = sell_quantity > 0
        amountSpecifiedRemaining = sell_quantity or -buy_quantity
        amountCalculated = 0
        protocolFee_current = 0

        zeroForOne = tkn_sell == self.asset_list[0]  # zeroForOne means selling x, buying y
        if price_limit is None:
            sqrt_price_limit_x96 = -float('inf') if zeroForOne else float('inf')
        else:
            sqrt_price_limit_x96 = sqrt(price_limit) * 2**96

        while abs(amountSpecifiedRemaining) > 1e-12:
            next_tick: Tick = self.next_initialized_tick(zeroForOne)

            # get the price for the next tick
            sqrt_price_next_x96 = next_tick.sqrt_price_x96 if next_tick else (
                sqrt(tick_to_price(MIN_TICK)) if zeroForOne else sqrt(tick_to_price(MAX_TICK))
            )

            # compute values to swap to the target tick, price limit, or point where input/output amount is exhausted
            self.sqrt_price_x96, amountIn, amountOut, feeAmount = self.compute_swap_step(
                self.sqrt_price_x96,
                sqrt_ratio_target_x96=(
                    sqrt_price_limit_x96 if
                    (sqrt_price_next_x96 < sqrt_price_limit_x96 if zeroForOne else sqrt_price_next_x96 > sqrt_price_limit_x96)
                    else sqrt_price_next_x96
                ),
                amount_remaining=amountSpecifiedRemaining
            )

            if exact_input:
                amountSpecifiedRemaining -= amountIn + feeAmount
                amountCalculated -= amountOut
            else:
                amountSpecifiedRemaining += amountOut
                amountCalculated += amountIn + feeAmount

            # if the protocol fee is on, calculate how much is owed, decrement feeAmount, and increment protocolFee
            if self.protocol_fee > 0:
                delta = feeAmount / self.protocol_fee
                feeAmount -= delta
                protocolFee_current += delta

            # update global fee tracker
            # if self.liquidity > 0:
            #     state.feeGrowthGlobalX128 += feeAmount / self.liquidity

            # shift tick if we reached the next price
            if self.sqrt_price_x96 == sqrt_price_next_x96:
                # if the tick is initialized, run the tick transition
                self.liquidity += next_tick.liquidityNet if zeroForOne else -next_tick.liquidityNet

        # update the agent's holdings
        if tkn_buy not in agent.holdings:
            agent.holdings[tkn_buy] = 0
        if exact_input:
            agent.holdings[tkn_buy] -= amountCalculated
            agent.holdings[tkn_sell] -= sell_quantity
        else:
            agent.holdings[tkn_buy] += buy_quantity
            agent.holdings[tkn_sell] -= amountCalculated

        return self


    def compute_swap_step(
            self,
            sqrt_ratio_current_x96: int,
            sqrt_ratio_target_x96: int,
            amount_remaining: float
    ) -> tuple[int, float, float, float]: # sqrt_ratio_nex, amountIn, amountOut, feeAmount

        zeroForOne = sqrt_ratio_current_x96 >= sqrt_ratio_target_x96
        exactIn = amount_remaining >= 0
        amountIn: float = 0
        amountOut: float = 0

        if exactIn:
            amountRemainingLessFee = int(amount_remaining * (1 - self.fee))
            amountIn = (  # calculate amount that it would take to reach our sqrt price target
                self.getAmount0Delta(sqrt_ratio_target_x96, sqrt_ratio_current_x96)
                if zeroForOne else
                self.getAmount1Delta(sqrt_ratio_current_x96, sqrt_ratio_target_x96)
            )
            if amountRemainingLessFee >= amountIn:
                sqrt_ratio_next_x96 = sqrt_ratio_target_x96
            else:
                sqrt_ratio_next_x96 = self.getNextSqrtPriceFromInput(
                    amount_in=amountRemainingLessFee,
                    zero_for_one=zeroForOne
                )
        else:
            amountOut = (  # calculate amount that it would take to reach our sqrt price target
                self.getAmount1Delta(sqrt_ratio_target_x96, sqrt_ratio_current_x96)
                if zeroForOne else
                self.getAmount0Delta(sqrt_ratio_current_x96, sqrt_ratio_target_x96)
            )
            if -amount_remaining >= amountOut:
                sqrt_ratio_next_x96 = sqrt_ratio_target_x96
            else:
                sqrt_ratio_next_x96 = self.getNextSqrtPriceFromOutput(
                    amount_out=-amount_remaining,
                    zero_for_one=zeroForOne
                )

        is_max = sqrt_ratio_target_x96 == sqrt_ratio_next_x96

        # get the input/output amounts
        if zeroForOne:
            if not (is_max and exactIn):
                amountIn = self.getAmount0Delta(sqrt_ratio_next_x96, sqrt_ratio_current_x96)
            if exactIn or not is_max:
                amountOut = self.getAmount1Delta(sqrt_ratio_next_x96, sqrt_ratio_current_x96)
        else:
            if not(is_max and exactIn):
                amountIn = self.getAmount1Delta(sqrt_ratio_current_x96, sqrt_ratio_next_x96)
            if exactIn or not is_max:
                amountOut = self.getAmount0Delta(sqrt_ratio_current_x96, sqrt_ratio_next_x96)

        # cap the output amount to not exceed the remaining output amount
        if not exactIn and amountOut > -amount_remaining:
            amountOut = -amount_remaining

        if exactIn and not is_max:
            # we didn't reach the target, so take the remainder of the maximum input as fee
            feeAmount = amount_remaining - amountIn
        else:
            feeAmount = amountIn * self.fee

        return (
            sqrt_ratio_next_x96,
            amountIn,
            amountOut,
            feeAmount
        )

    # /// @notice Gets the next sqrt price given an input amount of token0 or token1
    # /// @dev Throws if price or liquidity are 0, or if the next price is out of bounds
    # /// @param sqrtPX96 The starting price, i.e., before accounting for the input amount
    # /// @param liquidity The amount of usable liquidity
    # /// @param amountIn How much of token0, or token1, is being swapped in
    # /// @param zeroForOne Whether the amount in is token0 or token1
    # /// @return sqrtQX96 The price after adding the input amount to token0 or token1
    def getNextSqrtPriceFromInput(
        self,
        amount_in: float,
        zero_for_one: bool
    ):
        # // round to make sure that we don't pass the target price
        return (
            self.getNextSqrtPriceFromAmount0(amount=amount_in, add=True)
            if zero_for_one else
            self.getNextSqrtPriceFromAmount1(amount=amount_in, add=True)
        )

    # /// @notice Gets the next sqrt price given an output amount of token0 or token1
    # /// @dev Throws if price or liquidity are 0 or the next price is out of bounds
    # /// @param sqrtPX96 The starting price before accounting for the output amount
    # /// @param liquidity The amount of usable liquidity
    # /// @param amountOut How much of token0, or token1, is being swapped out
    # /// @param zeroForOne Whether the amount out is token0 or token1
    # /// @return sqrtQX96 The price after removing the output amount of token0 or token1
    def getNextSqrtPriceFromOutput(
        self,
        amount_out: float,
        zero_for_one: bool,
    ):
        # round to make sure that we pass the target price
        return (
            self.getNextSqrtPriceFromAmount1(amount=amount_out, add=False)
            if zero_for_one else
            self.getNextSqrtPriceFromAmount0(amount=amount_out, add=False)
        )

    # Gets the next sqrt price given a delta of token0
    # @param sqrtPX96 The starting price, i.e. before accounting for the token0 delta
    # @param liquidity The amount of usable liquidity
    # @param amount How much of token0 to add or remove from virtual reserves
    # @param add Whether to add or remove the amount of token0
    # @return The price after adding or removing amount, depending on add
    def getNextSqrtPriceFromAmount0(
        self,
        amount: float,
        add: bool,
    ):
        if amount == 0:
            # we short circuit amount == 0 because the result is otherwise not guaranteed to equal the input price
            return self.sqrt_price_x96
        numerator1 = self.liquidity * 2**96
        if add:
            denominator = numerator1 + amount * self.sqrt_price_x96
        else:
            denominator = numerator1 - amount * self.sqrt_price_x96
        return numerator1 * self.sqrt_price_x96 // denominator


    # Gets the next sqrt price given a delta of token1
    # @param sqrtPX96 The starting price, i.e., before accounting for the token1 delta
    # @param liquidity The amount of usable liquidity
    # @param amount How much of token1 to add, or remove, from virtual reserves
    # @param add Whether to add, or remove, the amount of token1
    # @return The price after adding or removing `amount`
    def getNextSqrtPriceFromAmount1(
        self,
        amount: float,
        add: bool
    ):
        if add:
            return self.sqrt_price_x96 + amount * 2**96 // self.liquidity
        else:
            return self.sqrt_price_x96 - amount * 2**96 // self.liquidity


    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        if fee is None:
            fee = self.fee
        if tkn_buy == self.asset_list[0]:
            return self.sqrt_price ** 2 * (1 + fee)
        else:
            return 1 / (self.sqrt_price ** 2) * (1 + fee)

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        if fee is None:
            fee = self.fee
        if tkn_sell == self.asset_list[0]:
            return self.sqrt_price ** 2 * (1 - fee)
        else:
            return 1 / (self.sqrt_price ** 2) * (1 - fee)