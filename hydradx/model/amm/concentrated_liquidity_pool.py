from math import sqrt as sqrt
import math
from agents import Agent
from amm import AMM

tick_increment = 1.0001
def tick_to_price(tick):
    return tick_increment ** tick

class LiquidityPosition:
    def __init__(self, assets: list, min_tick: int, max_tick: int, liquidity: float):
        if len(assets) != 2:
            raise ValueError("Expected 2 assets.")
        self.asset_x = assets[0]
        self.asset_y = assets[1]
        self.min_tick = min_tick
        self.max_tick = max_tick
        self.liquidity = liquidity
        self.fees_accrued = {tkn: 0 for tkn in assets}

    def __str__(self):
        return f"""
        assets: {self.asset_x}, {self.asset_y}
        min_tick: {self.min_tick} ({tick_to_price(self.min_tick)}), 
        max_tick: {self.max_tick} ({tick_to_price(self.max_tick)})
        """

class ConcentratedLiquidityPoolState (AMM):
    def __init__(
            self,
            tokens: list[str],
            fee: float,
            tick_spacing: int = 10,
            liquidity_positions: list[LiquidityPosition] = None,
            unique_id: str = None
    ):
        super().__init__()

        self.asset_list = tokens
        self.fee = fee
        self.tick_spacing = tick_spacing
        self.liquidity_positions = liquidity_positions or []
        self.unique_id = unique_id or f"{'_'.join(tokens)}_pool"
        for position in self.liquidity_positions:
            if position.asset_x not in self.asset_list or position.asset_y not in self.asset_list:
                raise ValueError("Liquidity position contains an asset not in the pool.")
            if position.min_tick % self.tick_spacing != 0 or position.max_tick % self.tick_spacing != 0:
                raise ValueError(f"Tick values must be multiples of the tick spacing ({self.tick_spacing}).")
        self.current_price = self.calculate_current_price()

    def calculate_current_price(self):
        # Initialize sums of x and y across all positions within the active range
        total_x = 0
        total_y = 0

        # Sum up the amounts of x and y from each position that is within the current price range
        for position in self.liquidity_positions:
            if position.min_tick <= self.current_price <= position.max_tick:
                sqrt_price_min = math.sqrt(tick_to_price(position.min_tick))
                sqrt_price_max = math.sqrt(tick_to_price(position.max_tick))
                L = position.liquidity

                # Calculate x and y for this position using the liquidity formula
                delta_x = L * (1 / sqrt_price_min - 1 / sqrt_price_max)
                delta_y = L * (sqrt_price_max - sqrt_price_min)

                total_x += delta_x
                total_y += delta_y

        # Calculate the current price from total_x and total_y
        if total_x > 0:
            self.current_price = total_y / total_x
        else:
            self.current_price = 0  # This would mean no active liquidity or an error state
        return self.current_price

    def update_price_after_swap(self, amount_in, is_token0):
        # Start by assuming the swap only affects the current price range
        total_x = 0
        total_y = 0

        # Calculate the new reserves based on the swap
        for position in self.liquidity_positions:
            if position.min_tick <= self.current_price <= position.max_tick:
                sqrt_price_min = math.sqrt(tick_to_price(position.min_tick))
                sqrt_price_max = math.sqrt(tick_to_price(position.max_tick))
                L = position.liquidity

                # Initial calculations for reserves
                delta_x = L * (1 / sqrt_price_min - 1 / sqrt_price_max)
                delta_y = L * (sqrt_price_max - sqrt_price_min)

                # Adjust reserves based on whether token0 or token1 was input
                if is_token0:
                    delta_x += amount_in  # Add the input to the appropriate reserve
                    # Recalculate output using the constant product formula
                    new_y = L / delta_x if delta_x != 0 else 0
                    amount_out = delta_y - new_y
                    delta_y = new_y
                else:
                    delta_y += amount_in
                    new_x = L / delta_y if delta_y != 0 else 0
                    amount_out = delta_x - new_x
                    delta_x = new_x

                total_x += delta_x
                total_y += delta_y

        # Update the current price based on new total reserves
        if total_x > 0:
            self.current_price = total_y / total_x
        else:
            self.current_price = 0  # Handle div by zero or no active liquidity

        return amount_out
    def get_nearest_price_tick(self, p):
        raw_tick = math.log(p) / math.log(tick_increment)
        nearest_valid_tick = round(raw_tick / self.tick_spacing) * self.tick_spacing
        return nearest_valid_tick

    def add_liquidity(self, agent: Agent, quantity: dict[str: float], min_tick: int, max_tick: int):
        if [key for key in list(quantity.keys()) if key in self.asset_list]:
            raise ValueError(f"Expected the same assets as the pool ({self.asset_list}).")
        liquidity = sqrt(quantity[0] * quantity[1])
        new_liquidity_position = LiquidityPosition(self.asset_list, min_tick, max_tick, liquidity)
        self.liquidity_positions.append(new_liquidity_position)
        for tkn in quantity:
            agent.holdings[tkn] -= quantity[tkn]
        if self.unique_id not in agent.holdings:
            agent.holdings[self.unique_id] = []
        agent.holdings[self.unique_id].append(new_liquidity_position)

    def swap(self, amount_in, is_token0):
        # Variables to track the state of the swap
        remaining_amount_in = amount_in
        amount_out = 0
        current_price = self.current_price

        # Sort liquidity positions based on whether we're buying or selling
        sorted_positions = sorted(self.liquidity_positions, key=lambda x: x.min_tick if is_token0 else -x.max_tick)

        for position in sorted_positions:
            if remaining_amount_in <= 0:
                break

            if position.min_tick <= current_price <= position.max_tick:
                # Calculate maximum swap amount within this position's price range
                max_amount_in_this_range, output_from_this_range = self.calculate_swap_within_range(
                    remaining_amount_in, current_price, position, is_token0)

                # Update remaining input amount and total output amount
                remaining_amount_in -= max_amount_in_this_range
                amount_out += output_from_this_range

                # Update the current price based on the output of this position
                self.update_price_after_swap(max_amount_in_this_range, is_token0)

        # Update the overall pool price after all possible swaps
        self.current_price = current_price
        return amount_out

    def calculate_swap_within_range(self, amount_in, current_price, position, is_token0):
        # Extract the liquidity and price bounds from the position
        L = position.liquidity
        sqrt_price_min = math.sqrt(tick_to_price(position.min_tick))
        sqrt_price_max = math.sqrt(tick_to_price(position.max_tick))

        if is_token0:
            # Calculate maximum amount of token0 that can be swapped within this range
            max_amount_in_this_range = L * (1 / sqrt_price_min - 1 / sqrt_price_max)
            if amount_in > max_amount_in_this_range:
                amount_in = max_amount_in_this_range

            # Calculate the amount of token1 received
            delta_y = L * (sqrt_price_max - sqrt_price_min)
            amount_out = delta_y * (amount_in / max_amount_in_this_range)
        else:
            # Calculate maximum amount of token1 that can be swapped within this range
            max_amount_in_this_range = L * (sqrt_price_max - sqrt_price_min)
            if amount_in > max_amount_in_this_range:
                amount_in = max_amount_in_this_range

            # Calculate the amount of token0 received
            delta_x = L * (1 / sqrt_price_min - 1 / sqrt_price_max)
            amount_out = delta_x * (amount_in / max_amount_in_this_range)

        return amount_in, amount_out

    def calculate_liquidity(self, token0_amount, token1_amount, min_tick, max_tick):
        # Convert tick to price (using square root of the price)
        sqrt_price_min = sqrt(tick_to_price(min_tick))
        sqrt_price_max = sqrt(tick_to_price(max_tick))

        # Calculate possible liquidity for each token
        liquidity_by_token0 = token0_amount / (sqrt_price_max - sqrt_price_min)
        liquidity_by_token1 = token1_amount * (sqrt_price_max - sqrt_price_min)

        return min(liquidity_by_token0, liquidity_by_token1)

    def __str__(self):
        return "x={:.2f} y={:.2f} p={:.2f} a={:.2f} b={:.2f}".format(self.x, self.y, self.p, self.a, self.b)

# Reference functions from https://github.com/atiselsts/uniswap-v3-liquidity-math/blob/master/uniswap-v3-liquidity-math.py
#
# Liquidity math adapted from https://github.com/Uniswap/uniswap-v3-periphery/blob/main/contracts/libraries/LiquidityAmounts.sol
#

def get_liquidity_0(x, sa, sb):
    return x * sa * sb / (sb - sa)


def get_liquidity_1(y, sa, sb):
    return y / (sb - sa)


def get_liquidity(x, y, sp, sa, sb):
    if sp <= sa:
        liquidity = get_liquidity_0(x, sa, sb)
    elif sp < sb:
        liquidity0 = get_liquidity_0(x, sp, sb)
        liquidity1 = get_liquidity_1(y, sa, sp)
        liquidity = min(liquidity0, liquidity1)
    else:
        liquidity = get_liquidity_1(y, sa, sb)
    return liquidity


#
# Calculate x and y given liquidity and price range
#
def calculate_x(L, sp, sa, sb):
    sp = max(min(sp, sb), sa)  # if the price is outside the range, use the range endpoints instead
    return L * (sb - sp) / (sp * sb)


def calculate_y(L, sp, sa, sb):
    sp = max(min(sp, sb), sa)  # if the price is outside the range, use the range endpoints instead
    return L * (sp - sa)


#
# Two different ways how to calculate p_a. calculate_a1() uses liquidity as an input, calculate_a2() does not.
#
def calculate_a1(L, sp, sb, x, y):
    # https://www.wolframalpha.com/input/?i=solve+L+%3D+y+%2F+%28sqrt%28P%29+-+a%29+for+a
    # sqrt(a) = sqrt(P) - y / L
    return (sp - y / L) ** 2


def calculate_a2(sp, sb, x, y):
    # https://www.wolframalpha.com/input/?i=solve+++x+sqrt%28P%29+sqrt%28b%29+%2F+%28sqrt%28b%29++-+sqrt%28P%29%29+%3D+y+%2F+%28sqrt%28P%29+-+a%29%2C+for+a
    # sqrt(a) = (y/sqrt(b) + sqrt(P) x - y/sqrt(P))/x
    #    simplify:
    # sqrt(a) = y/(sqrt(b) x) + sqrt(P) - y/(sqrt(P) x)
    sa = y / (sb * x) + sp - y / (sp * x)
    return sa ** 2


#
# Two different ways how to calculate p_b. calculate_b1() uses liquidity as an input, calculate_b2() does not.
#
def calculate_b1(L, sp, sa, x, y):
    # https://www.wolframalpha.com/input/?i=solve+L+%3D+x+sqrt%28P%29+sqrt%28b%29+%2F+%28sqrt%28b%29+-+sqrt%28P%29%29+for+b
    # sqrt(b) = (L sqrt(P)) / (L - sqrt(P) x)
    return ((L * sp) / (L - sp * x)) ** 2


def calculate_b2(sp, sa, x, y):
    # find the square root of b:
    # https://www.wolframalpha.com/input/?i=solve+++x+sqrt%28P%29+b+%2F+%28b++-+sqrt%28P%29%29+%3D+y+%2F+%28sqrt%28P%29+-+sqrt%28a%29%29%2C+for+b
    # sqrt(b) = (sqrt(P) y)/(sqrt(a) sqrt(P) x - P x + y)
    P = sp ** 2
    return (sp * y / ((sa * sp - P) * x + y)) ** 2


#
# Calculating c and d
#
def calculate_c(p, d, x, y):
    return y / ((d - 1) * p * x + y)


def calculate_d(p, c, x, y):
    return 1 + y * (1 - c) / (c * p * x)


#
# Test a known good combination of values against the functions provided above.
#
# Some errors are expected because:
#  -- the floating point math is meant for simplicity, not accurate calculations!
#  -- ticks and tick ranges are ignored for simplicity
#  -- the test values taken from Uniswap v3 UI and are approximate
#
def test(x, y, p, a, b):
    sp = p ** 0.5
    sa = a ** 0.5
    sb = b ** 0.5

    L = get_liquidity(x, y, sp, sa, sb)
    print("L: {:.2f}".format(L))

    ia = calculate_a1(L, sp, sb, x, y)
    error = 100.0 * (1 - ia / a)
    print("a: {:.2f} vs {:.2f}, error {:.6f}%".format(a, ia, error))

    ia = calculate_a2(sp, sb, x, y)
    error = 100.0 * (1 - ia / a)
    print("a: {:.2f} vs {:.2f}, error {:.6f}%".format(a, ia, error))

    ib = calculate_b1(L, sp, sa, x, y)
    error = 100.0 * (1 - ib / b)
    print("b: {:.2f} vs {:.2f}, error {:.6f}%".format(b, ib, error))

    ib = calculate_b2(sp, sa, x, y)
    error = 100.0 * (1 - ib / b)
    print("b: {:.2f} vs {:.2f}, error {:.6f}%".format(b, ib, error))

    c = sb / sp
    d = sa / sp

    ic = calculate_c(p, d, x, y)
    error = 100.0 * (1 - ic / c)
    print("c^2: {:.2f} vs {:.2f}, error {:.6f}%".format(c ** 2, ic ** 2, error))

    id = calculate_d(p, c, x, y)
    error = 100.0 * (1 - id ** 2 / d ** 2)
    print("d^2: {:.2f} vs {:.2f}, error {:.6f}%".format(d ** 2, id ** 2, error))

    ix = calculate_x(L, sp, sa, sb)
    error = 100.0 * (1 - ix / x)
    print("x: {:.2f} vs {:.2f}, error {:.6f}%".format(x, ix, error))

    iy = calculate_y(L, sp, sa, sb)
    error = 100.0 * (1 - iy / y)
    print("y: {:.2f} vs {:.2f}, error {:.6f}%".format(y, iy, error))
    print("")


def test_1():
    print("test case 1")
    p = 20.0
    a = 19.027
    b = 25.993
    x = 1
    y = 4
    test(x, y, p, a, b)


def test_2():
    print("test case 2")
    p = 3227.02
    a = 1626.3
    b = 4846.3
    x = 1
    y = 5096.06
    test(x, y, p, a, b)


def tests():
    test_1()
    test_2()


#
# Example 1 from the technical note
#
def example_1():
    print("Example 1: how much of USDC I need when providing 2 ETH at this price and range?")
    p = 2000
    a = 1500
    b = 2500
    x = 2

    sp = p ** 0.5
    sa = a ** 0.5
    sb = b ** 0.5
    L = get_liquidity_0(x, sp, sb)
    y = calculate_y(L, sp, sa, sb)
    print("amount of USDC y={:.2f}".format(y))

    # demonstrate that with the calculated y value, the given range is correct
    c = sb / sp
    d = sa / sp
    ic = calculate_c(p, d, x, y)
    id = calculate_d(p, c, x, y)
    C = ic ** 2
    D = id ** 2
    print("p_a={:.2f} ({:.2f}% of P), p_b={:.2f} ({:.2f}% of P)".format(
        D * p, D * 100, C * p, C * 100))
    print("")


#
# Example 2 from the technical note
#
def example_2():
    print("Example 2: I have 2 ETH and 4000 USDC, range top set to 3000 USDC. What's the bottom of the range?")
    p = 2000
    b = 3000
    x = 2
    y = 4000

    sp = p ** 0.5
    sb = b ** 0.5

    a = calculate_a2(sp, sb, x, y)
    print("lower bound of the price p_a={:.2f}".format(a))
    print("")


#
# Example 3 from the technical note
#
def example_3():
    print("Example 3: Using the position created in Example 2, what are asset balances at 2500 USDC per ETH?")
    p = 2000
    a = 1333.33
    b = 3000
    x = 2
    y = 4000

    sp = p ** 0.5
    sa = a ** 0.5
    sb = b ** 0.5
    # calculate the initial liquidity
    L = get_liquidity(x, y, sp, sa, sb)

    P1 = 2500
    sp1 = P1 ** 0.5

    x1 = calculate_x(L, sp1, sa, sb)
    y1 = calculate_y(L, sp1, sa, sb)
    print("Amount of ETH x={:.2f} amount of USDC y={:.2f}".format(x1, y1))

    # alternative way, directly based on the whitepaper

    # this delta math only works if the price is in the range (including at its endpoints),
    # so limit the square roots of prices to the range first
    sp = max(min(sp, sb), sa)
    sp1 = max(min(sp1, sb), sa)

    delta_p = sp1 - sp
    delta_inv_p = 1 / sp1 - 1 / sp
    delta_x = delta_inv_p * L
    delta_y = delta_p * L
    x1 = x + delta_x
    y1 = y + delta_y
    print("delta_x={:.2f} delta_y={:.2f}".format(delta_x, delta_y))
    print("Amount of ETH x={:.2f} amount of USDC y={:.2f}".format(x1, y1))


def examples():
    example_1()
    example_2()
    example_3()


def main():
    # test with some values taken from Uniswap UI
    tests()
    # demonstrate the examples given in the paper
    examples()


if __name__ == "__main__":
    main()