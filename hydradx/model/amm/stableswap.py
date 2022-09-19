from .global_state import AMM
from .agents import Agent
from mpmath import mpf, mp
mp.dps = 50

N_COINS = 2  # I think we cannot currently go higher than this
# ann means how concentrated the liquidity is;
# the higher the number, the less the price changes as pool moves away from balance


class StableSwapPoolState:
    def __init__(self, tokens: dict, amplification: float):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There should only be two.
        """
        self.amplification = amplification
        self.liquidity = dict()
        self.asset_list: list[str] = []

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = mpf(quantity)


def has_converged(v0, v1, precision=1) -> bool:
    diff = abs(v0 - v1)
    if (v1 <= v0 and diff < precision) or (v1 > v0 and diff <= precision):
        return True
    return False


def calculate_d(xp, ann, N=128, precision=1):
    xp_sorted = sorted(xp)
    s = sum(xp_sorted)
    if s == 0:
        return 0

    d = s
    for i in range(N):

        d_p = d
        for x in xp_sorted:
            d_p *= d / (x * N_COINS)

        d_prev = d
        d = (ann * s + d_p * N_COINS) * d / ((ann - 1) * d + (N_COINS + 1) * d_p) + 2

        if has_converged(d_prev, d, precision):
            return d


def calculate_y(reserve, d, ann, N=128, precision=1):
    s = reserve
    c = d
    c *= d / (2 * reserve)
    c *= d / (ann * N_COINS)

    b = s + d / ann
    y = d
    for i in range(N):
        y_prev = y
        y = (y ** 2 + c) / (2 * y + b - d) + 2
        if has_converged(y_prev, y, precision):
            return y


# Calculate new amount of reserve OUT given amount to be added to the pool
def calculate_y_given_in(
    amount: float,
    reserve_in: float,
    reserve_out: float,
    ann: float,
    precision: int,
) -> float:
    new_reserve_in = reserve_in + amount
    d = calculate_d([reserve_in, reserve_out], ann, precision)
    return calculate_y(new_reserve_in, d, ann, precision)


# Calculate new amount of reserve IN given amount to be withdrawn from the pool
def calculate_y_given_out(
        amount: float,
        reserve_in: float,
        reserve_out: float,
        ann: float,
        precision: int,
) -> float:
    new_reserve_out = reserve_out - amount
    d = calculate_d([reserve_in, reserve_out], ann, precision)
    return calculate_y(new_reserve_out, d, ann, precision)


def calculate_asset_b_required(reserve_a, reserve_b, delta_a):
    updated_reserve_a = reserve_a + delta_a
    updated_reserve_b = updated_reserve_a * reserve_b / reserve_a


def spot_price(xp, d, ann):
    x, y = xp
    return (x / y) * (4 * ann * x * y ** 2 + d ** 3) / (4 * ann * x ** 2 * y + d ** 3)


reserves = [1000000000, 100000000]
ann = 4 * 10
d = calculate_d(reserves, ann)
print(f'spot price at {reserves}: {spot_price(reserves, d, ann)}')

# test that calculate_d and calculate_y are consistent
ann = 400
reserve_a = 100000000
reserve_b = 200000000
d = calculate_d([reserve_a, reserve_b], ann)
y = calculate_y(reserve_b, d, ann)

# fix value, i.e. fix p_x^y * x + y
D = 200000000
x_step_size = 500000
x_min = 10000000
x_max = 200000000
liq_depth = {}

