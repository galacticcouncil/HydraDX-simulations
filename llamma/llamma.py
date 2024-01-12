import math


class LlammaState:
    def __init__(self, x, y, p_up, p_down, oracle_price, A=None, y0=None):
        self.x = x
        self.y = y
        self.p_up = p_up
        self.p_down = p_down
        self.A = get_A(p_down, p_up) if A is None else A
        self.oracle_price = oracle_price
        self.y0 = solve_y0(x, y, oracle_price, p_up, self.A) if y0 is None else y0


def xyk_out_given_in(x, y, dx):
    return x * y / (x + dx) - y


def concentrated_xyk_out_given_in(x, y, f, g, dx):
    return xyk_out_given_in(x + f, y + g, dx)


def xyk_spot_price(x, y):
    return x / y


def concentrated_xyk_spot_price(x, y, f, g):
    return xyk_spot_price(x + f, y + g)


def get_A(p_down, p_up):
    return 1 / (1 - p_down / p_up)


def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    if b ** 2 - 4 * a * c < 0:
        return None
    return (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def solve_y0(x, y, price, p_up, A):
    a = price * A
    b = -p_up / price * (A - 1) * x - price ** 2 / p_up * A * y
    c = -x * y

    return solve_quadratic(a, b, c)


def execute_llamma_sell(state: LlammaState, dx):
    dy = xyk_out_given_in(state.x, state.y, dx)
    return LlammaState(state.x + dx, state.y + dy, state.p_up, state.p_down, state.oracle_price, state.A, state.y0)


def update_oracle_price(state: LlammaState, new_price):
    return LlammaState(state.x, state.y, state.p_up, state.p_down, new_price, state.A)
