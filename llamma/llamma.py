import math


class LlammaState:
    def __init__(self, x, y, p_up, p_down, oracle_price):
        self.x = x
        self.y = y
        self.p_up = p_up
        self.p_down = p_down
        self.A = get_A(p_down, p_up)
        self.oracle_price = oracle_price
        self.y0 = solve_y0(x, y, oracle_price, p_up, self.A)
        self.f = get_f(oracle_price, p_up, self.A, self.y0)
        self.g = get_g(oracle_price, p_up, self.A, self.y0)
        self.pcd = calculate_pcd(oracle_price, p_up)
        self.pcu = calculate_pcu(oracle_price, p_down)

    def update(self, new_oracle_price=None):
        if new_oracle_price is not None:
            self.oracle_price = new_oracle_price
        self.y0 = solve_y0(self.x, self.y, self.oracle_price, self.p_up, self.A)
        self.f = get_f(self.oracle_price, self.p_up, self.A, self.y0)
        self.g = get_g(self.oracle_price, self.p_up, self.A, self.y0)
        self.pcd = calculate_pcd(self.oracle_price, self.p_up)
        self.pcu = calculate_pcu(self.oracle_price, self.p_down)

    def __repr__(self):
        return f'LlammaState(x={self.x}, y={self.y}, p_up={self.p_up}, p_down={self.p_down}, oracle_price={self.oracle_price}, A={self.A}, y0={self.y0}, f={self.f}, g={self.g}, pcd={self.pcd}, pcu={self.pcu})'


def xyk_out_given_in(x, y, dx):
    return x * y / (x + dx) - y


def concentrated_xyk_out_given_in(x, y, f, g, dx):
    return xyk_out_given_in(x + f, y + g, dx)


def xyk_spot_price(x, y):
    return x / y


def concentrated_xyk_spot_price(x, y, f, g):
    return xyk_spot_price(x + f, y + g)


def llamma_spot_price(state: LlammaState):
    return concentrated_xyk_spot_price(state.x, state.y, state.f, state.g)


def get_A(p_down, p_up):
    return 1 / (1 - p_down / p_up)


def get_f(p0, p_up, A, y0):
    return p0**2 / p_up * A * y0


def get_g(p0, p_up, A, y0):
    return p_up / p0 * (A - 1) * y0


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


def calculate_pcd(p0, p_up):
    return p0**3 / p_up**2


def calculate_pcu(p0, p_down):
    return p0**3 / p_down**2


def execute_llamma_sell(state: LlammaState, dx):
    dy = concentrated_xyk_out_given_in(state.x, state.y, state.f, state.g, dx)
    if state.y + dy < 0:
        dy = -state.y
        dx = concentrated_xyk_out_given_in(state.y, state.x, state.g, state.f, dy)
    return LlammaState(state.x + dx, state.y + dy, state.p_up, state.p_down, state.oracle_price)


def update_oracle_price(state: LlammaState, new_price):
    return LlammaState(state.x, state.y, state.p_up, state.p_down, new_price)


def calc_arb_dx(state: LlammaState):
    return math.sqrt(state.oracle_price * (state.g + state.y) * (state.x + state.f)) - (state.x + state.f)


def arb_llamma(state: LlammaState):
    dx = calc_arb_dx(state)
    return execute_llamma_sell(state, dx)
