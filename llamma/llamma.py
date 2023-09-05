import math

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
    return (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

def solve_y0(x, y, price, p_up, A):
    # p0 * A * y0^2 - y0 (pu / p0 * (A - 1) * x + p0^2 / pu * A * y) - xy = 0
    a = price * A
    b = -p_up / price * (A - 1) * x - price**2 / p_up * A * y
    c = -x * y

    return solve_quadratic(a, b, c)
