import math

def sell_spot(reserves_in, reserves_out, f):
    return reserves_out / reserves_in * (1 - f)

def execute_sell(reserves_in, reserves_out, delta_in, f) -> tuple:
    assert(delta_in >= 0)
    new_in = reserves_in + delta_in
    delta_out = - reserves_out * delta_in / new_in * (1 - f)
    return (new_in, reserves_out + delta_out)


X = 3000000
Y = 100000
p = 0.8324
f = 0.003

s = sell_spot(Y, X, f)
if p < s:
    b = 2 * Y - (f / p) * X * (1 - f)
    c = Y**2 - X * Y / p * (1 - f)
    t = math.sqrt(b ** 2 - 4 * c)
    if -b < t:
        dY = (-b + t) / 2
    else:
        dY = (-b - t) / 2

    (new_in, new_out) = execute_sell(Y, X, dY, f)
    print((new_in, new_out))
    print("This is correct if it recovers p:")
    print(sell_spot(new_in, new_out, f))