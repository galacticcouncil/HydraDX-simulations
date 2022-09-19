from hydradx.model.amm.stableswap import calculate_y, calculate_d, spot_price

# fix value, i.e. fix p_x^y * x + y
D = 200000000
x_step_size = 500000
x_min = 10000000
x_max = 200000000
liq_depth = {}

# for A in range(10, 101, 10):
for A in [5, 10, 20, 50, 500]:
    ann = A * 4

    print("A is " + str(A))

    liq_depth[A] = [None] * ((x_max - x_min) // x_step_size)
    prices = [None] * ((x_max - x_min) // x_step_size)

    i = 0
    p_prev = 0
    for x in range(x_min, x_max + x_step_size, x_step_size):
        p_prev = p
        y = calculate_y(x, D, ann)
        p = spot_price([x, y], D, ann)
        if i > 0:
            liq_depth[A][i - 1] = x_step_size / (p - p_prev)
            prices[i - 1] = p
        # print((x, y, p))
        # print(liq_depth[A][i-1], p)
        i += 1
    # print(sum(liq_depth[A]))
    s = sum(liq_depth[A])
    for j in range(len(liq_depth[A])):
        # liq_depth[A][j] = liq_depth[A][j]/s
        liq_depth[A][j] = liq_depth[A][j]

