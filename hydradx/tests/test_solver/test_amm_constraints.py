import pytest, copy, numpy as np
from hypothesis import given, strategies as st, assume, settings, reproduce_failure

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.solver.amm_constraints import XykConstraints, StableswapConstraints, AmmIndexObject
from hydradx.model.amm.xyk_amm import ConstantProductPoolState

settings.register_profile("ci", deadline=None, print_blob=True)
settings.load_profile("ci")

def test_get_asset_indicator_matrix():
    global_asset_list = ['S', 'XYK-1', 'B', 'C', "SS-1"]

    amm = ConstantProductPoolState(tokens={'A': 1000, 'B': 2000}, unique_id='XYK-1')
    constraints = XykConstraints(amm)
    share_mat, asset_mat = constraints.get_indicator_matrices(global_asset_list)
    expected_share_matrix = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    expected_asset_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert share_mat.tolist() == expected_share_matrix
    assert asset_mat.tolist() == expected_asset_matrix

    amm = StableSwapPoolState(tokens = {'A': 1000, 'B': 2000}, amplification=100, unique_id="SS-1")
    constraints = StableswapConstraints(amm)
    share_mat, asset_mat = constraints.get_indicator_matrices(global_asset_list)
    expected_share_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ]
    expected_asset_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert share_mat.tolist() == expected_share_matrix
    assert asset_mat.tolist() == expected_asset_matrix


def check_all_cone_feasibility(s, cones, cone_sizes, tol=2e-5):
    from hydradx.model.solver.omnix_solver import check_cone_feasibility
    res = check_cone_feasibility(s, cones, cone_sizes, tol)
    for _, _, cone_feas in res:
        if not cone_feas:
            return False
    return True


def test_get_amm_limits_A():
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = ConstantProductPoolState(tokens=xyk_liquidity)
    amm_directions = []
    last_amm_deltas = []

    for amm, trading_pairs in [[stablepool4, [list(range(ss_tkn_ct + 1)), [0, 2], [1, 3]]], [xyk, [[1, 2]]]]:
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        amm_i = amm_constraints.amm_i
        for trading_is in trading_pairs:
            trading_tkns = [all_trading_tkns[i] for i in trading_is]
            A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(amm_directions, last_amm_deltas,
                                                                                     trading_tkns)

            assert A_limits.shape[1] == amm_constraints.k
            # assert A_limits.shape[0] == 2 * len(trading_tkns) + 2 * (tkn_ct + 1 - len(trading_tkns))
            # in this case, A_limits should be enforcing that Li >= 0 for the restricted trading tokens
            # it should be enforcing that Xi == 0 and Li == 0 for the other tokens
            x = np.zeros(amm_constraints.k)
            for i in trading_is:
                x[amm_i.shares_out + i] = 1  # set correct Lis to 1
            s = b_limits - A_limits @ x
            if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                raise AssertionError("Cone feasibility check failed for Li >= 0")
            # check if appropriate Xis != 0
            for val in [1, -1]:
                x_copy = copy.deepcopy(x)
                for i in trading_is:
                    x_copy[amm_i.shares_net + i] = val  # set correct Xis to 1
                s = b_limits - A_limits @ x_copy
                if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                    raise AssertionError("Cone feasibility check failed for Xis != 0")
            # next we check what happens if we make one of the restricted Lis negative
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_out + i] = -1  # make one of the Ls negative
                s = b_limits - A_limits @ x_copy
                if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                    raise AssertionError("Cone feasibility check should fail for negative Li")
            # next we check what happens if we make one of the other Xs or Ls non-zero
            non_trading_is = [i for i in range(len(amm.asset_list) + 1) if i not in trading_is]
            for i in non_trading_is:
                for val in [-1, 1]:
                    for offset in [amm_i.shares_net, amm_i.shares_out]:
                        x_copy = copy.deepcopy(x)
                        x_copy[offset + i] = val  # make variable nonzero
                        s = b_limits - A_limits @ x_copy
                        if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
                            raise AssertionError("Cone feasibility check should fail for non-trading variables")
            # next we check what happens if we make Xj + Lj < 0
            for i in trading_is:
                x_copy = copy.deepcopy(x)
                x_copy[amm_i.shares_net + i] = -2  # Xj + Lj = -1
                s = b_limits - A_limits @ x_copy
                assert not check_all_cone_feasibility(s, cones, cones_sizes, tol=0)


def test_get_amm_limits_A_specific():
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = ConstantProductPoolState(tokens=xyk_liquidity)
    last_amm_deltas = []

    examples = []
    # unknown directions, all trading tokens in xyk pool
    directions = ['none', 'none', 'none']
    # For all of these we expect Xi + Li >= 0, Li >= 0 for all tokens
    l = [
         ([-1, 0, 0, 1, 0, 0], True),
         ([1, 0, 0, -1, 0, 0], False),
         ([0, -1, 0, 0, 1, 0], True),
         ([0, 1, 0, 0, -1, 0], False),
         ([0, -1, 0, 0, -1, 0], False),
         ([0, 0, 1, 0, 0, 1], True),
        ([0, 0, 0, 0, 0, 1], True),
        ([0, 1, 0, 0, 0, 1], True),
         ([0, 0, 0, 0, 0, 0], True)
    ]
    for x, result in l:
        examples.append({'amm': xyk, 'directions': directions, 'trading_is': [0, 1, 2], 'sol': x, 'result': result})

    # unknown directions, subset of trading tokens in stableswap pool
    directions = ['none', 'none', 'none', 'none', 'none']
    trading_is = [0, 2, 3]
    # For trading tokens we expect Xi + Li >= 0, Li >= 0. For non-trading tokens we expect Xi == 0, Li == 0
    l  = [
        ([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0], True),
        ([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], False),  # trading token with nonzero vars
        ([0, -1, 0, 0, 0, 0, 1, 0, 0, 0], False),  # trading token with nonzero vars
        ([0, -1, 0, 0, 0, 1, 1, 0, 0, 0], False),  # trading token with nonzero vars
        ([-1, 0, 1, -1, 0, 1, 0, 1, 1, 0], True),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True)
        ]

    for x, result in l:
        examples.append({'amm': stablepool4, 'directions': directions, 'trading_is': trading_is, 'sol': x + [0, 0, 0, 0, 0], 'result': result})

    # some directions known, some tokens trading, some not in stableswap pool
    directions = ['buy', 'buy', 'sell', 'none', 'buy']
    trading_is = [0, 2, 3, 4]
    # For trading tokens we expect Xi + Li >= 0, Li >= 0. For non-trading tokens we expect Xi == 0, Li == 0
    l  = [
        ([1, 0, -1, 1, 1, 0, 0, 1, 1, 0], True),
        ([0, 0, -1, 1, 1, 0, 0, 1, 1, 0], True),  # tradeable asset doesn't trade
        ([1, 1, -1, 1, 1, 0, 0, 1, 1, 0], False),  # non-tradeable asset trades
        ([1, 0, -1, 1, 1, 0, 0, 1, 1, 1], False),  # buy asset with Li > 0
        ([1, 0, -1, 1, -1, 0, 0, 1, 1, 1], False),  # buy asset with Xi < 0
        ([1, 0, 0, 1, 1, 0, 0, 1, 1, 0], False),  # sell asset with Xi + Li > 0
        ([1, 0, 1, 1, 1, 0, 0, -1, 1, 0], False),  # sell asset with Xi > 0
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True)
    ]

    for x, result in l:
        examples.append({'amm': stablepool4, 'directions': directions, 'trading_is': trading_is, 'sol': x + [0, 0, 0, 0, 0], 'result': result})

    for ex in examples:
        amm = ex['amm']
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        trading_tkns = [all_trading_tkns[i] for i in ex['trading_is']]
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(ex['directions'], last_amm_deltas,
                                                                                  trading_tkns)
        s = b_limits - A_limits @ ex['sol']
        assert A_limits.shape[1] == amm_constraints.k
        feas = check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
        if feas != ex['result']:
            raise AssertionError(f"Cone feasibility check failed with x={ex['sol']}, expected {ex['result']}, got {feas}")


@given(st.lists(st.integers(min_value=-1, max_value=1), min_size=5, max_size=5),
       st.lists(st.integers(min_value=-1, max_value=1), min_size=3, max_size=3),
       st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=5, unique=True),
       st.lists(st.integers(min_value=0, max_value=2), min_size=2, max_size=3, unique=True),
       st.lists(st.integers(min_value=-2, max_value=2), min_size=10, max_size=10))
def test_get_amm_limits_A_random(ss_dirs, xyk_dirs, ss_trading_indices, xyk_trading_indices, x_raw):
    ss_tkn_ct = 4
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(ss_tkn_ct)}
    stablepool4 = StableSwapPoolState(tokens=liquidity, amplification=1000)
    xyk_liquidity = {"xyk1": 1_000_000, "xyk2": 1_000_000}
    xyk = ConstantProductPoolState(tokens=xyk_liquidity)
    xyk_directions = []
    for i in xyk_dirs:
        if i == 1:
            xyk_directions.append('buy')
        elif i == -1:
            xyk_directions.append('sell')
        else:
            xyk_directions.append('none')
    ss_directions = []
    for i in ss_dirs:
        if i == 1:
            ss_directions.append('buy')
        elif i == -1:
            ss_directions.append('sell')
        else:
            ss_directions.append('none')
    x_xyk = [x_raw[i] for i in range(6)]
    x_ss = [x for x in x_raw] + [0] * 5
    last_amm_deltas = []

    for amm, directions, trading_is, x in [[stablepool4, ss_directions, ss_trading_indices, x_ss],
                                           [xyk, xyk_directions, xyk_trading_indices, x_xyk]]:
        all_trading_tkns = [amm.unique_id] + list(amm.liquidity.keys())
        trading_tkns = [all_trading_tkns[i] for i in trading_is]
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        amm_i = amm_constraints.amm_i

        A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(directions, last_amm_deltas,
                                                                                  trading_tkns)
        assert A_limits.shape[1] == amm_constraints.k
        expected_result = True
        for i in range(len(amm.asset_list) + 1):
            if i not in trading_is:  # expect that Xi == 0 and Li == 0 for non-trading tokens
                if x[amm_i.shares_out + i] != 0 or x[amm_i.shares_net + i] != 0:
                    expected_result = False
                    break
            elif directions[i] == 'buy':  # expect that Xi >= 0, Li = 0
                if x[amm_i.shares_out + i] != 0 or x[amm_i.shares_net + i] < 0:
                    expected_result = False
                    break
            elif directions[i] == 'sell':  # expect that Xi <= 0, Xi + Li == 0
                if x[amm_i.shares_net + i] > 0 or x[amm_i.shares_net + i] + x[amm_i.shares_out + i] != 0:
                    expected_result = False
                    break
            else:  # expect that Xi + Li >= 0, Li >= 0 for trading tokens without direction info
                if x[amm_i.shares_out + i] < 0 or x[amm_i.shares_net + i] + x[amm_i.shares_out + i] < 0:
                    expected_result = False
                    break
        s = b_limits - A_limits @ x
        feas = check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
        if feas != expected_result:
            raise AssertionError(f"Cone feasibility check failed with x={x}, expected {expected_result}, got {feas}")


def test_get_xyk_bounds():
    amm = ConstantProductPoolState(tokens={"A": 1_000_000, "B": 2_000_000})  # spot price is 2 B = 1 A
    constraints = XykConstraints(amm)
    scaling = {tkn: 1 for tkn in (amm.asset_list + [amm.unique_id])}
    amm_i = constraints.amm_i

    for approx in ["none", "linear"]:  # mult = 10 should test full approximation, 0.1 should test linear approximation
        A, b, cones, cones_sizes = constraints.get_amm_bounds(approx, scaling)
        x = np.zeros(constraints.k)
        # selling 5 B for 1 A should work
        b_sell_amt, a_buy_amt = 5, 1
        x[amm_i.asset_net[0]] = -a_buy_amt
        x[amm_i.asset_net[1]] = b_sell_amt
        x[amm_i.asset_out[0]] = a_buy_amt
        s = b - A @ x
        if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
            raise AssertionError("Cone feasibility check failed for valid XYK bounds")
        # selling 1 A for 1 5 should not work
        a_sell_amt, b_buy_amt = 1, 5
        x[amm_i.asset_net[1]] = -b_buy_amt
        x[amm_i.asset_net[0]] = a_sell_amt
        x[amm_i.asset_out[1]] = b_buy_amt
        s = b - A @ x
        if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
            raise AssertionError("Cone feasibility check should fail")
        # selling 1 A for 1 B should work
        a_sell_amt, b_buy_amt = 1, 1
        x[amm_i.asset_net[1]] = -b_buy_amt
        x[amm_i.asset_net[0]] = a_sell_amt
        x[amm_i.asset_out[1]] = b_buy_amt
        s = b - A @ x
        if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
            raise AssertionError("Cone feasibility check should succeed")
        # selling 1 B for 1 A should not work
        b_sell_amt, a_buy_amt = 1, 1
        x[amm_i.asset_net[0]] = -a_buy_amt
        x[amm_i.asset_net[1]] = b_sell_amt
        x[amm_i.asset_out[0]] = a_buy_amt
        s = b - A @ x
        if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
            raise AssertionError("Cone feasibility check should fail")


def test_xyk_upgrade_approx():
    amm = ConstantProductPoolState(tokens={"A": 1_000_000, "B": 2_000_000})  # spot price is 2 B = 1 A
    constraints = XykConstraints(amm)

    # current_approx is linear, we need to upgrade to full because of shares
    current_approx = "linear"
    deltas = [amm.shares / 10000, amm.liquidity["A"] / 1000000, amm.liquidity["B"] / 1000000]
    if constraints.upgrade_approx(deltas, current_approx) != "full":
        raise AssertionError("Upgrade to full approximation should be required due to shares delta")
    # current_approx is linear, we need to upgrade to full because of an asset
    current_approx = "linear"
    deltas = [amm.shares / 1000000, amm.liquidity["A"] / 10000, amm.liquidity["B"] / 1000000]
    if constraints.upgrade_approx(deltas, current_approx) != "full":
        raise AssertionError("Upgrade to full approximation should be required due to asset delta")
    # current_approx is full, all deltas are low enough for linear approximation
    current_approx = "full"
    deltas = [amm.shares / 10000, amm.liquidity["A"] / 10000, amm.liquidity["B"] / 10000]
    if constraints.upgrade_approx(deltas, current_approx) != "full":
        raise AssertionError("Full approximation should never be downgraded to linear")
    # current_approx is linear, all deltas are low enough for linear approximation
    current_approx = "linear"
    deltas = [amm.shares / 1000000, amm.liquidity["A"] / 1000000, amm.liquidity["B"] / 1000000]
    if constraints.upgrade_approx(deltas, current_approx) != "linear":
        raise AssertionError("Linear approximation should be kept as deltas are low enough")
    # current_approx is full, correct approximation is full
    current_approx = "full"
    deltas = [amm.shares / 10000, amm.liquidity["A"] / 1000000, amm.liquidity["B"] / 1000000]
    if constraints.upgrade_approx(deltas, current_approx) != "full":
        raise AssertionError("Full approximation should always be kept")


def test_stableswap_upgrade_approx():
    amm = StableSwapPoolState(tokens={"A": 1_000_000, "B": 2_000_000}, amplification=100)  # spot price is 2 B = 1 A
    constraints = StableswapConstraints(amm)

    # current_approx is entirely linear
    current_approx = ["linear", "linear", "linear"]
    examples = [  # [delta_mults, expected_approx]
        [[1e-4, 1e-7, 1e-7], ["full", "linear", "linear"]],
        [[1e-7, 1e-4, 1e-7], ["full", "full", "linear"]],
        [[1e-7, 1e-6, 1e-4], ["full", "linear", "full"]],
        [[1e-4, 1e-4, 1e-4], ["full", "full", "full"]],
        [[1e-7, 1e-7, 1e-7], ["linear", "linear", "linear"]]
    ]
    for delta_mults, expected_approx in examples:
        deltas = [amm.shares * delta_mults[0], amm.liquidity["A"] * delta_mults[1], amm.liquidity["B"] * delta_mults[2]]
        real_approx = constraints.upgrade_approx(deltas, current_approx)
        for i, approx in enumerate(real_approx):
            if approx != expected_approx[i]:
                raise AssertionError(f"Expected {expected_approx[i]} for {amm.asset_list[i]} but got {approx}")

    # current_approx is mixed
    current_approx = ["full", "linear", "full"]
    examples = [  # [delta_mults, expected_approx]
        [[1e-4, 1e-7, 1e-7], ["full", "linear", "full"]],
        [[1e-7, 1e-4, 1e-7], ["full", "full", "full"]],
        [[1e-7, 1e-7, 1e-4], ["full", "linear", "full"]],
        [[1e-4, 1e-4, 1e-4], ["full", "full", "full"]],
        [[1e-7, 1e-7, 1e-7], ["full", "linear", "full"]]
    ]
    for delta_mults, expected_approx in examples:
        deltas = [amm.shares * delta_mults[0], amm.liquidity["A"] * delta_mults[1], amm.liquidity["B"] * delta_mults[2]]
        real_approx = constraints.upgrade_approx(deltas, current_approx)
        for i, approx in enumerate(real_approx):
            if approx != expected_approx[i]:
                raise AssertionError(f"Expected {expected_approx[i]} for {amm.asset_list[i]} but got {approx}")


# TODO test with auxiliary variable calculation
# def test_get_stableswap_bounds():
#     amm = StableSwapPoolState(tokens={"A": 1_000_000, "B": 2_000_000}, amplification=100)
#     constraints = StableswapConstraints(amm)
#     scaling = {tkn: 1 for tkn in (amm.asset_list + [amm.unique_id])}
#     amm_i = constraints.amm_i
#
#     for approx in ["Linear", "None"]:
#         A, b, cones, cones_sizes = constraints.get_amm_bounds(approx, scaling)
#         x = np.zeros(constraints.k)
#         # selling 5 B for 1 A should work
#         b_sell_amt, a_buy_amt = 5, 1
#         x[amm_i.asset_net[0]] = -a_buy_amt
#         x[amm_i.asset_net[1]] = b_sell_amt
#         x[amm_i.asset_out[0]] = a_buy_amt
#         s = b - A @ x
#         if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
#             raise AssertionError("Cone feasibility check failed for valid Stableswap bounds")
#         # selling 1 A for 1 5 should not work
#         a_sell_amt, b_buy_amt = 1, 5
#         x[amm_i.asset_net[1]] = -b_buy_amt
#         x[amm_i.asset_net[0]] = a_sell_amt
#         x[amm_i.asset_out[1]] = b_buy_amt
#         s = b - A @ x
#         if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
#             raise AssertionError("Cone feasibility check should fail")
#         # selling 5 A for 1 B should work
#         a_sell_amt, b_buy_amt = 5, 1
#         x[amm_i.asset_net[1]] = -b_buy_amt
#         x[amm_i.asset_net[0]] = a_sell_amt
#         x[amm_i.asset_out[1]] = b_buy_amt
#         s = b - A @ x
#         if not check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
#             raise AssertionError("Cone feasibility check should succeed")
#         # selling 1 B for 5 A should not work
#         b_sell_amt, a_buy_amt = 1, 5
#         x[amm_i.asset_net[0]] = -a_buy_amt
#         x[amm_i.asset_net[1]] = b_sell_amt
#         x[amm_i.asset_out[0]] = a_buy_amt
#         s = b - A @ x
#         if check_all_cone_feasibility(s, cones, cones_sizes, tol=0):
#             raise AssertionError("Cone feasibility check should fail")
