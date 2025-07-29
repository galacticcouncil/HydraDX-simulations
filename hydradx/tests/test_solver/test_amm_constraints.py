import pytest, copy, numpy as np
from hypothesis import given, strategies as st, assume, settings, reproduce_failure

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.solver.amm_constraints import XykConstraints, StableswapConstraints, AmmIndexObject, \
    OmnipoolConstraints
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
        examples.append({'amm': xyk, 'directions': directions, 'sol': x, 'result': result})

    # unknown directions, subset of trading tokens in stableswap pool
    directions = ['none', 'none', 'none', 'none', 'none']
    # we expect Xi + Li >= 0, Li >= 0
    l  = [
        ([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0], True),
        ([1, 0, 0, 0, 0, -1, 0, 0, 0, 0], False),  # Li <= 0
        ([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # Xi + Li <= 0
        ([0, -1, 0, 0, 0, 0, 1, 0, 0, 0], True),
        ([-1, 0, 1, -1, 0, 1, 0, 1, 1, 0], True),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True)
        ]

    for x, result in l:
        examples.append({'amm': stablepool4, 'directions': directions, 'sol': x + [0, 0, 0, 0, 0], 'result': result})

    # some directions known, some tokens trading, some not in stableswap pool
    directions = ['buy', 'buy', 'sell', 'none', 'buy']
    # we expect Xi + Li >= 0, Li >= 0. For non-trading tokens we expect Xi == 0, Li == 0
    l  = [
        ([1, 0, -1, 1, 1, 0, 0, 1, 1, 0], True),
        ([0, 0, -1, 1, 1, 0, 0, 1, 1, 0], True),  # asset doesn't trade
        ([1, 0, -1, 1, 1, 0, 0, 1, 1, 1], False),  # buy asset with Li > 0
        ([1, 0, -1, 1, -1, 0, 0, 1, 1, 1], False),  # buy asset with Xi < 0
        ([1, 0, 0, 1, 1, 0, 0, 1, 1, 0], False),  # sell asset with Xi + Li > 0
        ([1, 0, 1, 1, 1, 0, 0, -1, 1, 0], False),  # sell asset with Xi > 0
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True)
    ]

    for x, result in l:
        examples.append({'amm': stablepool4, 'directions': directions, 'sol': x + [0, 0, 0, 0, 0], 'result': result})

    for ex in examples:
        amm = ex['amm']
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(ex['directions'], last_amm_deltas)
        s = b_limits - A_limits @ ex['sol']
        assert A_limits.shape[1] == amm_constraints.k
        feas = check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
        if feas != ex['result']:
            raise AssertionError(f"Cone feasibility check failed with x={ex['sol']}, expected {ex['result']}, got {feas}")


@given(st.lists(st.integers(min_value=-1, max_value=1), min_size=5, max_size=5),
       st.lists(st.integers(min_value=-1, max_value=1), min_size=3, max_size=3),
       st.lists(st.integers(min_value=-2, max_value=2), min_size=10, max_size=10))
def test_get_amm_limits_A_random(ss_dirs, xyk_dirs, x_raw):
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

    for amm, directions, x in [[stablepool4, ss_directions, x_ss], [xyk, xyk_directions, x_xyk]]:
        if isinstance(amm, ConstantProductPoolState):
            amm_constraints = XykConstraints(amm)
        elif isinstance(amm, StableSwapPoolState):
            amm_constraints = StableswapConstraints(amm)
        amm_i = amm_constraints.amm_i

        A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(directions, last_amm_deltas)
        assert A_limits.shape[1] == amm_constraints.k
        expected_result = True
        for i in range(len(amm.asset_list) + 1):
            if directions[i] == 'buy':  # expect that Xi >= 0, Li = 0
                if x[amm_i.shares_out + i] != 0 or x[amm_i.shares_net + i] < 0:
                    expected_result = False
                    break
            elif directions[i] == 'sell':  # expect that Xi <= 0, Xi + Li == 0
                if x[amm_i.shares_net + i] > 0 or x[amm_i.shares_net + i] + x[amm_i.shares_out + i] != 0:
                    expected_result = False
                    break
            else:  # expect that Xi + Li >= 0, Li >= 0 without direction info
                if x[amm_i.shares_out + i] < 0 or x[amm_i.shares_net + i] + x[amm_i.shares_out + i] < 0:
                    expected_result = False
                    break
        s = b_limits - A_limits @ x
        feas = check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
        if feas != expected_result:
            raise AssertionError(f"Cone feasibility check failed with x={x}, expected {expected_result}, got {feas}")


def test_get_limits_A_omnipool_specific():
    tkn_ct = 2
    liquidity = {f"tkn_{i}": 1_000_000 for i in range(tkn_ct-1)}
    liquidity['HDX'] = 1_000_000
    pool = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': 100_000} for tkn in liquidity}
    )
    last_omnipool_deltas = []

    examples = []
    # unknown directions
    directions = ['none'] * 2 * tkn_ct
    # For all of these we expect Xi + Li >= 0, Li >= 0 for all tokens
    # note that for Omnipool, structure is [X0, X1, L0, L1, X2, X3, L2, L3, ...]
    l = [
         ([-1, 0, 1, 0, 0, 0, 0, 0], True),  # LRNA variable works
        ([0, -1, 0, 1, 0, 0, 0, 0], True),  # non-LRNA variable works
         ([1, 0, -1, 0, 0, 0, 0, 0], False),
        ([0, 1, 0, -1, 0, 0, 0, 0], False),
        ([-1, 0, 1, 0, -1, 0, 1, 0], True),  # mix different tkns
        ([0, -1, 0, 1, 1, 0, -1, 0], False),
        ([-1, -1, 1, 1, 0, 0, 0, 0], True),  # this satisfies the Xi, Li constraints but would fail AMM invariant
         ([0, 0, 0, 0, 0, 0, 0, 0], True)
    ]
    for x, result in l:
        examples.append({'directions': directions, 'sol': x, 'result': result})

    for ex in examples:
        amm_constraints = OmnipoolConstraints(pool)
        A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(ex['directions'], last_omnipool_deltas)
        s = b_limits - A_limits @ ex['sol']
        assert A_limits.shape[1] == amm_constraints.k
        feas = check_all_cone_feasibility(s, cones, cones_sizes, tol=0)
        if feas != ex['result']:
            raise AssertionError(f"Cone feasibility check failed with x={ex['sol']}, expected {ex['result']}, got {feas}")


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
