import pytest, copy, numpy as np
import math
from hypothesis import given, strategies as st, assume, settings, reproduce_failure

from hydradx.model.amm.agents import Agent
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

    directions = ['buy', 'sell', 'sell', 'buy']
    # when direction is 'buy', we expect Xi >= 0, Li = 0
    # when direction is 'sell', we expect Xi <= 0, Xi + Li = 0
    # note that for Omnipool, structure is [X0, X1, L0, L1, X2, X3, L2, L3, ...]
    l = [
         ([1, 0, 0, 0, 0, 0, 0, 0], True),  # AMM buying LRNA in first asset
        ([-1, 0, 1, 0, 0, 0, 0, 0], False),  # AMM selling LRNA in first asset
        ([0, 0, 0, 0, 0, 1, 0, 0], True),  # AMM buying asset in second asset
        ([0, 0, 0, 0, 0, -1, 0, 1], False),  # AMM selling asset in second asset
        ([0, 0, 0, 0, -1, 0, 1, 0], True),  # AMM selling LRNA in second asset
        ([0, 0, 0, 0, 1, 0, 0, 0], False),  # AMM buying LRNA in second asset
        ([1, -1, 0, 1, 0, 0, 0, 0], True),  # AMM trading in first asset, realistic
        ([1, -1, 0, 1, -1, 1, 1, 0], True),  # AMM trading in correct direction in all assets
        ([1, -1, 0, 1, -1, -1, 1, 1], False),
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


@given(st.lists(st.integers(min_value=-1, max_value=1), min_size=4, max_size=4),
       st.lists(st.integers(min_value=-2, max_value=2), min_size=8, max_size=8))
def test_get_amm_limits_A_random_omnipool(op_dirs, x_raw):
    liquidity = {"tkn": 1_000_000, "HDX": 1_000_000}
    pool = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': 100_000} for tkn in liquidity}
    )
    op_directions = []
    for i in op_dirs:
        if i == 1:
            op_directions.append('buy')
        elif i == -1:
            op_directions.append('sell')
        else:
            op_directions.append('none')
    x = [x_raw[i] for i in range(8)]
    last_amm_deltas = []
    # l = [[pool, op_directions, x_op]]

    amm_constraints = OmnipoolConstraints(pool)

    A_limits, b_limits, cones, cones_sizes = amm_constraints.get_amm_limits_A(op_directions, last_amm_deltas)
    assert A_limits.shape[1] == amm_constraints.k
    expected_result = True
    for i in range(len(pool.asset_list)):
        for j in range(2):
            if op_directions[2*i + j] == 'buy':  # expect that Xi >= 0, Li = 0
                if x[4*i + j + 2] != 0 or x[4*i + j] < 0:
                    expected_result = False
                    break
            elif op_directions[2*i + j] == 'sell':  # expect that Xi <= 0, Xi + Li == 0
                if x[4*i + j] > 0 or x[4*i + j] + x[4*i + j + 2] != 0:
                    expected_result = False
                    break
            else:  # expect that Xi + Li >= 0, Li >= 0 without direction info
                if x[4*i + j + 2] < 0 or x[4*i + j] + x[4*i + j + 2] < 0:
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


@given(
    st.floats(min_value=1e-5, max_value=1e-2),
    st.floats(min_value=1e-8, max_value=1e-7)
)
def test_get_omnipool_bounds(buy_mult_big, buy_mult_small):
    from hydradx.model.amm.omnipool_amm import simulate_swap
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1_000_000, 'LRNA': 100_000},
            "TKN": {'liquidity': 2_000_000, 'LRNA': 100_000}
        }
    )
    constraints = OmnipoolConstraints(omnipool)
    scaling = {tkn: 1 for tkn in constraints.asset_list}
    buy_amt_big = buy_mult_big * omnipool.liquidity['HDX']
    buy_amt_small = buy_mult_small * omnipool.liquidity['HDX']

    for approx, buy_amt in [["none", buy_amt_big], ["linear", buy_amt_small]]:
        A, b, cones, cone_sizes = constraints.get_amm_bounds([approx, approx], scaling)
        agent = Agent(enforce_holdings=False)
        new_omnipool, new_agent = simulate_swap(omnipool, agent, 'HDX', 'TKN', buy_quantity = buy_amt)
        sell_amt = -new_agent.holdings['TKN']
        lrna_delta = omnipool.lrna['TKN'] - new_omnipool.lrna['TKN']
        for m in [0.9, 1.1]:
            X_1 = -buy_amt
            X_3 = sell_amt * m
            X_0 = lrna_delta
            X_2 = -lrna_delta
            x = np.zeros(8)
            x[0], x[1], x[4], x[5] = X_0, X_1, X_2, X_3  # LRNA HDX
            for i in [2, 3, 6, 7]:
                x[i] = max([x[i-2], 0])  # L_i = max(X_i, 0)

            s = b - A @ x
            if (m >= 1) != check_all_cone_feasibility(s, cones, cone_sizes, tol=1e-10):
                raise AssertionError("Cone feasibility check failed for valid Omnipool bounds")


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


def test_omnipool_upgrade_approx():
    liquidity = {"tkn": 1_000_000, "HDX": 1_000_000}
    pool = OmnipoolState(
        tokens={tkn: {'liquidity': liquidity[tkn], 'LRNA': 100_000} for tkn in liquidity}
    )
    constraints = OmnipoolConstraints(pool)

    # current_approx is linear in both, we need to upgrade both
    current_approx = ["linear", "linear"]
    deltas = [pool.lrna['tkn'] / 100000, pool.liquidity["tkn"] / 100000, pool.lrna['HDX'] / 100000, pool.liquidity["HDX"] / 100000]
    if constraints.upgrade_approx(deltas, current_approx) != ["full", "full"]:
        raise AssertionError("Incorrect approximation")
    # current_approx is linear, we need to upgrade only HDX
    deltas = [pool.lrna['tkn'] / 10000000, pool.liquidity["tkn"] / 10000000, pool.lrna['HDX'] / 100000, pool.liquidity["HDX"] / 100000]
    if constraints.upgrade_approx(deltas, current_approx) != ["linear", "full"]:
        raise AssertionError("Incorrect approximation")
    # current_approx is linear, we do not need to upgrade
    deltas = [pool.lrna['tkn'] / 10000000, pool.liquidity["tkn"] / 10000000, pool.lrna['HDX'] / 10000000, pool.liquidity["HDX"] / 10000000]
    if constraints.upgrade_approx(deltas, current_approx) != ["linear", "linear"]:
        raise AssertionError("Incorrect approximation")
    current_approx = ["full", "linear"]
    # upgrade 2nd asset to full
    deltas = [pool.lrna['tkn'] / 100000, pool.liquidity["tkn"] / 100000, pool.lrna['HDX'] / 100000, pool.liquidity["HDX"] / 100000]
    if constraints.upgrade_approx(deltas, current_approx) != ["full", "full"]:
        raise AssertionError("Incorrect approximation")
    # do not upgrade anything
    deltas = [pool.lrna['tkn'] / 10000000, pool.liquidity["tkn"] / 10000000, pool.lrna['HDX'] / 10000000, pool.liquidity["HDX"] / 10000000]
    if constraints.upgrade_approx(deltas, current_approx) != ["full", "linear"]:
        raise AssertionError("Incorrect approximation")


@given(
    st.floats(min_value=1e-5, max_value=1e-2),
    st.floats(min_value=1e-8, max_value=1e-7)
)
def test_get_stableswap_bounds_swap(buy_mult_big, buy_mult_small):
    from hydradx.model.amm.stableswap_amm import simulate_swap
    amm = StableSwapPoolState(tokens={"A": 1_000_000, "B": 2_000_000}, amplification=100)
    constraints = StableswapConstraints(amm)
    scaling = {tkn: 1 for tkn in (amm.asset_list + [amm.unique_id])}
    buy_amt_big = buy_mult_big * amm.liquidity['A']
    buy_amt_small = buy_mult_small * amm.liquidity['A']

    # for approx, buy_amt in [["none", buy_amt_big], ["linear", buy_amt_small]]:
    for approx, buy_amt in [["none", buy_amt_big]]:
        A, b, cones, cone_sizes = constraints.get_amm_bounds([approx, approx, approx], scaling)
        agent = Agent(enforce_holdings=False)
        new_omnipool, new_agent = simulate_swap(amm, agent, 'B', 'A', buy_quantity = buy_amt)
        sell_amt = -new_agent.holdings['B']
        for m in [0.9, 1.1]:
            X_1 = -buy_amt
            X_2 = sell_amt * m
            x = np.zeros(9)
            x[1], x[2] = X_1, X_2
            for i in [4, 5]:
                x[i] = max([x[i-2], 0])  # L_i = max(X_i, 0)
            # need to set auxiliary variables too
            term0 = 1  # delta shares is zero for swap
            term1 = 1 + scaling[amm.asset_list[0]] / amm.liquidity[amm.asset_list[0]] * X_1
            a1 = term0 * math.log(term1 / term0)
            term2 = 1 + scaling[amm.asset_list[1]] / amm.liquidity[amm.asset_list[1]] * X_2
            a2 = term0 * math.log(term2 / term0)
            a0 = -a1 - a2
            x[6], x[7], x[8] = a0, a1, a2
            d_prime = amm.d * (1 - 1/amm.ann)
            denom = amm.liquidity["A"] + amm.liquidity["B"] - d_prime

            s = b - A @ x
            if (m >= 1) != check_all_cone_feasibility(s, cones, cone_sizes, tol=1e-6):
                raise AssertionError("Cone feasibility check failed for valid Stableswap bounds")

    # for approx, buy_amt in [["none", buy_amt_big], ["linear", buy_amt_small]]:
    for approx, buy_amt in [["linear", buy_amt_small]]:
        A, b, cones, cone_sizes = constraints.get_amm_bounds([approx, approx, approx], scaling)
        agent = Agent(enforce_holdings=False)
        new_omnipool, new_agent = simulate_swap(amm, agent, 'B', 'A', buy_quantity = buy_amt)
        sell_amt = -new_agent.holdings['B']

        X_1 = -buy_amt
        X_2 = sell_amt
        x = np.zeros(9)
        x[1], x[2] = X_1, X_2
        for i in [4, 5]:
            x[i] = max([x[i-2], 0])  # L_i = max(X_i, 0)
        # need to set auxiliary variables too
        a1 = scaling[amm.asset_list[0]] / amm.liquidity[amm.asset_list[0]] * X_1
        a2 = scaling[amm.asset_list[1]] / amm.liquidity[amm.asset_list[1]] * X_2
        a0 = -a1 - a2
        x[6], x[7], x[8] = a0, a1, a2
        d_prime = amm.d * (1 - 1/amm.ann)
        denom = amm.liquidity["A"] + amm.liquidity["B"] - d_prime

        s = b - A @ x
        if not check_all_cone_feasibility(s, cones, cone_sizes, tol=1e-10):
            raise AssertionError("Cone feasibility check failed for valid Stableswap bounds")

@reproduce_failure('6.127.0', b'ACg/gAAAAAAAACg+cAAAAAAAAA==')
@given(
    st.floats(min_value=1e-5, max_value=1e-2),
    st.floats(min_value=1e-8, max_value=1e-7)
)
def test_get_stableswap_bounds_liquidity(add_mult_big, add_mult_small):
    from hydradx.model.amm.stableswap_amm import simulate_add_liquidity, simulate_remove_liquidity
    amm = StableSwapPoolState(tokens={"A": 1_000_000, "B": 2_000_000}, amplification=100)
    constraints = StableswapConstraints(amm)
    scaling = {tkn: 1 for tkn in (amm.asset_list + [amm.unique_id])}
    add_amt_big = add_mult_big * amm.liquidity['A']
    add_amt_small = add_mult_small * amm.liquidity['A']


    A, b, cones, cone_sizes = constraints.get_amm_bounds(["none", "none", "none"], scaling)
    agent = Agent(enforce_holdings=False)
    new_omnipool, new_agent = simulate_add_liquidity(amm, agent, add_amt_big, 'A')
    shares_amt = new_agent.holdings[amm.unique_id]
    X_0 = shares_amt
    X_1 = add_amt_big
    x = np.zeros(9)
    x[0], x[1] = X_0, X_1
    for i in [3, 4, 5]:
        x[i] = max([x[i-2], 0])  # L_i = max(X_i, 0)
    # need to set auxiliary variables too
    term0 = 1 + scaling[amm.unique_id] / amm.shares * X_0
    term1 = 1 + scaling[amm.asset_list[0]] / amm.liquidity[amm.asset_list[0]] * X_1
    term2 = 1
    a1 = term0 * math.log(term1 / term0)
    a2 = term0 * math.log(term2 / term0)
    a0 = -a1 - a2
    x[6], x[7], x[8] = a0, a1, a2

    s = b - A @ x
    if not check_all_cone_feasibility(s, cones, cone_sizes, tol=1e-10):
        raise AssertionError("Cone feasibility check failed for valid Stableswap bounds")

    # for approx, buy_amt in [["none", buy_amt_big], ["linear", buy_amt_small]]:
    # for approx, buy_amt in [["linear", buy_amt_small]]:
    A, b, cones, cone_sizes = constraints.get_amm_bounds(["linear", "linear", "linear"], scaling)
    agent = Agent(enforce_holdings=False)
    new_omnipool, new_agent = simulate_add_liquidity(amm, agent, add_amt_small, 'A')
    shares_amt = new_agent.holdings[amm.unique_id]
    X_0 = shares_amt
    X_1 = add_amt_big
    x = np.zeros(9)
    x[0], x[1] = X_0, X_1
    for i in [3, 4, 5]:
        x[i] = max([x[i - 2], 0])  # L_i = max(X_i, 0)
    # need to set auxiliary variables too
    a1 = scaling[amm.asset_list[0]] / amm.liquidity[amm.asset_list[0]] * X_1 - scaling[amm.unique_id] / amm.shares * X_0
    a2 = -scaling[amm.unique_id] / amm.shares * X_0
    a0 = -a1 - a2
    x[6], x[7], x[8] = a0, a1, a2

    s = b - A @ x
    if not check_all_cone_feasibility(s, cones, cone_sizes, tol=1e-10):
        raise AssertionError("Cone feasibility check failed for valid Stableswap bounds")