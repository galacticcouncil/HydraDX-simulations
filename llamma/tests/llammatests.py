import pytest

from llamma.llamma import xyk_out_given_in, concentrated_xyk_out_given_in, get_A, solve_quadratic, solve_y0, \
    execute_llamma_sell, LlammaState
from hypothesis import given, strategies as st, settings, reproduce_failure


@given(st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=100, max_value=1000)
       )
def test_xyk_out_given_in(x, y, dx):
    dy = xyk_out_given_in(x, y, dx)
    if x * y != pytest.approx((x + dx) * (y + dy), rel=1e-15):
        raise


@given(st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=10000, max_value=100000),
       st.floats(min_value=100, max_value=1000)
       )
def test_concentrated_xyk_out_given_in(x, y, f, g, dx):
    dy = concentrated_xyk_out_given_in(x, y, f, g, dx)
    assert (x + f) * (y + g) == pytest.approx((x + f + dx) * (y + g + dy), rel=1e-15)


@given(st.floats(min_value=0.00001, max_value=10000.0),
         st.floats(min_value=1e-6, max_value=1 - 1e-6)
         )
def test_get_A(p_up, p_ratio):
    p_down = p_up * p_ratio
    A = get_A(p_down, p_up)
    assert p_down / p_up == pytest.approx((A - 1) / A, rel=1e-15)


@given(st.floats(min_value=-10000.0, max_value=10000.0),
         st.floats(min_value=-10000.0, max_value=10000.0),
         st.floats(min_value=-10000.0, max_value=10000.0)
         )
def test_solve_quadratic(a, b, c):
    x = solve_quadratic(a, b, c)
    if x is not None:
        assert a * x**2 + b * x + c == pytest.approx(0, abs=1e-6)


@given(st.floats(min_value=0.00001, max_value=10000.0),
         st.floats(min_value=1e-6, max_value=1 - 1e-6),
         st.floats(min_value=1e-6, max_value=1 - 1e-6),
        st.floats(min_value=100, max_value=100000),
        st.floats(min_value=100, max_value=100000)
         )
def test_solve_y0(p0, p_ratio_1, p_ratio_2, x, y):
    p_up = p0 / p_ratio_1
    p_down = p0 * p_ratio_2
    A = get_A(p_down, p_up)
    y0 = solve_y0(x, y, p0, p_up, A)
    assert (p0**2 / p_up * A * y0 + x) * (p_up / p0 * (A - 1) * y0 + y) == pytest.approx(p0 * A**2 * y0**2, rel=1e-15)


@given(st.floats(min_value=0.00001, max_value=10000.0),
         st.floats(min_value=1e-6, max_value=1 - 1e-6),
         st.floats(min_value=1e-6, max_value=1 - 1e-6),
        st.floats(min_value=100, max_value=100000),
        st.floats(min_value=100, max_value=100000),
        st.floats(min_value=100, max_value=100000)
         )
def test_execute_llamma_sell(p0, p_ratio_1, p_ratio_2, x, y, dx):
    p_up = p0 / p_ratio_1
    p_down = p0 * p_ratio_2
    state = LlammaState(x, y, p_up, p_down, p0)
    new_state = execute_llamma_sell(state, dx)
    assert state.x + dx == new_state.x
    assert state.y + xyk_out_given_in(state.x, state.y, dx) == new_state.y
