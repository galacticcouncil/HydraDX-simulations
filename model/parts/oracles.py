import numpy as np


def constant_function(x, constant=0):
    """
    Returns a constant value `a` such that f(x) = a.

    Parameters
    ============
    x: the input value.
    constant: the constant value.
    """
    return constant


def step_function(x, steps=[(-np.inf, 0), (0, 1), (np.inf, 0)]):
    """
    Returns the value from a step function.
    
    The function returns the step according to the list of values such that
    f(x) = y_0 if x_0 <= x < x_1,
           y_1 if x_1 <= x < x_2,
           y_2 if x_2 <= x < x_3,
           ...

    Typically it's a good idea to use very large values for the boundary pairs.

    Parameters
    ============
    x: the input value.
    steps: a list of (x, y) pairs that define the steps.
    """
    for n in range(1, len(steps)):
        if steps[n-1][0] <= x and x < steps[n][0]:
            return steps[n-1][1]
    else:
        return 0
