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


def dirac_delta_function(x, steps=[(1, 1)]):
    """
    Returns a Dirac delta function such that

    f(x) = y_0 if x = x_0,
           y_1 if x = x_1,
           ...
           else 0

    Parameters
    ============
    x: the input value.
    steps: a list of deltas.
    """
    for x_n, y_n in steps:
        if x == x_n:
            return y_n
    else:
        return 0


def random_stochastic_function(x, delta):
    """
    Creates a random stochastic function that adds a value between
    [-delta, delta]

    Parameters
    ============
    x: the input value.
    delta: defines the range
    """
    return (np.random.random_sample() * 2 * delta) - delta


def random_gaussian_function(x, sigma):
    """
    Samples from a Gaussian distribution.

    Parameters
    ============
    x: the input value.
    delta: defines the variance of the Gaussian curve.
    """
    return np.random.normal(0, sigma)
