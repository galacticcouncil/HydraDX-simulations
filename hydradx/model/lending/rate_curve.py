def rate_function(u, c, u_1, r_1, r_0=0, d=1):
    """return interest rate given utilization ratio and curve parameters

    u: utilization ratio, between 0 and 1
    c: any real number, controls the speed at which rate increases
    u_1: utilization ratio at which we switch from linear increases
    r_1: interest rate at u_1
    r_0: interest rate at u = 0, default is 0
    d: greater than 1, controls the interest rate at u = 1. d=1 means that interest rate goes to infinity at u = 1
    """

    if u < u_1:
        return r_0 + (r_1 - r_0) / u_1 * u  # line passing through (0, r_0) and (u_1, r_1)
    else:  # form is A + (Bu^2 + Cu) / (D - u)^2
        b = (r_1 - r_0) / 2 * (1 - u_1)**3/u_1**2 - (c + c*u_1) / (2 * u_1)  # this will match slope at u = u_1
        return r_1 - (b * u_1**2 + c * u_1) / (d - u_1)**2 + (b * u**2 + c * u) / (d - u)**2
