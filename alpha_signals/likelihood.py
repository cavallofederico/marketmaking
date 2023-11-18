# class Likelihood():
#     def __init__(self, k, eta_plus, eta_minus, theta):
#         self.k = k
#         self.eta_plus = eta_plus
#         self.eta_minus = eta_minus
#         self.theta = theta


import numpy as np


def likelihood(x, alpha, T=300):
    k = x[0]
    eta_plus = x[1]
    eta_minus = x[2]
    theta = x[3]
    likelihood = (
        -2 * theta * T
        + integral_alpha_plus(k, alpha, eta_plus, T)
        - integral_alpha_minus()
        + sum_log_alpha_plus()
        + sum_log_alpha_minus()
    )


def integral_alpha_plus(k, alpha, eta_plus, T):
    alpha_tau_i_0 = 0
    tau_0_plus = np.array([1, 3, 5])
    tau_0_minus = np.array([2, 4, 6])

    tau_0 = np.concatenate(
        [np.zeros([1, 1]), tau_0_minus, tau_0_plus, np.ones([1, 1]) * T]
    ).sort()
    result = -1 / k * sum(alpha_tau_i_0)
    eta_plus * sum()  # j=1 a n+
    i = 0
    j = 0
    np.exp(
        -k * ((np.logical_or(tau_0[i + 1], tau_0_plus[j])) - tau_0_plus[j])
    ) - np.exp(-k * ((np.logical_or(tau_0[i], tau_0_plus[j])) - tau_0_plus[j]))
