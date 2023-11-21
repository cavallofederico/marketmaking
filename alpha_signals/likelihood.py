import numpy as np


T = 10
tau_0_plus = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]]) * T
tau_0_minus = np.array([[0.2, 0.4, 0.6]]) * T


def likelihood(x):
    k = x[0]
    eta_plus = x[1]
    eta_minus = x[2]
    theta = x[3]
    likelihood = (
        -2 * theta * T
        - integral_alpha_s(k, eta_minus, eta_plus)
        + sum_log_alpha_plus()
        + sum_log_alpha_minus()
    )
    return likelihood


def likelihood_to_minimize(x):
    return -likelihood(x)


def integral_alpha_s(k, eta_minus, eta_plus):
    tau_0 = np.concatenate(
        [
            np.zeros([1, 1]),
            tau_0_minus,
            tau_0_plus,
            np.ones([1, 1]) * T
        ], axis=1
    )

    eta_minus_vector = -np.ones([tau_0_minus.shape[1], 1]) * eta_minus
    eta_plus_vector = np.ones([tau_0_plus.shape[1], 1]) * eta_plus
    eta_vector = np.concatenate(
        [
            np.zeros([1, 1]),
            eta_minus_vector,
            eta_plus_vector,
            np.zeros([1, 1])
        ], axis=0
    ).T
    tau_eta = np.concatenate([tau_0, eta_vector])
    tau_eta = tau_eta[:, tau_eta[0, :].argsort()]
    tau_0 = tau_eta[:, tau_eta[0, :].argsort()][0:1, :]
    eta_0 = tau_eta[:, tau_eta[0, :].argsort()][1:2, :]

    tau_matrix = tau_0 * np.ones([1, tau_0.shape[1]]).T
    eta_matrix = eta_0 * np.ones([1, eta_0.shape[1]]).T
    tau_matrix_1 = np.roll(
        tau_matrix, -1
    )  # numero de fila es j, numero de columna es i

    tau_matrix_diff = tau_matrix - tau_matrix.T
    tau_matrix_diff = np.where(tau_matrix_diff > 0, tau_matrix_diff, 0)
    tau_matrix_diff_1 = tau_matrix_1 - tau_matrix.T
    tau_matrix_diff_1 = np.where(tau_matrix_diff_1 > 0, tau_matrix_diff_1, 0)

    alpha_tau_matrix = eta_matrix * (
        np.exp(-k * tau_matrix_diff_1) - np.exp(-k * tau_matrix_diff)
    )
    alpha_tau = np.sum(alpha_tau_matrix, axis=0)
    alpha_s_plus = np.sum(-np.where(alpha_tau >= 0, alpha_tau, 0) / k)  # The signs seem to be wrong in the paper
    alpha_s_minus = np.sum(np.where(alpha_tau <= 0, alpha_tau, 0) / k)  # The signs seem to be wrong in the paper

    integral_alpha_s = alpha_s_plus - alpha_s_minus

    return integral_alpha_s


def sum_log_alpha_plus():
    return 0


def sum_log_alpha_minus():
    return 0
