import numpy as np


class MaximumLikelihood:
    def __init__(self, T, tau_0_plus, tau_0_minus, t_plus, t_minus) -> None:
        self._T = T
        self._tau_0_plus = tau_0_plus
        self._tau_0_minus = tau_0_minus
        self._t_plus = t_plus
        self._t_minus = t_minus

    def likelihood_to_minimize(self, x):
        return -self.likelihood(x)

    def likelihood(self, x):
        k = x[0]
        eta_plus = x[1]
        eta_minus = x[2]
        theta = x[3]
        likelihood = (
            -2 * theta * self._T
            - self.integral_alpha_s(k, eta_minus, eta_plus)
            + self.sum_log_alpha_plus(k, eta_minus, eta_plus, theta)
            + self.sum_log_alpha_minus(k, eta_minus, eta_plus, theta)
        )
        return likelihood

    def integral_alpha_s(self, k, eta_minus, eta_plus):
        tau_0 = np.concatenate(
            [
                np.zeros([1, 1]),
                self._tau_0_minus,
                self._tau_0_plus,
                np.ones([1, 1]) * self._T,
            ],
            axis=1,
        )

        eta_minus_vector = -np.ones([self._tau_0_minus.shape[1], 1]) * eta_minus
        eta_plus_vector = np.ones([self._tau_0_plus.shape[1], 1]) * eta_plus
        eta_vector = np.concatenate(
            [np.zeros([1, 1]), eta_minus_vector, eta_plus_vector, np.zeros([1, 1])],
            axis=0,
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
        # The signs seem to be wrong in the paper, keeping them anyway
        alpha_s_plus = np.sum(-np.where(alpha_tau >= 0, alpha_tau, 0) / k)
        alpha_s_minus = np.sum(np.where(alpha_tau <= 0, -alpha_tau, 0) / k)

        integral_alpha_s = alpha_s_plus - alpha_s_minus

        return integral_alpha_s

    def sum_log_alpha_plus(self, k, eta_minus, eta_plus, theta):
        eta_0, tau_0 = self._get_tau_eta(eta_minus, eta_plus)
        tau_matrix = tau_0 * np.ones([1, self._t_plus.shape[1]]).T
        eta_matrix = eta_0 * np.ones([1, self._t_plus.shape[1]]).T

        t_plus_matrix = self._t_plus.T * np.ones(
            [1, tau_0.shape[1]]
        )  # numero de fila es t, numero de columna es tau

        tau_matrix_diff = t_plus_matrix - tau_matrix
        tau_matrix_diff = np.where(tau_matrix_diff > 0, tau_matrix_diff, 0)

        alpha_tau_matrix = eta_matrix * (np.exp(-k * tau_matrix_diff))
        alpha_tau = np.sum(alpha_tau_matrix, axis=1)
        # The signs seem to be wrong in the paper, keeping them anyway
        result = np.sum(np.log(np.where(alpha_tau >= 0, alpha_tau, 0) + theta))
        return result

    def _get_tau_eta(self, eta_minus, eta_plus):
        tau_0 = np.concatenate(
            [
                self._tau_0_minus,
                self._tau_0_plus,
            ],
            axis=1,
        )
        eta_minus_vector = -np.ones([self._tau_0_minus.shape[1], 1]) * eta_minus
        eta_plus_vector = np.ones([self._tau_0_plus.shape[1], 1]) * eta_plus
        eta_vector = np.concatenate([eta_minus_vector, eta_plus_vector], axis=0).T
        tau_eta = np.concatenate([tau_0, eta_vector])
        tau_eta = tau_eta[:, tau_eta[0, :].argsort()]
        tau_0 = tau_eta[:, tau_eta[0, :].argsort()][0:1, :]
        eta_0 = tau_eta[:, tau_eta[0, :].argsort()][1:2, :]
        return eta_0, tau_0

    def sum_log_alpha_minus(self, k, eta_minus, eta_plus, theta):
        eta_0, tau_0 = self._get_tau_eta(eta_minus, eta_plus)
        tau_matrix = tau_0 * np.ones([1, self._t_minus.shape[1]]).T
        eta_matrix = eta_0 * np.ones([1, self._t_minus.shape[1]]).T
        t_minus_matrix = self._t_minus.T * np.ones(
            [1, tau_0.shape[1]]
        )  # numero de fila es t, numero de columna es tau

        tau_matrix_diff = t_minus_matrix - tau_matrix
        tau_matrix_diff = np.where(tau_matrix_diff > 0, tau_matrix_diff, 0)

        alpha_tau_matrix = eta_matrix * (np.exp(-k * tau_matrix_diff))
        alpha_tau = np.sum(alpha_tau_matrix, axis=1)
        # The signs seem to be wrong in the paper, keeping them anyway
        result = np.sum(np.log(np.where(alpha_tau <= 0, -alpha_tau, 0) + theta))
        return result
