from types import SimpleNamespace


class Parameters(SimpleNamespace):
    @property
    def dt(self):
        return (self.k * self.A / self.dalpha + (self.xi**2) / (2 * self.dalpha ** 2) + self.lambda_plus + self.lambda_minus)**(-1)

base_simulation_parameters_dict = {
    'q_max': 4,
    'T': 60,
    'A': 300,
    'dalpha': 10,
    'Delta': 0.005,
    'epsilon': 0.005,
    'psi': 0.01,
    'phi_': 1e-6,
    'eta': 60.0,
    'sigma': 0.01,
    'k': 200.0,
    'xi': 1.0,
    'lambda_plus': 1.0,
    'lambda_minus': 1.0,
    'theta': 0.1,
    's0': 100,
    'n': 200
}
base_p = Parameters(**base_simulation_parameters_dict)