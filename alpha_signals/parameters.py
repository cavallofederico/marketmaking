from types import SimpleNamespace
import pandas as pd


class Parameters(SimpleNamespace):
    @property
    def dt(self):
        return ((self.k * self.A / self.dalpha + (self.xi**2) / (2 * self.dalpha ** 2) + self.lambda_plus + self.lambda_minus)**(-1)) * self.dt_scaling

    @property
    def dt_scaling(self):
        """Scaling dt to reduce size of h."""
        return 1


base_simulation_parameters_dict = {
    'q_max': 4,
    'T': 60,
    'A': 300,
    'dalpha': 5,
    'Delta': 0.005,
    'epsilon': 0.005,
    'psi': 0.01,
    'phi_': 1e-6,
    'eta_plus': 60.0,
    'eta_minus': 60.0,
    'sigma': 0.01,
    'k': 200.0,
    'xi': 1.0,
    'lambda_plus': 1.0,
    'lambda_minus': 1.0,
    'theta': 0.1,
    's0': 100,
    'n': 200,
    'drift': True
}
base_p = Parameters(**base_simulation_parameters_dict)

def load_parameters_from_excel(ticker='DEFAULT', path='/Users/federico/Library/CloudStorage/GoogleDrive-fc.cavallo@gmail.com/My Drive/1 Projects/2021-22-23 Tesis Quant UdeSA/', file_name='Parameters_NASDAQ.xlsx'):
    df = pd.read_excel(path + file_name)
    df.set_index("TICKER")
    df[df['TICKER'] == ticker].iloc[0].to_dict()

    p = Parameters(**base_simulation_parameters_dict)
    p.__dict__.update(df[df['TICKER'] == ticker].iloc[0].to_dict())
    return p


def load_simulation_parameters_from_excel(path='/Users/federico/Library/CloudStorage/GoogleDrive-fc.cavallo@gmail.com/My Drive/1 Projects/2021-22-23 Tesis Quant UdeSA/', file_name='Simulations_1.xlsx'):
    pd.read_excel(path + file_name)
    df = pd.read_excel(path + file_name).to_dict('records')
    return df