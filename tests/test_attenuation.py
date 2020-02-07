import emc2
import xarray as xr
import numpy as np

from emc2.core.instrument import ureg


def test_gaseous_attenuation():
    # Create an X array object with a standard atmosphere that is saturated
    # Gaseous attenuation should be small (under 0.1 dB/km)
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.attenuation.calc_radar_atm_attenuation(instrument, my_model)
    my_ds = my_model.ds
    assert np.nanmax(my_ds['kappa_o2'].values) < 0.1
    assert np.nanmax(my_ds['kappa_wv'].values) < 0.5


def test_theory_beta_m():
    # Create an X array object with a standard atmosphere that is saturated
    # Index of refraction is always > 1 and less than 2
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.attenuation.calc_theory_beta_m(my_model, 0.520)
    my_ds = my_model.ds
    green_sigma = my_ds["sigma"].values
    assert np.all(my_ds["n_s"].values > 1)
    assert np.all(my_ds["n_s"].values < 2)

    # Blue light should attenuate more than green light
    my_model = emc2.simulator.attenuation.calc_theory_beta_m(my_model, 0.430)
    my_ds = my_model.ds
    blue_sigma = my_ds["sigma"].values
    assert np.all(np.greater(blue_sigma, green_sigma))

    # We usually have around 1e25 molecules/m3 in a volume of air
    assert np.all(my_ds["N_s"].values > 1e24)
