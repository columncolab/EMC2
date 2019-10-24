import emc2
import xarray as xr
import numpy as np


def test_gaseous_attenuation():
    # Create an X array object with a standard atmosphere that is saturated
    # Gaseous attenuation should be small (under 0.1 dB/km)
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.attenuation.calc_radar_atm_attenuation(instrument, my_model)
    my_ds = my_model.ds
    assert np.all(my_ds['kappa_o2'].values < 0.1)
    assert np.all(my_ds['kappa_wv'].values < 0.1)
