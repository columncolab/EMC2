import emc2
import xarray as xr
import numpy as np


def test_gaseous_attenuation():
    # Create an X array object with a standard atmosphere that is saturated
    # Gaseous attenuation should be small (under 0.1 dB/km)
    instrument = emc2.core.instruments.KAZR('nsa')
    heights = xr.DataArray(np.linspace(0, 11000., 1000))
    temp = 15.04 - 0.00649 * heights + 273.15
    p = 101.29 * (temp / 288.08)**5.256
    es = 0.6112 * np.exp(17.67 * temp / (temp + 243.5))
    qv = 0.622 * es / (p - es)
    my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp})
    my_ds = emc2.simulator.attenuation.calc_radar_atm_attenuation(instrument, my_ds)
    assert np.all(my_ds['kappa_o2'].values < 0.1)
    assert np.all(my_ds['kappa_wv'].values < 0.1)
