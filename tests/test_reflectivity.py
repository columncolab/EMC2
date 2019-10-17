import emc2
import xarray as xr
import numpy as np

def test_convective_reflectivity():
    instrument = emc2.core.instruments.KAZR('nsa')
    heights = xr.DataArray(np.linspace(0, 11000., 1000))
    temp = 15.04 - 0.00649*heights + 273.15
    q = np.linspace(0, 0.5, len(temp))
    p = 101.29 * (temp / 288.08) ** 5.256
    my_ds = xr.Dataset({'q': q, 't': temp, 'p_3d': p})
    my_ds = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_ds, "cl")
    assert np.all(my_ds["Ze"].values < -10)
    assert my_ds["Ze"].values.max() > -20.
    my_ds = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_ds, "pl")
    assert np.all(my_ds["Ze"].values < 50)
    assert my_ds["Ze"].values.max() > 25.
    my_ds = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_ds, "ci")
    assert np.all(my_ds["Ze"].values < 10)
    assert my_ds["Ze"].values.max() > -10.
    my_ds = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_ds, "pi")
    assert np.all(my_ds["Ze"].values < 10)
    assert my_ds["Ze"].values.max() > -10