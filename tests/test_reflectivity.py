import emc2
import xarray as xr
import numpy as np


def test_convective_reflectivity():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_model, "cl")
    assert np.all(my_model.ds["Ze"].values < 30)
    assert my_model.ds["Ze"].values.max() > -20.
    my_model = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_model, "pl")
    assert my_model.ds["Ze"].values.max() > 25.
    my_model = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_model, "ci")
    assert np.all(my_model.ds["Ze"].values < 10)
    assert my_model.ds["Ze"].values.max() > -10.
    my_model = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(instrument, my_model, "pi")
    assert np.all(my_model.ds["Ze"].values < 10)
    assert my_model.ds["Ze"].values.max() > -10
