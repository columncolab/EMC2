import emc2
import numpy as np


def test_mie_file():
    KAZR = emc2.core.instruments.KAZR('nsa')
    assert "cl" in KAZR.mie_table.keys()
    assert "alpha_p" in KAZR.mie_table["cl"].variables.keys()
    assert KAZR.mie_table["cl"]["alpha_p"][0] == 1.463017e-16
    assert np.all(KAZR.mie_table["cl"]["wavelength"] == 8.6e3)


def test_to_netcdf():
    model = emc2.core.model.TestAllStratiform()
    KAZR = emc2.core.instruments.KAZR('nsa')
    model = emc2.simulator.main.make_simulated_data(model, KAZR, 8)
    model.subcolumns_to_netcdf('test.nc')
    model.ds.close()

    model = emc2.core.model.TestModel()
    model.load_subcolumns_from_netcdf('test.nc')
    assert "sub_col_Ze_cl_strat" in [x for x in model.ds.variables.keys()]
    assert "conv_frac_subcolumns_pi" in [x for x in model.ds.variables.keys()]
    model.ds.close()
