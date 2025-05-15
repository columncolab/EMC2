import emc2
import xarray as xr
import numpy as np


def test_lambda_mu():
    # We have a cloud with a constant N, increasing LWC
    # Therefore, if dispersion is fixed, slope should decrease with LWC
    # N_0 will also increases since it is directly proportional to lambda

    my_model = emc2.core.model.TestAllStratiform()
    my_model.ds["strat_q_subcolumns_cl"] = my_model.ds[my_model.q_names_stratiform['cl']]
    my_model.ds["strat_n_subcolumns_cl"] = my_model.ds[my_model.N_field['cl']]
    my_model = emc2.simulator.psd.calc_mu_lambda(my_model, hyd_type="cl", calc_dispersion=False)
    my_ds = my_model.ds
    assert np.all(my_ds["mu"] == 1 / 0.09)
    diffs = np.diff(my_ds["lambda"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
    diffs = np.diff(my_ds["N_0"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)


def test_calc_and_set_psd_params():
    my_model = emc2.core.model.TestAllStratiform()
    my_model.ds["strat_q_subcolumns_cl"] = my_model.ds[my_model.q_names_stratiform['cl']]
    my_model.ds["strat_n_subcolumns_cl"] = my_model.ds[my_model.N_field['cl']]
    my_model.ds["strat_q_subcolumns_ci"] = my_model.ds["strat_q_subcolumns_cl"]
    my_model.ds["strat_n_subcolumns_ci"] = my_model.ds["strat_n_subcolumns_cl"]

    # Test for liquid hydrometeor type
    hyd_type = "cl"
    fits_ds = emc2.simulator.psd.calc_and_set_psd_params(my_model, hyd_type)

    # Assertions
    assert isinstance(fits_ds, xr.Dataset), "fits_ds should be an xarray.Dataset"
    assert fits_ds["mu"].shape == my_model.ds["strat_n_subcolumns_cl"].shape, "Shape of 'mu' should match"
    assert fits_ds["lambda"].shape == my_model.ds["strat_n_subcolumns_cl"].shape, "Shape of 'lambda' should match"
    assert fits_ds["N_0"].shape == my_model.ds["strat_n_subcolumns_cl"].shape, "Shape of 'N_0' should match"

    # Test for ice hydrometeor type
    hyd_type = "ci"
    fits_ds = emc2.simulator.psd.calc_and_set_psd_params(my_model, hyd_type)

    # Assertions for ice hydrometeor type
    assert isinstance(fits_ds, xr.Dataset), "fits_ds should be an xarray.Dataset"
    assert "mu" in fits_ds, "fits_ds should contain 'mu'"
    assert "lambda" in fits_ds, "fits_ds should contain 'lambda'"
    assert "N_0" in fits_ds, "fits_ds should contain 'N_0'"
    assert np.all(fits_ds["mu"] == 0), "mu for ice should always be equal to 0 in MG2"
