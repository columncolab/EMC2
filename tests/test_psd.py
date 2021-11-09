import emc2
import xarray as xr
import numpy as np


def test_lambda_mu():
    # We have a cloud with a constant N, increasing LWC
    # Therefore, if dispersion is fixed, slope should decrease with LWC
    # N_0 will also increases since it is directly proportional to lambda

    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.psd.calc_mu_lambda(my_model, hyd_type="cl", calc_dispersion=False)
    my_ds = my_model.ds
    assert np.all(my_ds["mu"] == 1 / 0.09)
    diffs = np.diff(my_ds["lambda"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
    diffs = np.diff(my_ds["N_0"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)


def test_re():
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.psd.calc_re_thompson(my_model, hyd_type="cl")
    my_ds = my_model.ds
    assert np.all(my_ds["re_cl"] < 100.)
    print(my_ds["re_cl"])
