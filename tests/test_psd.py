import emc2
import xarray as xr
import numpy as np


def test_lambda_mu():
    # We have a cloud with a constant N, increasing LWC
    # Therefore, if dispersion is fixed, slope should decrease with LWC
    # N_0 will also increases since it is directly proportional to lambda

    q = np.linspace(0, 1, 100.)
    N = 100 * np.ones_like(q)
    my_ds = xr.Dataset({'q': q, 'N': N})
    my_ds = emc2.simulator.psd.calc_mu_lambda(my_ds, hyd_type="cl", calc_dispersion=False)
    assert np.all(my_ds["mu"] == 1 / 0.09)
    diffs = np.diff(my_ds["lambda"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
    diffs = np.diff(my_ds["N_0"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
    my_ds = emc2.simulator.psd.calc_mu_lambda(my_ds, hyd_type="cl", calc_dispersion=True)

    # Make sure calculated mu is within bounds
    assert np.all(my_ds["mu"] >= 2)
    assert np.all(my_ds["mu"] <= 15)
    assert ~np.all(my_ds["mu"].values == 2)
    assert ~np.all(my_ds["mu"].values == 15)
    diffs = np.diff(my_ds["lambda"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
    diffs = np.diff(my_ds["N_0"])
    diffs = diffs[np.isfinite(diffs)]
    assert np.all(diffs < 0)
