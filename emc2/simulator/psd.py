import xarray as xr
import numpy as np

from ..core import hydrometeor_info
from scipy.special import gamma


def calc_mu_lambda(column_ds, q_name="q", N_name="N", hyd_type="cl",
                   calc_dispersion=True, dispersion_mu_bounds=(2, 15)):

    """
    Calculates the :math:`\mu` and :math:`\lambda` of the gamma PSD given :math:`N_{0}`.
    The gamma size distribution takes the form:

        :math: 'N(D) = N_{0}e^{-\lambdaD}D^{\mu}

    Where :math:`N_{0}` is the intercept, :math:`\lambda` is the slope, and
    :math:`\mu` is the dispersion.


    Parameters
    ----------
    column_ds: xarray Dataset
        The xarray dataset containing the column parameters.
    q_name: str
        The name of the variable containing the liquid water mixing ratio.
    N_name: str
        The name of the variable containing the cloud particle number concentration.
    hyd_type: str
        The assumed hydrometeor type. Must be one of:
        'cl' (cloud liquid), 'ci' (cloud ice),
        'pl' (liquid precipitation), 'pi' (ice precipitation).
    calc_dispersion: bool
        If False, the :math:`\mu` parameter will be fixed at 1/0.09. If True and
        the hydrometeor type is "cl", then the Martin et al. (1994) method
        will be used to calculate :math:`\mu`.
    dispersion_mu_bounds: 2-tuple
        The lower and upper bounds for the :math:`\mu` parameter.

    Returns
    -------
    column_ds: xarray Dataset
        The dataset with the :math:`\lambda` and :math:`\mu` parameters added.

    References
    ----------
    Ulbrich, C. W., 1983: Natural variations in the analytical form of the raindrop size
    distribution: J. Climate Appl. Meteor., 22, 1764-1775

    Martin, G.M., D.W. Johnson, and A. Spice, 1994: The Measurement and Parameterization
    of Effective Radius of Droplets in Warm Stratocumulus Clouds.
    J. Atmos. Sci., 51, 1823–1842, https://doi.org/10.1175/1520-0469(1994)051<1823:TMAPOE>2.0.CO;2
    """

    if hyd_type == "cl":
        if calc_dispersion:
            mus = 0.0005714 * (column_ds[N_name].values / 1e6 * hydrometeor_info.Rho_hyd["cl"]) + 0.2714
            mus = 1 / mus**2 - 1
            mus = np.where(mus < dispersion_mu_bounds[0], dispersion_mu_bounds[0], mus)
            mus = np.where(mus > dispersion_mu_bounds[1], dispersion_mu_bounds[1], mus)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims)
        else:
            mus = 1 / 0.09 * np.ones_like(column_ds[N_name].values)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims)
    else:
        column_ds["mu"] = xr.DataArray(
            np.zeros_like(column_ds[q_name], dims=column_ds[q_name].dims))

    column_ds["mu"].attrs["long_name"] = "Gamma fit dispersion"
    column_ds["mu"].attrs["units"] = "1"

    d = 3.0
    c = np.pi * hydrometeor_info.Rho_hyd[hyd_type] / 6.0
    fit_lambda = (c * column_ds[N_name] * gamma(column_ds["mu"] + d + 1) /
                  (column_ds[q_name] * gamma(column_ds["mu"] + 1)))**(1 / d)

    # Eventually need to make this unit aware, pint as a dependency?
    column_ds["lambda"] = fit_lambda.where(column_ds[q_name] > 0)
    column_ds["lambda"].attrs["long_name"] = "Slope of gamma distribution fit"
    column_ds["lambda"].attrs["units"] = "cm-1"
    column_ds["N_0"] = column_ds[N_name] * column_ds["lambda"]**(column_ds["mu"] + 1.) \
        / gamma(column_ds["mu"] + 1.)
    column_ds["N_0"].attrs["long_name"] = "Intercept of gamma fit"
    column_ds["N_0"].attrs["units"] = "cm-4"

    return column_ds
