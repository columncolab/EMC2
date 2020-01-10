import xarray as xr
import numpy as np

from scipy.special import gamma


def calc_mu_lambda(model, hyd_type="cl",
                   calc_dispersion=True, dispersion_mu_bounds=(2, 15),
                   subcolumns=False):

    """
    Calculates the :math:`\mu` and :math:`\lambda` of the gamma PSD given :math:`N_{0}`.
    The gamma size distribution takes the form:

    .. math::
         N(D) = N_{0}e^{-\lambda D}D^{\mu}

    Where :math:`N_{0}` is the intercept, :math:`\lambda` is the slope, and
    :math:`\mu` is the dispersion.

    Parameters
    ----------
    model: :py:mod:`emc2.core.Model`
        The model to generate the parameters for.
    hyd_type: str
        The assumed hydrometeor type. Must be a hydrometeor type in Model.
    calc_dispersion: bool
        If False, the :math:`\mu` parameter will be fixed at 1/0.09. If True and
        the hydrometeor type is "cl", then the Martin et al. (1994) method
        will be used to calculate :math:`\mu`.
    dispersion_mu_bounds: 2-tuple
        The lower and upper bounds for the :math:`\mu` parameter.
    subcolumns: bool
        If True, the fit parameters will be generated for the generated subcolumns
        rather than the model data itself.
    Returns
    -------
    model: :py:mod:`emc2.core.Model`
        The Model with the :math:`\lambda` and :math:`\mu` parameters added.

    References
    ----------
    Ulbrich, C. W., 1983: Natural variations in the analytical form of the raindrop size
    distribution: J. Climate Appl. Meteor., 22, 1764-1775

    Martin, G.M., D.W. Johnson, and A. Spice, 1994: The Measurement and Parameterization
    of Effective Radius of Droplets in Warm Stratocumulus Clouds.
    J. Atmos. Sci., 51, 1823–1842, https://doi.org/10.1175/1520-0469(1994)051<1823:TMAPOE>2.0.CO;2
    """

    if not subcolumns:
        N_name = model.N_field[hyd_type]
        q_name = model.q_names_stratiform[hyd_type]
    else:
        N_name = "strat_n_subcolumns_%s" % hyd_type
        q_name = "strat_q_subcolumns_%s" % hyd_type

    Rho_hyd = model.Rho_hyd[hyd_type].magnitude
    column_ds = model.ds

    if hyd_type == "cl":
        if calc_dispersion:
            mus = 0.0005714 * (column_ds[N_name].values / 1e6 * model.Rho_hyd["cl"].magnitude) + 0.2714
            mus = 1 / mus**2 - 1
            mus = np.where(mus < dispersion_mu_bounds[0], dispersion_mu_bounds[0], mus)
            mus = np.where(mus > dispersion_mu_bounds[1], dispersion_mu_bounds[1], mus)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims)
        else:
            mus = 1 / 0.09 * np.ones_like(column_ds[N_name].values)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims)
    else:
        column_ds["mu"] = xr.DataArray(
            np.zeros_like(column_ds[q_name].values), dims=column_ds[q_name].dims)

    column_ds["mu"].attrs["long_name"] = "Gamma fit dispersion"
    column_ds["mu"].attrs["units"] = "1"

    d = 3.0
    c = np.pi * Rho_hyd / 6.0
    fit_lambda = ((c * column_ds[N_name] * gamma(column_ds["mu"] + d + 1)) /
                  (column_ds[q_name] * gamma(column_ds["mu"] + 1)))**(1 / d)

    # Eventually need to make this unit aware, pint as a dependency?
    column_ds["lambda"] = fit_lambda.where(column_ds[q_name] > 0).astype('float64')
    column_ds["lambda"].attrs["long_name"] = "Slope of gamma distribution fit"
    column_ds["lambda"].attrs["units"] = "cm-1"
    column_ds["N_0"] = column_ds[N_name] * column_ds["lambda"]**(column_ds["mu"] + 1.) \
        / gamma(column_ds["mu"] + 1.)
    column_ds["N_0"].attrs["long_name"] = "Intercept of gamma fit"
    column_ds["N_0"].attrs["units"] = "cm-4"
    model.ds = column_ds
    return model
