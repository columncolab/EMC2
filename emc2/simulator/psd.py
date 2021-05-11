import xarray as xr
import numpy as np

from scipy.special import gamma


def calc_mu_lambda(model, hyd_type="cl",
                   calc_dispersion=False, dispersion_mu_bounds=(2, 15),
                   subcolumns=False, is_conv=False, **kwargs):

    """
    Calculates the :math:`\mu` and :math:`\lambda` of the gamma PSD given :math:`N_{0}`.
    The gamma size distribution takes the form:

    .. math::
         N(D) = N_{0}e^{-\lambda D}D^{\mu}

    Where :math:`N_{0}` is the intercept, :math:`\lambda` is the slope, and
    :math:`\mu` is the dispersion.

    Note: this method only accepts the microphysical cloud fraction in order to maintain
    consistency because the PSD calculation is necessarily related only to the MG2 scheme
    without assumption related to the radiation logic

    Parameters
    ----------
    model: :py:mod:`emc2.core.Model`
        The model to generate the parameters for.
    hyd_type: str
        The assumed hydrometeor type. Must be a hydrometeor type in Model.
    calc_dispersion: bool
        If False, the :math:`\mu` parameter will be fixed at 1/0.09 per
        Geoffroy et al. (2010). If True and the hydrometeor type is "cl",
        then the Martin et al. (1994) method will be used to calculate
        :math:`\mu`. Otherwise, :math:`\mu` is set to  0.
    dispersion_mu_bounds: 2-tuple
        The lower and upper bounds for the :math:`\mu` parameter.
    subcolumns: bool
        If True, the fit parameters will be generated for the generated subcolumns
        rather than the model data itself.
    is_conv: bool
        If True, calculate from convective properties. IF false, do stratiform.
    LES_mode: bool
        If True, then assume each point is a subcolumn.

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
        if not is_conv:
            N_name = "strat_n_subcolumns_%s" % hyd_type
            q_name = "strat_q_subcolumns_%s" % hyd_type
            if not LES_mode:
                frac_name = model.strat_frac_names[hyd_type]
            else:
                frac_name = "strat_frac_subcolumns_%s" % hyd_type
        else:
            N_name = "conv_n_subcolumns_%s" % hyd_type
            q_name = "conv_q_subcolumns_%s" % hyd_type
            if not LES_mode:
                frac_name = model.conv_frac_names[hyd_type]
            else:
                frac_name = "conv_frac_subcolumns_%s" % hyd_type

        if not LES_mode:
            frac_array = np.tile(
                model.ds[frac_name].values, (model.num_subcolumns, 1, 1))
        else:
            frac_array = model.ds[frac_name].values
        frac_array = np.where(frac_array == 0, 1, frac_array)
    Rho_hyd = model.Rho_hyd[hyd_type].magnitude
    column_ds = model.ds

    if hyd_type == "cl":
        if calc_dispersion is True:
            if not subcolumns:
                mus = 0.0005714 * (column_ds[N_name].values) + 0.2714
            else:
                mus = 0.0005714 * (column_ds[N_name].values * frac_array) + 0.2714
            mus = 1 / mus**2 - 1
            mus = np.where(mus < dispersion_mu_bounds[0], dispersion_mu_bounds[0], mus)
            mus = np.where(mus > dispersion_mu_bounds[1], dispersion_mu_bounds[1], mus)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims).astype('float64')
        else:
            mus = 1 / 0.09 * np.ones_like(column_ds[N_name].values)
            column_ds["mu"] = xr.DataArray(mus, dims=column_ds[q_name].dims).astype('float64')
    else:
        column_ds["mu"] = xr.DataArray(
            np.zeros_like(column_ds[q_name].values), dims=column_ds[q_name].dims).astype('float64')

    column_ds["mu"].attrs["long_name"] = "Gamma fit dispersion"
    column_ds["mu"].attrs["units"] = "1"

    d = 3.0
    c = np.pi * Rho_hyd / 6.0
    if not subcolumns:
        fit_lambda = ((c * column_ds[N_name].astype('float64') * 1e6 * gamma(column_ds["mu"] + d + 1.)) /
                      (column_ds[q_name].astype('float64') * gamma(column_ds["mu"] + 1.)))**(1 / d)
    else:
        fit_lambda = ((c * column_ds[N_name].astype('float64') * frac_array *
                       1e6 * gamma(column_ds["mu"] + d + 1.)) /
                      (column_ds[q_name].astype('float64') * gamma(column_ds["mu"] + 1.))) ** (1 / d)

    # Eventually need to make this unit aware, pint as a dependency?
    column_ds["lambda"] = fit_lambda.where(column_ds[q_name] > 0).astype(np.longdouble)
    column_ds["lambda"].attrs["long_name"] = "Slope of gamma distribution fit"
    column_ds["lambda"].attrs["units"] = "m-1"
    column_ds["N_0"] = column_ds[N_name].astype(np.longdouble) * 1e6 * \
        column_ds["lambda"]**(column_ds["mu"] + 1.) / gamma(column_ds["mu"] + 1.)
    column_ds["N_0"].attrs["long_name"] = "Intercept of gamma fit"
    column_ds["N_0"].attrs["units"] = "m-4"
    model.ds = column_ds
    return model
