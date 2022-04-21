import xarray as xr
import numpy as np

from scipy.special import gamma
from ..core.instrument import ureg, quantity


def calc_mu_lambda(model, hyd_type="cl",
                   calc_dispersion=None, dispersion_mu_bounds=(2, 15),
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
    calc_dispersion: bool or None
        If False, the :math:`\mu` parameter will be fixed at 1/0.09 per
        Geoffroy et al. (2010). If True and the hydrometeor type is "cl",
        then the Martin et al. (1994) method will be used to calculate
        :math:`\mu`. Otherwise, :math:`\mu` is set to  0.
        If None (default), setting calculation parameterization based on model logic.
    dispersion_mu_bounds: 2-tuple
        The lower and upper bounds for the :math:`\mu` parameter.
    subcolumns: bool
        If True, the fit parameters will be generated for the generated subcolumns
        rather than the model data itself.
    is_conv: bool
        If True, calculate from convective properties. IF false, do stratiform.


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

    if calc_dispersion is None:
        if model.model_name in ["E3SM", "CESM2", "WRF"]:
            calc_dispersion = True
        else:
            calc_dispersion = False
    if not subcolumns:
        N_name = model.N_field[hyd_type]
        if not is_conv:
            q_name = model.q_names_stratiform[hyd_type]
            frac_name = model.strat_frac_names[hyd_type]
        else:
            q_name = model.q_names_convective[hyd_type]
            frac_name = model.conv_frac_names[hyd_type]
    else:
        if not is_conv:
            N_name = "strat_n_subcolumns_%s" % hyd_type
            q_name = "strat_q_subcolumns_%s" % hyd_type
            frac_name = model.strat_frac_names[hyd_type]
        else:
            N_name = "conv_n_subcolumns_%s" % hyd_type
            q_name = "conv_q_subcolumns_%s" % hyd_type
            frac_name = model.conv_frac_names[hyd_type]

    frac_array = np.tile(
        model.ds[frac_name].values, (model.num_subcolumns, 1, 1))
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
    column_ds["lambda"] = fit_lambda.where(column_ds[q_name] > 0).astype(float)
    column_ds["lambda"].attrs["long_name"] = "Slope of gamma distribution fit"
    column_ds["lambda"].attrs["units"] = r"$m^{-1}$"
    column_ds["N_0"] = column_ds[N_name].astype(float) * 1e6 * \
        column_ds["lambda"]**(column_ds["mu"] + 1.) / gamma(column_ds["mu"] + 1.)
    column_ds["N_0"].attrs["long_name"] = "Intercept of gamma fit"
    column_ds["N_0"].attrs["units"] = r"$m^{-4}$"
    model.ds = column_ds
    return model


def calc_re_thompson(model, hyd_type,
                     is_conv=True, subcolumns=False, **kwargs):
    """
    Calculate the effective radius using the Thompson et al. (2004)
    microphysics scheme.

    Parameters
    ----------
    model: emc2.core.Model
        The input model to calculate the effective radius fields for
    hyd_type: str
        The hydrometeor type to calculate the effective radius for
    is_conv: bool
        Whether or not we are calculating the convective properties
    subcolumns: bool
        If true, calculate the effective radius from the generated subcolumns.
        Else, generate it from the original model data.

    Returns
    -------
    model: emc2.core.Model
        The model structure with the calculated effective radius values.

    Reference
    ---------
    Thompson, G., Rasmussen, R. M., & Manning, K. (2004). Explicit forecasts
    of winter precipitation using an improved bulk microphysics scheme.
    Part I: Description and sensitivity analysis. Monthly Weather Review,
    132(2), 519–542.
    https://doi.org/10.1175/1520-0493(2004)132%3C0519:EFOWPU%3E2.0.
    """

    if not subcolumns:
        if not is_conv:
            N_name = model.N_field[hyd_type]
            q_name = model.q_names_stratiform[hyd_type]
            re_name = model.strat_re_fields[hyd_type]
        else:
            N_name = model.N_field[hyd_type]
            q_name = model.q_names_convective[hyd_type]
            re_name = model.conv_re_fields[hyd_type]

    else:
        if not is_conv:
            N_name = "strat_n_subcolumns_%s" % hyd_type
            q_name = "strat_q_subcolumns_%s" % hyd_type
            re_name = model.strat_re_fields[hyd_type]
        else:
            # We'll have to assume that stratiform and convective have 
            # Same N, probably not a great assumption
            N_name = "strat_n_subcolumns_%s" % hyd_type
            q_name = "conv_q_subcolumns_%s" % hyd_type
            re_name = model.conv_re_fields[hyd_type]

   
    q_w = model.ds[q_name].values
    N_w = model.ds[N_name].values
    rho_w = model.Rho_hyd[hyd_type].magnitude

    p = model.ds[model.p_field].values * getattr(
        ureg, model.ds[model.p_field].attrs["units"])
    p = p.to(ureg.Pa).magnitude
    t = model.ds[model.T_field].values * getattr(
        ureg, model.ds[model.T_field].attrs["units"])
    t = t.to(ureg.kelvin).magnitude

    rho_a = p / (model.consts["R_d"] * t)
    if hyd_type == 'pl':
        k = 2.4
    else:
        k = 3.
    r_w = 0.5 * ((6 * rho_a * q_w) / (np.pi * rho_w * N_w)) ** (1 / k)
    r_w = xr.DataArray(r_w * 1e4, dims=model.ds[q_name].dims)

    r_w.attrs['units'] = 'microns'
    r_w.attrs['long_name'] = ('Particle effective radius following ' +
                              'Thompson et al. (2004)')
    model.ds[re_name] = r_w
    return model
