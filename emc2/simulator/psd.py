import xarray as xr
import numpy as np

from scipy.special import gamma
from ..core.instrument import ureg, quantity


def calc_velocity_nssl(dmax, rhoe, hyd_type):
    """
    Calculate the terminal velocity according to the NSSL 2-moment scheme.

    Parameters
    ----------
    dmax: float array
        The particle maximum dimensions in m.
    rhoe: float array
        The particle effective density. 
    hyd_type: str
        The hydrometeor type code (i.e. 'cl', 'gr').
    """

    rhoair_800mb = 1.007
    if hyd_type.lower() == "pl":
        return 10 * (1 - np.exp(-516.575 * dmax))
    if hyd_type.lower() == "gr":
        cd = np.fmax(0.45, np.fmin(1.2,
                0.45 + 0.55 * (800 - np.fmax(170, np.fmin(800, rhoe)))))
        vt = np.sqrt(4.0 * rhoe * 9.81 / (3.0 * cd * rhoair_800mb)) \
            * np.sqrt(dmax)
        return vt
    elif hyd_type.lower() == "ha":
        cd = np.fmax(0.45, np.fmin(1.2,
                0.45 + 0.55 * (800 - np.fmax(500, np.fmin(800, rhoe)))))
        vt = np.sqrt(4.0 * rhoe * 9.81 / (3.0 * cd * rhoair_800mb)) \
            * np.sqrt(dmax)
        return vt
    elif hyd_type.lower() == "sn":
        vt = 21.52823061429272 * dmax ** 0.42
        return vt
    elif hyd_type.lower() == "cl":
        vt = 131.6 * dmax ** 0.824
        return vt
    return np.zeros_like(dmax)


def calc_mu_lambda(model, hyd_type="cl",
                   calc_dispersion=None, dispersion_mu_bounds=(2, 15),
                   subcolumns=False, is_conv=False, **kwargs):

    """
    This method calculated the Gamma PSD parameters following Morrison and Gettelman (2008).
    Note that the dispersion cacluation from MG2008 is used in all models implementing this
    parameterization except for ModelE and DHARMA, which use a fixed definition.
    Note #2: `subcolumns` are hardwired as `True` in `radar_moments.py` and `lidar_moments.py`,
    which means that we assume that the PSD mu and lambda definition applies to the SGS. This
    defintion might be under future discussion (whether to apply this assumption or not).
    Calculates the :math:`\mu` and :math:`\lambda` of the gamma PSD given :math:`N_{0}`.
    The gamma size distribution takes the form:

    .. math::
         N(D) = N_{0}e^{-\lambda D}D^{\mu}

    Where :math:`N_{0}` is the intercept, :math:`\lambda` is the slope, and
    :math:`\mu` is the dispersion.

    Note: this method only accepts the microphysical cloud fraction in order to maintain
    consistency because the PSD calculation is necessarily related only to the MG2 scheme
    without assumption related to the radiation logic.

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
        If True, the fit parameters will be generated using the generated subcolumns
        (in-cloud) q and N quantities) rather than using the "raw" (grid-cell mean) output.
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
        if model.model_name in ["ModelE", "DHARMA"]:
            calc_dispersion = False
        else:
            calc_dispersion = True
    if not subcolumns:
        N_name = model.N_field[hyd_type]
        if not is_conv:
            q_name = model.q_names_stratiform[hyd_type]
        else:
            q_name = model.q_names_convective[hyd_type]
    else:
        if not is_conv:
            N_name = "strat_n_subcolumns_%s" % hyd_type
            q_name = "strat_q_subcolumns_%s" % hyd_type
        else:
            N_name = "conv_n_subcolumns_%s" % hyd_type
            q_name = "conv_q_subcolumns_%s" % hyd_type

    
    if model.Rho_hyd[hyd_type] == 'variable':    
        Rho_hyd = model.ds[model.variable_density[hyd_type]].values
    else:
        Rho_hyd = model.Rho_hyd[hyd_type].magnitude

    column_ds = model.ds

    if hyd_type == "cl":
        if calc_dispersion is True:
            if not subcolumns:
                mus = 0.0005714 * (column_ds[N_name].values * 1e-6) + 0.2714  # converting to cm-3 per Martin, 1994
            else:
                mus = 0.0005714 * (column_ds[N_name].values * 1e-6) + 0.2714  # converting to cm-3
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
        fit_lambda = ((c * column_ds[N_name].astype('float64') *
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


def calc_and_set_psd_params(model, hyd_type, subcolumns=True, **kwargs):
    """
    Calculate and set particle size distribution (PSD) parameters for a given hydrometeor type,
    microphysics scheme, and model ouput dataset. Supports both liquid and ice hydrometeor classes.

    Parameters
    ==========
    model: object
        The model object containing microphysics scheme information and dataset attributes.
    hyd_type: str
        The hydrometeor type, e.g., "cl" or "pl" for liquid classes, and other values for ice classes.
    subcolumns: bool, optional
        Whether to use subcolumns for PSD calculations. Defaults to True.
    **kwargs: dict
        Additional keyword arguments passed to the PSD calculation functions.

    Returns
    =======
    fits_ds: xarray.Dataset or dict
        Containing the calculated PSD parameter fields such as "N_0", "lambda", and "mu".

    """
    if hyd_type in ["cl", "pl"]:  # liquid classes
        if model.mcphys_scheme.lower() in ["mg2", "mg", "morrison", "nssl", "p3"]:
            fits_ds = calc_mu_lambda(model, hyd_type, subcolumns=subcolumns, **kwargs).ds
        else:
            raise ValueError(f"no liquid PSD calulation method implemented for scheme {model.mcphys_scheme}")
    else:  # ice classes
        if model.mcphys_scheme.lower() in ["mg2", "mg", "morrison", "nssl"]:  # NOTE: NSSL PSD assumed like MG
            fits_ds = calc_mu_lambda(model, hyd_type, subcolumns=True, **kwargs).ds
        elif model.mcphys_scheme.lower() in ["p3"]:
            fits_ds = {"N_0": model.ds[model.p3_kws["N0_ice_name"]] * model.ds[model.p3_kws["in_cld_Ni_name"]],
                       "lambda": model.ds[model.lambda_field["ci"]],
                       "mu": model.ds[model.mu_field["ci"]],
            }
        else:
            raise ValueError(f"no ice PSD calulation method implemented for scheme {model.mcphys_scheme}")
    return fits_ds
