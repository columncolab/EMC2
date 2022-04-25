import xarray as xr
import numpy as np
from ..core import Instrument
from ..core.instrument import ureg, quantity


def calc_radar_Ze_min(instrument, model, ref_rng=1000):
    """
    This function calculates the minimum detectable radar signal (Ze_min) profile
    given radar detectability at a reference range.

    Parameters
    ----------
    instrument: :py:mod:`emc2.core.Instrument`
        The Instrument class that you wish to calculate Ze_min for.
    model: :py:mod:`emc2.core.Model`
        The Model class that you wish to calculate the profile for.
    ref_rng: scalar
        Reference altitude for Ze_min
    Returns
    -------
    model: :py:mod:`emc2.core.Model`
        The Model class that will store the Ze_min profile.
    """

    Ze_min = instrument.Z_min_1km + 20 * np.log10(model.ds[model.z_field]) - 20 * np.log10(ref_rng)
    model.ds['Ze_min'] = xr.DataArray(Ze_min, dims=model.ds[model.z_field].dims)
    model.ds["Ze_min"].attrs["long_name"] = "Minimum discernable radar reflectivity factor"
    model.ds["Ze_min"].attrs["units"] = 'dBZ'
    return model


def calc_radar_atm_attenuation(instrument, model):
    """
    This function calculates atmospheric attenuation due to water vapor and O2
    for a given model column.

    Parameters
    ----------
    instrument: :py:mod:`emc2.core.Instrument`
        The Instrument class that you wish to calculate the attenuation parameters for.
    model: :py:mod:`emc2.core.Model`
        The Model class that you wish to calculate the attenuation parameters for.

    Returns
    -------
    model: :py:mod:`emc2.core.Model`
        The Model class that will store the attenuation parameters.
    """

    if not isinstance(instrument, Instrument):
        raise ValueError(str(instrument) + ' is not an Instrument!')

    q_field = model.q_field
    p_field = model.p_field
    t_field = model.T_field

    # Convert to assumed units
    t_temp = quantity(model.ds[t_field].values, model.ds[t_field].attrs["units"]).to("kelvin").magnitude
    p_temp = quantity(model.ds[p_field].values, model.ds[p_field].attrs["units"]).to("hPa").magnitude

    column_ds = model.ds

    rho_wv = column_ds[q_field] * 1e3 * (p_temp * 1e2) / (instrument.R_d * t_temp)
    three_hundred_t = 300. / t_temp
    gamma_l = 2.85 * (p_temp / 1013.) * (three_hundred_t)**0.626 * \
        (1 + 0.018 * rho_wv * t_temp / p_temp)
    kappa_wv = (2 * instrument.freq)**2 * rho_wv * (three_hundred_t)**1.5 * gamma_l * \
        (three_hundred_t) * np.exp(-644 / t_temp) * \
        1 / ((494.4 - instrument.freq**2)**2 + 4 * instrument.freq**2) * gamma_l**2 + 1.2e-6
    f0 = 60.

    gamma_0 = 0.59 * (1 + 3.1e-3 * (333 - p_temp))
    gamma_0[column_ds[p_field].values >= 333] = 0.59
    gamma_0[column_ds[p_field].values < 25.] = 1.18
    gamma_l = gamma_0 * (p_temp / 1013.) * three_hundred_t**0.85
    kappa_o2 = (1.1e-2 * instrument.freq**2) * (p_temp / 1013.) * three_hundred_t**2 * \
        gamma_l * (1. / ((instrument.freq - f0)**2 + gamma_l**2) + 1. /
                   (instrument.freq**2 + gamma_l**2))

    column_ds['kappa_o2'] = xr.DataArray(kappa_o2, dims=model.ds[t_field].dims)
    column_ds['kappa_o2'].attrs["long_name"] = "Gaseous attenuation due to O2"
    column_ds['kappa_o2'].attrs["units"] = r"$dB\ km^{-1}$"

    column_ds['kappa_wv'] = xr.DataArray(kappa_wv.values, dims=model.ds[t_field].dims)
    column_ds['kappa_wv'].attrs["long_name"] = "Gaseous attenuation due to water vapor"
    column_ds['kappa_wv'].attrs["units"] = r"$dB\ km^{-1}$"

    column_ds['kappa_att'] = column_ds['kappa_wv'] + column_ds['kappa_o2']
    column_ds['kappa_att'].attrs["long_name"] = "Gaseous attenuation due to O2 and water vapor"
    column_ds['kappa_att'].attrs["units"] = r"$dB\ km^{-1}$"

    model.ds = column_ds
    return model


def calc_theory_beta_m(model, Lambda, OD_from_sfc=True):
    """
    This calculates the molecular scattering parameters for a given model. In particular, the
    two-way transmittance, optical depth, volume extinction/backscatter cross sections,
    Rayleigh scattering cross sections, number density profile and refreactive index will be
    calculated.

    Parameters
    ----------
    model: Model
        The model to calculate the parameters for.
    Lambda: float
        The wavelength (in microns).
    OD_from_sfc: bool
        If True, optical depth will be calculated from the surface. If false, optical depth will
        be calculated from the top of the atmosphere.

    Returns
    -------
    model: Model
        The model with the molecular scattering parameters added.
    """

    Theta = np.pi
    raw_n = 0.035
    alpha = 0.00366
    nu = 1 / Lambda

    p_temp = model.ds[model.p_field].values * getattr(ureg, model.ds[model.p_field].attrs["units"])
    P = p_temp.to('hPa').magnitude
    t_temp = quantity(model.ds[model.T_field].values, model.ds[model.T_field].attrs["units"])
    T = t_temp.to('degC').magnitude
    z_temp = model.ds[model.z_field].values * getattr(ureg, model.ds[model.z_field].attrs["units"])
    Z = z_temp.to('meter').magnitude
    raw = P * 100 / (model.consts["R_d"] * (T + 273.15)) * 1e3 / 1e6
    p_cos = 0.7629 * (1 + 0.932 * np.cos(Theta)**2)
    n_s_ref = 1 + (6432.8 + 2949810 / (146 - nu**2) + 25540 / (41 - nu**2)) * 1e-8
    n_s = (n_s_ref - 1) * ((1 + alpha * 15) / (1 + alpha * T)) * (P / 1013.25) + 1
    N_s = P * 100 / (model.consts["R"] * (T + 273.15)) * model.consts["Avogadro_c"] / 1e6
    sigma = 8 * np.pi**3 / 3 * (n_s**2 - 1)**2 / \
        (N_s * (Lambda * 1e-4)**4 * N_s) * (6 + 3 * raw_n) / (6 - 7 * raw_n)
    beta = 8 * np.pi**3 / 3 * (n_s**2 - 1)**2 * N_s / \
        (N_s * (Lambda * 1e-4)**4 * N_s) * (6 + 3 * raw_n) / (6 - 7 * raw_n)
    kappa = beta / raw
    sigma_180 = np.pi**2 * (n_s**2 - 1)**2 * 2 * (2 + raw_n) / \
        (N_s * (Lambda * 1e-4)**4 * N_s * (6 - 7 * raw_n)) * p_cos
    sigma_180_vol = sigma_180 * N_s

    sigma = sigma * 1e-4
    N_s = N_s / 1e-6
    beta = beta / 1e-2
    kappa = kappa * 1e-3 / 1e-6
    sigma_180 = sigma_180 * 1e-4
    sigma_180_vol = sigma_180_vol / 1e-2

    Z_4_trap = np.diff(Z, axis=1) / 2.
    summed_beta = beta[:, :-1] + beta[:, 1:]
    u = np.zeros_like(beta)
    if OD_from_sfc:
        u[:, 1:] = np.cumsum(Z_4_trap * summed_beta, axis=1)
    else:
        u[:, :-1] = np.flip(np.cumsum(np.flip(Z_4_trap * summed_beta, axis=1), axis=1), axis=1)

    tau = np.exp(-2 * u)

    my_dims = model.ds[model.T_field].dims
    model.ds["tau"] = xr.DataArray(tau, dims=my_dims)
    model.ds["tau"].attrs["long_name"] = "Two-way transmittance"
    model.ds["tau"].attrs["units"] = "1"

    model.ds["u"] = xr.DataArray(u, dims=my_dims)
    model.ds["u"].attrs["long_name"] = "Atmospheric optical depth"
    model.ds["u"].attrs["units"] = "1"

    model.ds["beta"] = xr.DataArray(beta, dims=my_dims)
    model.ds["beta"].attrs["long_name"] = "Volume extinction cross section"
    model.ds["beta"].attrs["units"] = "m-1"

    model.ds["sigma_180_vol"] = xr.DataArray(sigma_180_vol, dims=my_dims)
    model.ds["sigma_180_vol"].attrs["long_name"] = "Volume backscatter cross section"
    model.ds["sigma_180_vol"].attrs["units"] = r"$m^{-1}$"

    model.ds["sigma_180"] = xr.DataArray(sigma_180, dims=my_dims)
    model.ds["sigma_180"].attrs["long_name"] = "backscatter cross section per molecule"
    model.ds["sigma_180"].attrs["units"] = r"$m^2$"

    model.ds["sigma"] = xr.DataArray(sigma, dims=my_dims)
    model.ds["sigma"].attrs["long_name"] = "Rayleigh scattering cross section per molecule"
    model.ds["sigma"].attrs["units"] = r"$m^2$"

    model.ds["kappa"] = xr.DataArray(kappa, dims=my_dims)
    model.ds["kappa"].attrs["long_name"] = "Mass extinction cross section per molecule"
    model.ds["kappa"].attrs["units"] = r"$kg\ m^{-3}$"

    model.ds["N_s"] = xr.DataArray(N_s, dims=my_dims)
    model.ds["N_s"].attrs["long_name"] = "Number density profile"
    model.ds["N_s"].attrs["units"] = r"$m^{-3}$"

    model.ds["n_s"] = xr.DataArray(n_s, dims=my_dims)
    model.ds["n_s"].attrs["long_name"] = "Refractive index"
    model.ds["n_s"].attrs["units"] = "1"

    return model
