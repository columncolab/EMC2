import xarray as xr
import numpy as np

from ..core import Instrument, Model
from .attenuation import calc_radar_atm_attenuation
from .psd import calc_mu_lambda


def calc_radar_reflectivity_conv(instrument, model, hyd_type):
    """
    This estimates the radar reflectivity given a profile of liquid water mixing ratio.
    Convective DSDs are assumed.

    Parameters
    ----------
    instrument: :func:`emc2.core.Instrument` class
        The instrument to calculate the reflectivity parameters for.
    model: :func:`emc2.core.Model` class
        The model to calculate the parameters for.
    hyd_type: str
        The assumed hydrometeor type. Must be one of:
        'cl' (cloud liquid), 'ci' (cloud ice),
        'pl' (liquid precipitation), 'pi' (ice precipitation).

    Returns
    -------
    model: :func:`emc2.core.Model`
        Returns a Model with an added reflectivity field.
    """
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    if hyd_type.lower() not in ['cl', 'ci', 'pl', 'pi']:
        raise ValueError("%s is not a valid hydrometeor type. Valid choices are cl, ci, pl, and pi." % hyd_type)
    q_field = model.q_names_convective[hyd_type]
    p_field = model.p_field
    t_field = model.T_field
    column_ds = model.ds

    WC = column_ds[q_field] * 1e3 * column_ds[p_field] / (instrument.R_d * column_ds[t_field])
    if hyd_type.lower() == "cl":
        column_ds['Ze'] = 0.031 * WC**1.56
    elif hyd_type.lower() == "pl":
        column_ds['Ze'] = ((WC * 1e3) / 3.4)**1.75
    else:
        Tc = column_ds[t_field] - 273.15
        if instrument.freq >= 2 and instrument.freq < 4:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0197 * Tc + 1.7) / 0.060) / 10.)
        elif instrument.freq >= 27 and instrument.freq < 40:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.699)) / 10.)
        elif instrument.freq >= 75 and instrument.freq < 110:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.00706 * Tc + 0.992) / (0.000580 * Tc + 0.0923)) / 10.)
        else:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.0699)) / 10.)
    column_ds['Ze'] = 10 * np.log10(column_ds["Ze"])
    column_ds['Ze'].attrs["long_name"] = "Radar reflectivity factor"
    column_ds['Ze'].attrs["units"] = "dBZ"
    model.ds = column_ds
    return model


def calc_reflectivity(instrument, model, is_conv,
                      OD_from_sfc=True, **kwargs):
    """
    Calculates the reflectivity in a given column for the given radar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    z_field: str
        The name of the altitude field to use.

    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    Additional keyword arguments are passed into
    :func:`emc2.simulator.reflectivity.calc_radar_reflectivity_conv` and
    :func:`emc2.simulator.attenuation.calc_radar_atm_attenuation`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The xarray Dataset containing the calculated reflectivities
    """

    hyd_types = ["cl", "ci", "pl", "pi"]
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    p_field = model.P_field
    t_field = model.T_field
    z_field = model.z_field
    column_ds = model.ds

    if is_conv:
        q_names = model.q_names_convective
        for hyd_type in hyd_types:
            temp_ds = calc_radar_reflectivity_conv(instrument, column_ds, hyd_type,
                                                   q_field=q_names[hyd_type], p_field=p_field,
                                                   t_field=t_field, **kwargs)
            var_name = "sub_col_Ze_%s_conv" % hyd_type
            column_ds[var_name] = temp_ds["Ze"]
            if "sub_col_Ze_tot_cov" in column_ds.variables.keys():
                column_ds["sub_col_Ze_tot_conv"] += column_ds["sub_col_Ze_cl_conv"]
            else:
                column_ds["sub_col_Ze_tot_conv"] = column_ds[var_name]

        kappa_ds = calc_radar_atm_attenuation(instrument, column_ds,
                                              p_field=p_field, t_field=t_field, **kwargs)
        kappa_f = 6 * np.pi / instrument.wavelength * 1e-6 * model.Rho_hyd["cl"]
        WC = column_ds[q_names["cl"]] + column_ds[q_names["pl"]] * 1e3 * \
            column_ds[p_field] / (instrument.R_d * column_ds[t_field])
        dz = np.diff(column_ds[z_field].values, axis=1)

        if OD_from_sfc:
            WC_new = np.zeros_like(WC)
            WC_new[:, 1:, :, :] = WC[:, :-1, :, :]
            liq_ext = np.cumsum(kappa_f * dz * WC_new, axis=1)
            atm_ext = np.cumsum(kappa_ds["kappa_att"] * dz, axis=1)
        else:
            WC_new = np.zeros_like(WC)
            WC_new[:, :-1, :, :] = WC[:, 1:, :, :]
            liq_ext = np.flip(np.cumsum(kappa_f * dz * WC_new, axis=1), axis=1)
            atm_ext = np.flip(np.cumsum(kappa_ds["kappa_att"] * dz, axis=1), axis=1)

        column_ds["sub_col_Ze_att_tot_conv"] = column_ds["sub_col_Ze_tot_conv"] / \
            10**(2 * liq_ext / 10.) / 10**(2 * atm_ext / 10.)
        column_ds["sub_col_Ze_tot_conv"] = column_ds["sub_col_Ze_tot_conv"].where(
            column_ds["sub_col_Ze_tot_conv"] != 0)
        return column_ds

    q_names = model.q_names_stratiform
    # n_names = model.N_field
    # frac_names = model.strat_frac_names
    # Dims = column_ds[q_names["cl"]].values.shape
    # moment_denom_tot = np.zeros(Dims)
    # V_d_numer_tot = np.zeros(Dims)
    # sigma_d_numer_tot = np.zeros(Dims)
    # od_tot = np.zeros(Dims)
    for hyd_type in ["pi", "pl", "ci", "cl"]:
        # dD = np.diff(instrument.mie_table["p_diam"][1:3])
        column_ds = calc_mu_lambda(column_ds, model, **kwargs)
        # total_hydrometeor = column_ds[frac_names[hyd_type]] * column_ds[n_names[hyd_type]]
