import xarray as xr
import numpy as np

from ..core import Instrument, hydrometeor_info
from .attenuation import calc_radar_atm_attenuation
from .psd import calc_mu_lambda


def calc_radar_reflectivity_conv(instrument, column_ds, hyd_type,
                                 p_field="p_3d", t_field="t",
                                 q_field="q"):
    """
    This estimates the radar reflectivity given a profile of liquid water mixing ratio.
    Convective DSDs are assumed.

    Parameters
    ----------
    instrument: :func:`emc2.core.Instrument` class
        The instrument to calculate the reflectivity parameters for.
    column_ds: xarray Dataset
        The dataset to calculate the derived reflectivity from.
    hyd_type: str
        The assumed hydrometeor type. Must be one of:
        'cl' (cloud liquid), 'ci' (cloud ice),
        'pl' (liquid precipitation), 'pi' (ice precipitation).
    p_field: str
        The name of the pressure field.
    t_field: str
        The name of the temperature field.
    q_field: str
        The name of the liquid water mixing ratio field.

    Returns
    -------
    column_ds: xarray Dataset
        Returns a dataset with an added reflectivity field.
    """
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    if hyd_type.lower() not in ['cl', 'ci', 'pl', 'pi']:
        raise ValueError("%s is not a valid hydrometeor type. Valid choices are cl, ci, pl, and pi." % hyd_type)

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
    return column_ds


def calc_reflectivity(instrument, column_ds, is_conv,
                      N_field="N", p_field="p", t_field="t",
                      OD_from_sfc=True, z_field="z",
                      q_names=None, **kwargs):
    """
    Calculates the reflectivity in a given column for the given radar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate.
    column_ds: xarray Dataset
        The xarray Dataset containing the model data.
    is_conv: bool
        True if the cell is convective
    N_field: str
        The name of the number concentration variable to use.
    p_field: str
        The name of the pressure field to use.
    t_field: str
        The name of the temperature field to use.
    z_field: str
        The name of the altitude field to use.
    q_names: None or dict
        If None, use the default names for the mixing ratio fields for
        each hydrometeor type. If a dict, it is a dictionary with 4 keys
        labeled 'cl', 'ci', 'pl', and 'pi' whose values are the name of the
        variable corresponding to the mixing ratio of the given hydrometeor type.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    Additional keyword arguments are passed into
    :func:`emc2.simulator.reflectivity.calc_radar_reflectivity_conv` and
    :func:`emc2.simulator.attenuation.calc_radar_atm_attenuation`.

    Returns
    -------
    column_ds: xarray Dataset
        The xarray Dataset containing the calculated reflectivities
    """

    hyd_types = ["cl", "ci", "pl", "pi"]
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    if q_names is None:
        q_names = hydrometeor_info.q_names

    if is_conv:
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
        kappa_f = 6 * np.pi / instrument.wavelength * 1e-6 * hydrometeor_info.Rho_hyd["cl"]
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

    return
