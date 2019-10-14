import xarray as xr
import numpy as np

from ..core import Instrument

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
    if instrument.instrument_class.lower() is not "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    if hyd_type.lower() is not in ['cl', 'ci', 'pl', 'pi']:
        raise ValueError("%s is not a valid hydrometeor type. Valid choices are cl, ci, pl, and pi." % hyd_type)

    WC = column_ds[q_field]*1e3*column_ds[p_field]*(instrument.R_d*column_ds[t_field])
    if hyd_type.lower() == "cl":
        column_ds['Ze'] = 0.031*WC**1.56
    elif hyd_type.lower() == "pl":
        column_ds['Ze'] = ((WC*1e3)/3.4)**1.75
    else:
        Tc = column_ds[t_field] - 273.15
        if instrument.freq >= 2 and instrument.freq < 4:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0197 * Tc + 1.7) / 0.060) / 10.)
        elif instrument.freq >= 27 and instrument.freq < 40:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.699)) / 10.)
        elif instrument.freq >= 75 and instrument.freq < 110:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.00706 * Tc + 0.992) / (0.000580 * Tc + 0.0923)) /10.)
        else:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.0699)) / 10.)
    return column_ds