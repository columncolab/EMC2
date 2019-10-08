import xarray as xr
import numpy as np
from ..core import Instrument

def calc_radar_atm_attenuation(instrument, column_ds,
                               p_field="p_3d", t_field="t",
                               q_field="q"):
    """
    This function calculates atmospheric attenuation due to water vapor and CO2
    for a given model column.

    Parameters
    ----------
    instrument: EMC2.core.Instrument
        The Instrument class that you wish to calculate the attenuation parameters for.
    column_ds: xarray Dataset
        The model column dataset containing the temperature, pressure, and water vapor mixing ratio.
    p_field: str
        The name of the pressure field in the model.
    t_field: str
        The name of the temperature field in the model.
    q_field: str
        The name of the water vapor mixing ratio field in the model.

    Returns
    -------
    column_ds: xarray Dataset
        The dataset with the atmospheric attenuation added.
    """

    if not isinstance(instrument, Instrument):
        raise ValueError(str(instrument) + ' is not an Instrument!')

    rho_wv = column_ds[q_field] * 1e3 * (column_ds[p_field] * 1e2) / (instrument.R_d * column_ds[t_field])
    three_hundred_t = 300. / column_ds[t_field]
    gamma_l = 2.85 * (column_ds[p_field] / 1013.) * (three_hundred_t)**0.626 * \
              (1 + 0.018*rho_wv*column_ds[t_field]/column_ds[p_field])
    column_ds['kappa_wv'] = (2*instrument.freq)**2 * rho_wv * (three_hundred_t)**1.5 * gamma_l * \
                            (three_hundred_t) * np.exp(-644/column_ds[t_field]) * \
                            1/((494.4 - instrument.freq**2)**2 + 4*instrument.freq**2) * gamma_l**2 +1.2e-6
    f0 = 60.

    gamma_0 = 0.59 * (1 + 3.1e-3 * (333 - column_ds[p_field].values))
    gamma_0[column_ds[p_field].values >= 333] = 0.59
    gamma_0[column_ds[p_field].values < 25.] = 1.18
    gamma_l = gamma_0 * (column_ds[p_field]/1013) * (three_hundred_t)**0.85
    column_ds['kappa_o2'] = (1.1e-2*instrument.freq**2) * (column_ds[p_field] / 1013.) * three_hundred_t**2 * \
        gamma_l * (1./((instrument.freq - f0)**2 + gamma_l**2) + 1./(instrument.freq**2 + gamma_l**2))
    column_ds['kappa_att'] = column_ds['kappa_wv'] + column_ds['kappa_o2']
    return column_ds


