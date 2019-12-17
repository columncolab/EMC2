import xarray as xr
import numpy as np

from ..core import Instrument, Model

def calc_LDR(instrument, model, ext_OD, OD_from_sfc=True, LDR_per_hyd=None, is_conv=True):
    """
    Calculates the lidar extinction mask and linear depolarization ratio for
    the given model and lidar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    ext_OD: float
        The optical depth threshold for determining if the signal is extinct.
    OD_from_sfc: bool
        If True, optical depth will be calculated from the surface. If False,
        optical depth will be calculated from the top of the atmosphere.
    LDR_per_hyd: dict or None
        If a dict, the amount of LDR per hydrometeor class must be specified in
        a dictionary whose keywords are the model's hydrometeor classes. If None,
        the default settings from the model will be used.
    is_conv: bool
        If true, calculate LDR for convective clouds. If False, LDR will be
        calculated for stratiform clouds.
    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """

    if LDR_per_hyd is None:
        LDR_per_hyd = model.LDR_per_hyd

    numerator = -999999
    denominator = 0

    if is_conv:
        cloud_class = "conv"
    else:
        cloud_class = "strat"

    for hyd_type in model.hydrometeor_classes:
        beta_p_key = "sub_col_beta_p_%s_%s" % (hyd_type, cloud_class)
        if numerator == -999999:
            numerator = model.ds[beta_p_key]*model.LDR_per_hyd[hyd_type]
            denominator = model.ds[beta_p_key]
        else:
            numerator += model.ds[beta_p_key]*model.LDR_per_hyd[hyd_type]
            denominator += model.ds[beta_p_key]

    model.ds["LDR"] = numerator / denominator
    model.ds["LDR"] = model.ds["LDR"].where(model.ds["LDR"] != 0)
    model.ds["LDR"].attrs["long_name"] = "Linear depolarization ratio"

    OD_cum_p_tot = model.ds["sub_col_OD_tot_%s" % cloud_class]
    ext_tmp = np.where(np.diff(OD_cum_p_tot > ext_OD) == 1, 1, 0)
    ext_mask = np.where(np.cumsum(ext_tmp, axis=1) > 0, 2, 0) - ext_tmp

    if not OD_from_sfc:
        ext_mask = np.flip(ext_mask, axis=1)

    model.ds["ext_mask"] = ext_mask
    model.ds["ext_mask"].long_name = "Extinction mask"
    model.ds["ext_mask"].units = "2 = Signal extinct, 1 = layer where signal becomes extinct, 0 = signal not extinct"

    return model


def calc_lidar_moments(instrument, model, is_conv, ext_OD,
                       OD_from_sfc=True, **kwargs):
    """
    Calculates the lidar backscatter, extinction, and optical depth
    in a given column for the given lidar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    ext_OD: float
        The optical depth at which the lidar signal is fully extinct.
    z_field: str
        The name of the altitude field to use.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.

    Additonal keyword arguments are passed into
    :func:`emc2.simulator.lidar_moments.calc_LDR`.
    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """

    hyd_types = ["cl", "ci", "pl", "pi"]
    if not instrument.instrument_class.lower() == "lidar":
        raise ValueError("Instrument must be a lidar!")

    p_field = model.p_field
    t_field = model.T_field
    z_field = model.z_field
    column_ds = model.ds

    if is_conv:
        q_names = model.q_names_convective
        Dims = model.ds[q_names["cl"]].values.shape
        column_ds['sub_col_beta_p_tot_conv'] = xr.DataArray(np.zeros(Dims),
                                                            dims=model.ds[q_names].dims)
        column_ds['sub_col_alpha_p_tot_conv'] = xr.DataArray(np.zeros(Dims),
                                                            dims=model.ds[q_names].dims)
        column_ds['sub_col_OD_tot_conv'] = xr.DataArray(np.zeros(Dims),
                                                        dims=model.ds[q_names].dims)
        for hyd_type in hyd_types:
            WC = model.ds[q_names[hyd_type]] * 1e3 * model.ds[p_field] / (instrument.R_d * model.ds[t_field])
            WC = np.tile(WC, (model.num_subcolumns, 1))
            empr_array = model.ds[model.re_fields{hyd_type}].values
            if hyd_type == "cl" or hyd_type == "pl":
                model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = (3 * WC)/(2 * model.Rho_hyd[hyd_type] * 1e-3) * \
                    np.tile(empr_array, (model.num_subcolumns, 1, 1))
            else:
                temp = model.ds[t_field].values
                a = 0.00532 * ((temp - 273.15) + 90)**2.55
                b = 1.31 * np.exp(0.0047 * (temp - 273.15))
                a = np.tile(a, (model.num_subcolumns, 1, 1))
                b = np.tile(b, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = WC / (a**(1 / b))
            model.ds["sub_col_beta_p_%s_conv" % hyd_type] = model.ds["sub_col_alpha_p_%s_conv" % hyd_type] / \
                                                            model.lidar_ratio[hyd_type]
            model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = model.ds["sub_col_alpha_p_%s_conv" % hyd_type].fillna(0)
            model.ds["sub_col_beta_p_%s_conv" % hyd_type] = model.ds["sub_col_beta_p_%s_conv" % hyd_type].fillna(0)
            if OD_from_sfc:
                dz = np.diff(column_ds[z_field].values, axis=0, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_conv" % hyd_type] = np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_conv" % hyd_type])
            else:
                dz = np.diff(column_ds[z_field].values, axis=0, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_conv" % hyd_type] = np.flip(np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_conv" % hyd_type]))
            model.ds["sub_col_beta_p_tot_conv"] += model.ds["sub_col_beta_p_%s_conv" % hyd_type]
            model.ds["sub_col_alpha_p_tot_conv"] += model.ds["sub_col_alpha_p_%s_conv" % hyd_type]
            model.ds["sub_col_OD_tot_conv"] += model.ds["sub_col_OD_%s_conv" % hyd_type]
            if hyd_type == "pl":
                model.ds = calc_LDR(instrument, model, ext_OD, OD_from_sfc, **kwargs)
                model.ds["sub_col_beta_p_tot_conv"]  = model.ds["sub_col_beta_p_tot_conv"].where(
                    model.ds["sub_col_beta_p_tot_conv"] != 0)
                model.ds["sub_col_alpha_p_tot_conv"] = model.ds["sub_col_alpha_p_tot_conv"].where(
                    model.ds["sub_col_alpha_p_tot_conv"] != 0)
                model.ds["sub_col_OD_tot_conv"] = model.ds["sub_col_OD_tot_conv"].where(
                    model.ds["sub_col_OD_tot_conv"] != 0)






