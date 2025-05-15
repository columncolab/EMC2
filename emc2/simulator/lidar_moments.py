import xarray as xr
import numpy as np
import dask.bag as db
from time import time
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator

from .attenuation import calc_theory_beta_m
from .psd import calc_and_set_psd_params
from .radar_moments import _set_p3_tiled_arrays
from ..core.instrument import ureg, quantity


def calc_total_alpha_beta(model, OD_from_sfc=True, eta=1):
    """
    Calculates total (strat+conv) lidar variables.

    Parameters
    ----------
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
    eta: float
        Multiple scattering coefficient.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    if OD_from_sfc:
        OD_str = "layer base"
    else:
        OD_str = "layer top"

    if model.process_conv:
        model.ds["sub_col_beta_p_tot"] = model.ds["sub_col_beta_p_tot_conv"].fillna(0) + \
            model.ds["sub_col_beta_p_tot_strat"].fillna(0)
        model.ds["sub_col_alpha_p_tot"] = model.ds["sub_col_alpha_p_tot_conv"].fillna(0) + \
            model.ds["sub_col_alpha_p_tot_strat"].fillna(0)
        model.ds["sub_col_OD_tot"] = model.ds["sub_col_OD_tot_conv"].fillna(0) + \
            model.ds["sub_col_OD_tot_strat"].fillna(0)
    else:
        model.ds["sub_col_beta_p_tot"] = model.ds["sub_col_beta_p_tot_strat"].fillna(0)
        model.ds["sub_col_alpha_p_tot"] = model.ds["sub_col_alpha_p_tot_strat"].fillna(0)
        model.ds["sub_col_OD_tot"] = model.ds["sub_col_OD_tot_strat"].fillna(0)
    model.ds["sub_col_beta_p_tot"].attrs["long_name"] = \
        "Total backscatter coefficient (convective + stratiform)"
    model.ds["sub_col_beta_p_tot"].attrs["units"] = r"$m^{-1} sr^{-1}$"
    model.ds["sub_col_alpha_p_tot"].attrs["long_name"] = \
        "Total extinction coefficient (convective + stratiform)"
    model.ds["sub_col_alpha_p_tot"].attrs["units"] = r"$m^{-1}$"
    model.ds["sub_col_OD_tot"].attrs["long_name"] = \
        "Total cumulative optical depth at %s (convective + stratiform)" % OD_str
    model.ds["sub_col_OD_tot"].attrs["units"] = "1"
    beta_m = np.tile(model.ds['sigma_180_vol'].values, (model.num_subcolumns, 1, 1))
    T = np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
    model.ds['sub_col_beta_att_tot'] = (beta_m + model.ds['sub_col_beta_p_tot']) * \
        T * np.exp(-2 * eta * model.ds['sub_col_OD_tot'])
    model.ds["sub_col_beta_att_tot"].attrs["long_name"] = \
        "Total attenuated backscatter coefficient (convective + stratiform)"
    model.ds["sub_col_beta_att_tot"].attrs["units"] = r"$m^{-1} sr^{-1}$"

    return model


def calc_LDR_and_ext(model, ext_OD=4., OD_from_sfc=True, LDR_per_hyd=None, hyd_types=None):
    """
    Calculates the lidar extinction mask (for conv+strat) and linear depolarization ratio
    (per strat, conv, and strat+conv) for the given model and lidar. Run after calculating
    'sub_col_OD_tot'.

    Parameters
    ----------
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
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    if model.process_conv:
        cld_classes = ["conv", "strat"]
    else:
        cld_classes = ["strat"]

    if LDR_per_hyd is None:
        LDR_per_hyd = model.LDR_per_hyd

    if OD_from_sfc:
        OD_str = ("layer base", "from surface")
    else:
        OD_str = ("layer top", "from TOA")

    numerator_tot = xr.zeros_like(model.ds["sub_col_beta_p_%s_strat" % model.hydrometeor_classes[0]])
    denominator_tot = xr.zeros_like(model.ds["sub_col_beta_p_%s_strat" % model.hydrometeor_classes[0]])
    for cloud_str in cld_classes:
        numerator = 0.
        denominator = 0.

        for hyd_type in hyd_types:
            beta_p_key = "sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)
            numerator += model.ds[beta_p_key].fillna(0) * model.LDR_per_hyd[hyd_type].magnitude
            denominator += model.ds[beta_p_key].fillna(0)
        denominator_no_zeros = np.where(denominator > 0, denominator, np.nan)
        model.ds["sub_col_LDR_%s" % cloud_str] = numerator / denominator_no_zeros
        model.ds["sub_col_LDR_%s" % cloud_str].attrs["long_name"] = \
            "Linear depolarization ratio in %s" % cloud_str
        model.ds["sub_col_LDR_%s" % cloud_str].attrs["units"] = "1"
        numerator_tot += numerator
        denominator_tot += denominator

    denominator_tot = np.where(denominator_tot > 0, denominator_tot, np.nan)
    model.ds["sub_col_LDR_tot"] = numerator_tot / denominator_tot
    model.ds["sub_col_LDR_tot"].attrs["long_name"] = "Linear depolarization ratio (convective + stratiform)"
    model.ds["sub_col_LDR_tot"].attrs["units"] = "1"

    OD_cum_p_tot = \
        np.where(model.ds["sub_col_OD_tot"].values > ext_OD, 2, 0.)
    if OD_from_sfc:
        my_diff = np.diff(OD_cum_p_tot, axis=2, append=0)
    else:
        my_diff = np.flip(np.diff(np.flip(OD_cum_p_tot, axis=2), axis=2, append=0), axis=2)
    ext_tmp = np.where(my_diff > 0., 1, 0)
    ext_mask = OD_cum_p_tot + ext_tmp

    model.ds["ext_mask"] = xr.DataArray(ext_mask, dims=model.ds["sub_col_LDR_strat"].dims)
    model.ds["ext_mask"].attrs["long_name"] = "Extinction mask at %s based on optical thickness considerations \
        (convective + stratiform; calculated %s)" % OD_str
    model.ds["ext_mask"].attrs["units"] = ("2 = Signal extinct, 1 = layer where signal becomes " +
                                           "extinct, 0 = signal not extinct")

    return model


def accumulate_OD(model, is_conv, z_values, hyd_type, OD_from_sfc=True, **kwargs):
    """
    Accumulates optical thickness from TOA or the surface.

    Parameters
    ----------
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    z_values: ndarray
        model output height array in m.
    hyd_type: string
        hydrometeor class name to include in calcuation.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    if is_conv:
        cloud_str = "conv"
    else:
        cloud_str = "strat"

    Dims = model.ds["%s_q_subcolumns_%s" % (cloud_str, hyd_type)].shape
    if OD_from_sfc:
        dz = np.tile(np.diff(z_values, axis=1, prepend=0.), (model.num_subcolumns, 1, 1))
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(np.cumsum(
            dz * np.concatenate((np.zeros(Dims[:2] + (1,)),
                                 model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)][:, :, :-1]), axis=2),
            axis=2), dims=model.ds["%s_q_subcolumns_%s" % (cloud_str, hyd_type)].dims)
    else:
        dz = np.tile(np.diff(z_values, axis=1, append=0.), (model.num_subcolumns, 1, 1))
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(np.flip(np.cumsum(
            np.flip(dz * np.concatenate((model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)][:, :, 1:],
                                         np.zeros(Dims[:2] + (1,))), axis=2), axis=2), axis=2), axis=2),
            dims=model.ds["%s_q_subcolumns_%s" % (cloud_str, hyd_type)].dims)

    return model


def calc_lidar_empirical(instrument, model, is_conv, p_values, t_values, z_values,
                         OD_from_sfc=True, hyd_types=None, **kwargs):
    """
    Calculates the lidar stratiform or convective backscatter, extinction, and
    optical depth in a sub-columns using empirical formulation from literature.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    p_values: ndarray
        model output pressure array in Pa.
    t_values: ndarray
        model output temperature array in C.
    z_values: ndarray
        model output height array in m.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.
    Additonal keyword arguments are passed into
    :py:func:`emc2.simulator.lidar_moments.accumulate_OD`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    if is_conv:
        cloud_str = "conv"
    else:
        cloud_str = "strat"

    Dims = model.ds["%s_q_subcolumns_cl" % cloud_str].shape
    model.ds['sub_col_beta_p_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)
    model.ds['sub_col_alpha_p_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)
    model.ds['sub_col_OD_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)

    for hyd_type in hyd_types:
        WC = model.ds["%s_q_subcolumns_%s" % (cloud_str, hyd_type)] * p_values / \
            (instrument.R_d * (t_values + 273.15))
        if is_conv:
            empr_array = model.ds[model.conv_re_fields[hyd_type]].values

        else:
            empr_array = model.ds[model.strat_re_fields[hyd_type]].values
        if hyd_type == "cl" or hyd_type == "pl":
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                (3 * WC) / (2 * model.Rho_hyd[hyd_type] * 1e-6 *
                            np.tile(empr_array, (model.num_subcolumns, 1, 1))),
                dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)
        else:
            # Heymsfield et al. (2014)
            a = 0.00532 * (t_values + 90) ** 2.55
            b = 1.31 * np.exp(0.0047 * t_values)
            a = np.tile(a, (model.num_subcolumns, 1, 1))
            b = np.tile(b, (model.num_subcolumns, 1, 1))
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                (WC / a) ** (1 / b), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)

        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)] = \
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] / \
            model.lidar_ratio[hyd_type].magnitude
        model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = \
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)] = \
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model = accumulate_OD(model, is_conv, z_values, hyd_type, OD_from_sfc, **kwargs)

        model.ds["sub_col_beta_p_tot_%s" % cloud_str] += \
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model.ds["sub_col_alpha_p_tot_%s" % cloud_str] += \
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model.ds["sub_col_OD_tot_%s" % cloud_str] += \
            model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].fillna(0)

    return model


def calc_lidar_bulk(instrument, model, is_conv, p_values, z_values, OD_from_sfc=True,
                    hyd_types=None, mie_for_ice=False, **kwargs):
    """
    Calculates the lidar stratiform or convective backscatter, extinction, and
    optical depth in a sub-columns using bulk scattering LUTs assuming geometric
    scatterers (radiation scheme logic).
    Effective radii for each hydrometeor class must be provided (in model.ds).

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    p_values: ndarray
        model output pressure array in Pa.
    z_values: ndarray
        model output height array in m.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.
    mie_for_ice: bool
        If True, using bulk LUTs generated using solid ice spheres (Mie) calculations (e.g., consistent
        with the ice treatment in MG1 through MG2).
        Otherwise, using bulk LUTs that consider ice properties, e.g., the C6 8-column severly roughned
        aggregate for E3 or the m-D A-D for CESM and E3SMv1, consistent with the radiation schemes of
        these models.
        This parameter is set to `False` if `mcphys_scheme == P3` since ice shape is integrated into P3,
        which also affects mu, etc. and therefore the bulk LUT behavior in a similar manner to how the P3
        LUTs are integrated into EMC².
    Additonal keyword arguments are passed into
    :py:func:`emc2.simulator.lidar_moments.accumulate_OD`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    optional_ice_classes = ["ci", "pi", "sn", "gr", "ha", "pir", "pid", "pif"]

    if is_conv:
        cloud_str = "conv"
        re_fields = model.conv_re_fields
    else:
        cloud_str = "strat"
        re_fields = model.strat_re_fields

    n_subcolumns = model.num_subcolumns

    if model.rad_scheme_family == "CESM":
        bulk_ice_lut = "CESM_ice"
        bulk_mie_ice_lut = "mie_ice_CESM_PSD"
        bulk_liq_lut = "CESM_liq"
    elif model.rad_scheme_family == "ModelE":
        bulk_ice_lut = "E3_ice"
        bulk_mie_ice_lut = "mie_ice_E3_PSD"
        bulk_liq_lut = "E3_liq"
    elif model.rad_scheme_family == "P3":
        bulk_liq_lut = "CESM_liq"  # No change from CESM to P3 in liq PSD description
    else:
        raise ValueError(f"Unknown radiation scheme family: {model.rad_scheme_family}")

    Dims = model.ds["%s_q_subcolumns_cl" % cloud_str].shape
    model.ds['sub_col_beta_p_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)
    model.ds['sub_col_alpha_p_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)
    model.ds['sub_col_OD_tot_%s' % cloud_str] = xr.DataArray(
        np.zeros(Dims), dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims)

    rhoa_dz = np.tile(np.abs(np.diff(p_values, axis=1, append=0.)) / instrument.g,
                      (model.num_subcolumns, 1, 1))
    dz = np.tile(np.diff(z_values, axis=1, append=0.), (model.num_subcolumns, 1, 1))
    for hyd_type in hyd_types:
        rad_A = True  # calculate total surface area using classic derivation from r_eff and q_i
        if hyd_type[-1] == 'l':
            rho_b = model.Rho_hyd[hyd_type]  # bulk water
            re_array = np.tile(model.ds[re_fields[hyd_type]], (model.num_subcolumns, 1, 1))
            if model.lambda_field is not None:  # assuming my and lambda can be provided only for liq hydrometeors
                if not model.lambda_field[hyd_type] is None:
                    lambda_array = np.tile(
                        model.ds[model.lambda_field[hyd_type]].values, (model.num_subcolumns, 1, 1))
                    mu_array = np.tile(model.ds[model.mu_field[hyd_type]].values, (model.num_subcolumns, 1, 1))
        elif np.all([np.isin(hyd_type, ["ci"]), model.rad_scheme_family in ["P3"]]):  # P3 ice
            if instrument.instrument_str not in model.interpobj["bulk"].keys():
                model.interpobj["bulk"][instrument.instrument_str] = {}  # gen scattering interp objects
                scat_vars = ["Q_back_bulk", "Q_ext_bulk", "Q_back_Mie_bulk", "Q_ext_Mie_bulk"]
                for key in scat_vars:
                    model.interpobj["bulk"][instrument.instrument_str][key] = RegularGridInterpolator(
                        (instrument.scat_table['p3_ice']["Fr"].values,
                         instrument.scat_table['p3_ice']["rho_r"].values,
                         instrument.scat_table['p3_ice']["q_norm"].values),
                        instrument.scat_table['p3_ice'][key].values, bounds_error=False)
            i_calc_kws = {"Q_back_bulk": model.interpobj["bulk"][instrument.instrument_str]["Q_back_bulk"],
                          "Q_ext_bulk": model.interpobj["bulk"][instrument.instrument_str]["Q_ext_bulk"],
                          "Q_back_Mie_bulk": model.interpobj["bulk"][instrument.instrument_str]["Q_back_Mie_bulk"],
                          "Q_ext_Mie_bulk": model.interpobj["bulk"][instrument.instrument_str]["Q_ext_Mie_bulk"],
                          "Fr_in": model.ds["Fr"].values,
                          "rho_r_in": model.ds["rho_r"].values,
                          "q_norm_in": model.ds["qi_norm"].values,
                          "Ni_ice": model.ds[model.p3_kws["in_cld_Ni_name"]].values,
                         }  # allocate only once
            if model.p3_kws["use_hybrid_scat"] != 1:  # Considering PSD-informed surface area
                if model.p3_kws["use_hybrid_scat"] == 0:  # Full P3
                    A_var = "A_tot_norm"
                elif mie_for_ice:  # using A assuming solid spheres
                    A_var = "A_tot_Mie_norm"
                elif model.p3_kws["use_hybrid_scat"] == 2:  # A tot of eq. vol sphere
                    A_var = "A_tot_eq_V_norm"
                i_calc_kws["A_tot"] = model.interpobj["bulk"][A_var]
                rad_A = False
                A_interped =  i_calc_kws["A_tot"]((
                    i_calc_kws["Fr_in"], i_calc_kws["rho_r_in"], i_calc_kws["q_norm_in"]))
                A_hyd = np.tile(A_interped * i_calc_kws["Ni_ice"], (model.num_subcolumns, 1, 1))
            else:  # using bulk r_eff, in which the eff. density is already implicitly incorporated
                rho_b = instrument.rho_i  # bulk ice
                re_interped =  model.interpobj["bulk"]["ri_eff"]((
                    i_calc_kws["Fr_in"], i_calc_kws["rho_r_in"], i_calc_kws["q_norm_in"])) * 1e6  # um for now
                re_array = np.tile(re_interped, (model.num_subcolumns, 1, 1))
        else:
            rho_b = instrument.rho_i  # bulk ice
            rho_hyd = model.Rho_hyd[hyd_type]
            if rho_hyd == 'variable':
                rho_hyd = model.ds[model.variable_density[hyd_type]].values
            fi_factor = model.fluffy[hyd_type].magnitude * rho_hyd / rho_b + \
                (1 - model.fluffy[hyd_type].magnitude) * (rho_hyd / rho_b) ** (1 / 3)
            re_array = np.tile(model.ds[re_fields[hyd_type]].values * fi_factor,
                               (model.num_subcolumns, 1, 1))

        sub_frac_arr = model.ds["strat_frac_subcolumns_%s" % hyd_type].values
        if rad_A:
            tau_hyd = np.where(sub_frac_arr == 1,
                               3 * model.ds["%s_q_subcolumns_%s" % (cloud_str, hyd_type)] * rhoa_dz /
                               (2 * rho_b * re_array * 1e-6), 0)
            A_hyd = tau_hyd / (2 * dz)  # model assumes geometric scatterers

        if np.isin(hyd_type, optional_ice_classes):
            if model.rad_scheme_family in ["P3"]:  # P3 LUTs
                scat_vars = ["Q_back_bulk", "Q_ext_bulk"]
                if mie_for_ice:
                    scat_lut_vars = ["Q_back_Mie_bulk", "Q_ext_Mie_bulk"]
                else:
                    scat_lut_vars = ["Q_back_bulk", "Q_ext_bulk"]
                scat_dict = {}
                for lut_key, key in zip(scat_lut_vars, scat_vars):
                    scat_tmp = i_calc_kws[lut_key]((
                        i_calc_kws["Fr_in"], i_calc_kws["rho_r_in"], i_calc_kws["q_norm_in"]))
                    scat_dict[key] = np.tile(scat_tmp, (model.num_subcolumns, 1, 1))
            elif mie_for_ice:
                r_eff_bulk = instrument.bulk_table[bulk_mie_ice_lut]["r_e"].values.copy()
                Qback_bulk = instrument.bulk_table[bulk_mie_ice_lut]["Q_back"].values
                Qext_bulk = instrument.bulk_table[bulk_mie_ice_lut]["Q_ext"].values
            else:
                r_eff_bulk = instrument.bulk_table[bulk_ice_lut]["r_e"].values.copy()
                Qback_bulk = instrument.bulk_table[bulk_ice_lut]["Q_back"].values
                Qext_bulk = instrument.bulk_table[bulk_ice_lut]["Q_ext"].values
        else:
            if model.rad_scheme_family in ["CESM", "P3"]:  # rad families that have bulk LUTs vs. lambda and mu
                mu_b = np.tile(instrument.bulk_table[bulk_liq_lut]["mu"].values,
                               (instrument.bulk_table[bulk_liq_lut]["lambdas"].size)).flatten()
                lambda_b = instrument.bulk_table[bulk_liq_lut]["lambda"].values.flatten()
            else:  # rad families that have bulk LUTs vs. r_eff
                r_eff_bulk = instrument.bulk_table[bulk_liq_lut]["r_e"].values
            Qback_bulk = instrument.bulk_table[bulk_liq_lut]["Q_back"].values
            Qext_bulk = instrument.bulk_table[bulk_liq_lut]["Q_ext"].values

        # The following if condition is to determine if the LUTs are vs. the PSD lambda and mu parameters
        # The alternative default is to use LUTs vs. r_eff
        # ===========================================================
        if np.all([np.isin(hyd_type, ["cl", "pl"]), model.rad_scheme_family in ["CESM", "P3"]]):
            print("2-D interpolation of bulk liq lidar backscattering using mu-lambda values")
            rel_locs = sub_frac_arr == 1
            back_tmp = np.full(mu_array.shape, np.nan, dtype=float)
            ext_tmp = np.copy(back_tmp)
            interpolator = LinearNDInterpolator(np.stack((mu_b, lambda_b), axis=1), Qback_bulk.flatten())
            interp_vals = interpolator(mu_array[rel_locs], lambda_array[rel_locs])
            np.place(back_tmp, rel_locs, interp_vals)
            print("2-D interpolation of bulk liq lidar extinction using mu-lambda values")
            interpolator = LinearNDInterpolator(np.stack((mu_b, lambda_b), axis=1), Qext_bulk.flatten())
            interp_vals = interpolator(mu_array[rel_locs], lambda_array[rel_locs])
            np.place(ext_tmp, rel_locs, interp_vals)
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                back_tmp * A_hyd,
                dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims).fillna(0)
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                ext_tmp * A_hyd,
                dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims).fillna(0)
        elif np.all([np.isin(hyd_type, ["ci"]), model.rad_scheme_family in ["P3"]]):  # P3 ice
            print("3-D interpolation of bulk ice lidar backscattering using Fr-rho_r-q_norm values")
            back_tmp = np.where(sub_frac_arr, scat_dict["Q_back_bulk"], np.nan)
            print("3-D interpolation of bulk ice lidar extinction using Fr-rho_r-q_norm values")
            ext_tmp = np.where(sub_frac_arr, scat_dict["Q_ext_bulk"], np.nan)
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                back_tmp * A_hyd,
                dims=model.ds["%s_q_subcolumns_ci" % cloud_str].dims).fillna(0)
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                ext_tmp * A_hyd,
                dims=model.ds["%s_q_subcolumns_ci" % cloud_str].dims).fillna(0)
        else:
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                np.interp(re_array, r_eff_bulk, Qext_bulk) * A_hyd,
                dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims).fillna(0)
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)] = xr.DataArray(
                np.interp(re_array, r_eff_bulk, Qback_bulk) * A_hyd,
                dims=model.ds["%s_q_subcolumns_cl" % cloud_str].dims).fillna(0)

        model = accumulate_OD(model, is_conv, z_values, hyd_type, OD_from_sfc, **kwargs)

        model.ds["sub_col_beta_p_tot_%s" % cloud_str] += \
            model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model.ds["sub_col_alpha_p_tot_%s" % cloud_str] += \
            model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].fillna(0)
        model.ds["sub_col_OD_tot_%s" % cloud_str] += \
            model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].fillna(0)

    return model


def calc_lidar_micro(instrument, model, z_values, OD_from_sfc=True,
                     hyd_types=None, mie_for_ice=False, parallel=True, chunk=None,
                     **kwargs):
    """
    Calculates the lidar backscatter, extinction, and optical depth
    in a given column for the given lidar using the microphysics (MG2) logic.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    z_values: ndarray
        model output height array in m.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.
    mie_for_ice: bool
        If True, using bulk LUTs generated using solid ice spheres (Mie) calculations (e.g., consistent
        with the ice treatment in MG1 through MG2).
        Otherwise, using bulk LUTs that consider ice properties, e.g., the C6 8-column severly roughned
        aggregate for E3 or the m-D A-D for CESM and E3SMv1, consistent with the radiation schemes of
        these models.
        This parameter is set to `False` if `mcphys_scheme == P3` since ice shape is integrated into P3,
        which also affects mu, etc. and therefore the bulk LUT behavior in a similar manner to how the P3
        LUTs are integrated into EMC².
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    chunk: int or None
        The number of entries to process in one parallel loop. None will send all of
        the entries to the Dask worker queue at once. Sometimes, Dask will freeze if
        too many tasks are sent at once due to memory issues, so adjusting this number
        might be needed if that happens.
    Additonal keyword arguments are passed into
    :py:func:`emc2.psd.calc_and_set_psd_params`.
    :py:func:`emc2.simulator.lidar_moments.accumulate_OD`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    optional_ice_classes = ["ci", "pi", "sn", "gr", "ha", "pir", "pid", "pif"]

    if model.rad_scheme_family == "CESM":
        ice_lut = "CESM_ice"
        ice_diam_var = "p_diam"
    elif model.rad_scheme_family == "ModelE":
        ice_lut = "E3_ice"
        ice_diam_var = "p_diam_eq_V"
    elif model.rad_scheme_family == "P3":
        ice_lut = "p3_ice"
        ice_diam_var = "p_diam"
    else:
        raise ValueError(f"Unknown radiation scheme family: {model.rad_scheme_family}")

    Dims = model.ds["strat_q_subcolumns_cl"].values.shape
    for hyd_type in hyd_types:
        frac_names = "strat_frac_subcolumns_%s" % hyd_type
        n_names = "strat_n_subcolumns_%s" % hyd_type
        print(f"Generating strat lidar variables for hyd class {hyd_type} "
              f"({model.rad_scheme_family} rad; {model.mcphys_scheme} mcphys; Mie={int(mie_for_ice)})")
        if not np.isin("sub_col_beta_p_tot_strat", [x for x in model.ds.keys()]):
            model.ds["sub_col_beta_p_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
            model.ds["sub_col_alpha_p_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
            model.ds["sub_col_OD_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
        model.ds["sub_col_beta_p_%s_strat" % hyd_type] = xr.DataArray(
            np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
        model.ds["sub_col_alpha_p_%s_strat" % hyd_type] = xr.DataArray(
            np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)

        # set PSD parameters based on microphysics scheme and hydrometeor class
        fits_ds = calc_and_set_psd_params(model, hyd_type, **kwargs)
        N_columns = len(model.ds["subcolumn"])
        N_0 = fits_ds["N_0"].values
        mu = fits_ds["mu"].values
        lambdas = fits_ds["lambda"].values
        num_subcolumns = model.num_subcolumns
        total_hydrometeor = model.ds[frac_names] * model.ds[n_names]
        if np.isin(hyd_type, optional_ice_classes):
            if mie_for_ice:
                if hyd_type == "ci":
                    hyd_type_2_use = "ci"
                else:
                    hyd_type_2_use = "pi"  # Currently, all optional precipitating ice classes
                p_diam = instrument.mie_table[hyd_type_2_use]["p_diam"].values
                beta_p = instrument.mie_table[hyd_type_2_use]["beta_p"].values
                alpha_p = instrument.mie_table[hyd_type_2_use]["alpha_p"].values
            else:
                p_diam = instrument.scat_table[ice_lut][ice_diam_var].values
                beta_p = instrument.scat_table[ice_lut]["beta_p"].values
                alpha_p = instrument.scat_table[ice_lut]["alpha_p"].values
        else:  # Liquid classes (assuming only cl and pl)
            p_diam = instrument.mie_table[hyd_type]["p_diam"].values
            beta_p = instrument.mie_table[hyd_type]["beta_p"].values
            alpha_p = instrument.mie_table[hyd_type]["alpha_p"].values
        if (model.mcphys_scheme == "P3") & (hyd_type == "ci"):
            if instrument.instrument_str not in model.interpobj["single"].keys():
                model.interpobj["single"][instrument.instrument_str] = {}  # gen scattering interp objects
                scat_vars = ["beta_p", "alpha_p"]
                if model.p3_kws["use_hybrid_scat"] > 0:
                    scat_lut_vars = ["beta_p_eq_V", "alpha_p_eq_V"]
                else:
                    scat_lut_vars = ["beta_p", "alpha_p"]
                for lut_key, key in zip(scat_lut_vars, scat_vars):
                    model.interpobj["single"][instrument.instrument_str][key] = RegularGridInterpolator(
                        (instrument.scat_table['p3_ice']["Fr"].values,
                         instrument.scat_table['p3_ice']["rho_r"].values,
                         instrument.scat_table['p3_ice']["p_diam"].values),
                        instrument.scat_table['p3_ice'][lut_key].values, bounds_error=False)
                i_calc_kws = {"beta_p": model.interpobj["single"][instrument.instrument_str]["beta_p"],
                              "alpha_p": model.interpobj["single"][instrument.instrument_str]["alpha_p"],
                              "Fr_in": model.ds["Fr"].values,
                              "rho_r_in": model.ds["rho_r"].values,
                              "mie_for_ice": mie_for_ice,
                             }  # allocate only once
            calc_kws = i_calc_kws
        else:
            calc_kws = None
        sub_frac_arr = model.ds["strat_frac_subcolumns_%s" % hyd_type].values
        _calc_lidar = lambda x: _calc_strat_lidar_properties(
            x, N_0, lambdas, mu, p_diam, total_hydrometeor, hyd_type, sub_frac_arr, num_subcolumns,
            p_diam, beta_p, alpha_p, model.mcphys_scheme, calc_kws)
        if parallel:
            print("Doing parallel lidar calculations for %s" % hyd_type)
            if chunk is None:
                tt_bag = db.from_sequence(np.arange(0, Dims[1], 1))
                lists = tt_bag.map(_calc_lidar).compute()
            else:
                lists = []
                j = 0
                while j < Dims[1]:
                    if j + chunk >= Dims[1]:
                        ind_max = Dims[1]
                    else:
                        ind_max = j + chunk
                    print(" Processing columns %d-%d out of %d" % (j, ind_max, Dims[1]))
                    tt_bag = db.from_sequence(np.arange(j, ind_max, 1))
                    lists += tt_bag.map(_calc_lidar).compute()
                    j += chunk
        else:
            lists = [x for x in map(_calc_lidar, np.arange(0, Dims[1], 1))]
        beta_p_strat = np.stack([x[0] for x in lists], axis=1)
        alpha_p_strat = np.stack([x[1] for x in lists], axis=1)

        model.ds["sub_col_beta_p_%s_strat" % hyd_type][:, :, :] = beta_p_strat
        model.ds["sub_col_alpha_p_%s_strat" % hyd_type][:, :, :] = alpha_p_strat
        model.ds["sub_col_beta_p_%s_strat" % hyd_type] = \
            model.ds["sub_col_beta_p_%s_strat" % hyd_type].fillna(0)
        model.ds["sub_col_alpha_p_%s_strat" % hyd_type] = \
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type].fillna(0)
        model = accumulate_OD(model, False, z_values, hyd_type, OD_from_sfc, **kwargs)

        model.ds["sub_col_beta_p_tot_strat"] += model.ds["sub_col_beta_p_%s_strat" % hyd_type].fillna(0)
        model.ds["sub_col_alpha_p_tot_strat"] += model.ds["sub_col_alpha_p_%s_strat" % hyd_type].fillna(0)
        model.ds["sub_col_OD_tot_strat"] += model.ds["sub_col_OD_%s_strat" % hyd_type].fillna(0)

    return model


def calc_lidar_moments(instrument, model, is_conv,
                       OD_from_sfc=True, hyd_types=None, parallel=True, eta=1, chunk=None, mie_for_ice=False,
                       use_rad_logic=True, use_empiric_calc=False, **kwargs):
    """
    Calculates the lidar backscatter, extinction, and optical depth
    in a given column for the given lidar.

    NOTE:
    When starting a parallel task (in microphysics approach), it is recommended
    to wrap the top-level python script calling the EMC^2 processing ('lines_of_code')
    with the following command (just below the 'import' statements):
    
    .. code-block:: python
    
        if __name__ == “__main__”:
            lines_of_code

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    eta: float
        Multiple scattering coefficient.
    chunk: int or None
        The number of entries to process in one parallel loop. None will send all of
        the entries to the Dask worker queue at once. Sometimes, Dask will freeze if
        too many tasks are sent at once due to memory issues, so adjusting this number
        might be needed if that happens.
    mie_for_ice: bool
        If True, using bulk LUTs generated using solid ice spheres (Mie) calculations (e.g., consistent
        with the ice treatment in MG1 through MG2).
        Otherwise, using bulk LUTs that consider ice properties, e.g., the C6 8-column severly roughned
        aggregate for E3 or the m-D A-D for CESM and E3SMv1, consistent with the radiation schemes of
        these models.
        This parameter is set to `False` if `mcphys_scheme == P3` since ice shape is integrated into P3,
        which also affects mu, etc. and therefore the bulk LUT behavior in a similar manner to how the P3
        LUTs are integrated into EMC².
    use_rad_logic: bool
        When True using radiation scheme logic in calculations, which includes using
        the cloud fraction fields utilized in a model radiative scheme, as well as bulk
        scattering LUTs (effective radii dependent scattering variables). Otherwise, and
        only in the stratiform case, using the microphysics scheme logic, which includes
        the cloud fraction fields utilized by the model microphysics scheme and single
        particle scattering LUTs.
        NOTE: because of its single-particle calculation method, the microphysics
        approach is significantly slower than the radiation approach. Also, the cloud
        fraction logic in these  schemes does not necessarilytly fully overlap.
    use_empirical_calc: bool
        When True using empirical relations from literature for the fwd calculations
        (the cloud fraction still follows the scheme logic set by use_rad_logic).
    Additonal keyword arguments are passed into
    :py:func:`emc2.psd.calc_mu_lambda`.
    :py:func:`emc2.simulator.lidar_moments.accumulate_OD`.
    :py:func:`emc2.simulator.lidar_moments.calc_lidar_empirical`.
    :py:func:`emc2.simulator.lidar_moments.calc_lidar_bulk`.
    :py:func:`emc2.simulator.lidar_moments.calc_lidar_micro`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model dataset with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    if is_conv:
        cloud_str = "conv"
        cloud_str_full = "convective"
        if np.logical_and(not use_empiric_calc, not use_rad_logic):
            use_rad_logic = True  # Force rad scheme logic if in conv scheme
    else:
        cloud_str = "strat"
        cloud_str_full = "stratiform"

    if OD_from_sfc:
        OD_str = "model layer base"
    else:
        OD_str = "model layer top"

    if use_empiric_calc:
        scat_str = "Empirical (no utilized scattering database)"
    elif mie_for_ice:
        scat_str = "Mie"
    else:
        if model.rad_scheme_family == "CESM":
            scat_str = "m-D_A-D (D. Mitchell)"
        elif model.rad_scheme_family == "ModelE":
            scat_str = "C6"
        elif model.rad_scheme_family == "P3":
            scat_str = "m-D_A-D (D. Mitchell) for P3 (I. Silber)"

    if not instrument.instrument_class.lower() == "lidar":
        raise ValueError("Instrument must be a lidar!")

    if "%s_q_subcolumns_cl" % cloud_str not in model.ds.variables.keys():
        raise KeyError("Water mixing ratio in %s subcolumns must be generated first!" % cloud_str_full)

    p_field = model.p_field
    t_field = model.T_field
    z_field = model.z_field

    # Do unit conversions using pint - pressure in Pa, T in K, z in m
    p_temp = model.ds[p_field].values * getattr(ureg, model.ds[p_field].attrs["units"])
    p_values = p_temp.to('pascal').magnitude
    t_temp = quantity(model.ds[t_field].values, model.ds[t_field].attrs["units"])
    t_values = t_temp.to('celsius').magnitude
    z_temp = model.ds[z_field].values * getattr(ureg, model.ds[z_field].attrs["units"])
    z_values = z_temp.to('meter').magnitude
    del p_temp, t_temp, z_temp

    model = calc_theory_beta_m(model, instrument.wavelength)
    beta_m = np.tile(model.ds['sigma_180_vol'].values, (model.num_subcolumns, 1, 1))
    T = np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))

    t0 = time()
    if use_empiric_calc:
        print("Generating %s lidar variables using empirical formulation" % cloud_str_full)
        method_str = "Empirical"
        model = calc_lidar_empirical(instrument, model, is_conv, p_values, t_values, z_values,
                                     OD_from_sfc=OD_from_sfc, hyd_types=hyd_types, **kwargs)
    elif use_rad_logic:
        print("Generating %s lidar variables using radiation logic" % cloud_str_full)
        method_str = "Bulk (radiation logic)"
        model = calc_lidar_bulk(instrument, model, is_conv, p_values, z_values,
                                OD_from_sfc=OD_from_sfc, mie_for_ice=mie_for_ice, hyd_types=hyd_types, **kwargs)
    else:
        print("Generating %s lidar variables using microphysics logic (slowest processing)" % cloud_str_full)
        method_str = "LUTs (microphysics logic)"
        calc_lidar_micro(instrument, model, z_values, OD_from_sfc=OD_from_sfc,
                         hyd_types=hyd_types, mie_for_ice=mie_for_ice, parallel=parallel, chunk=chunk, **kwargs)

    for hyd_type in hyd_types:
        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].attrs["long_name"] = \
            "Particulate backscatter cross section from %s %s hydrometeors" % (cloud_str_full, hyd_type)
        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].attrs["units"] = r"$m^{-1} sr^{-1}$"
        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].attrs["Processing method"] = method_str
        model.ds["sub_col_beta_p_%s_%s" % (hyd_type, cloud_str)].attrs["Ice scattering database"] = scat_str
        model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].attrs["long_name"] = \
            "Particulate extinction cross section from %s %s hydrometeors" % (cloud_str_full, hyd_type)
        model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].attrs["units"] = r"$m^{-1}$"
        model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].attrs["Processing method"] = method_str
        model.ds["sub_col_alpha_p_%s_%s" % (hyd_type, cloud_str)].attrs["Ice scattering database"] = scat_str
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].attrs["long_name"] = \
            "Cumulative optical depth at %s from %s %s hydrometeors" % \
            (OD_str, cloud_str_full, hyd_type)
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].attrs["units"] = "1"
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].attrs["Processing method"] = method_str
        model.ds["sub_col_OD_%s_%s" % (hyd_type, cloud_str)].attrs["Ice scattering database"] = scat_str

    model.ds["sub_col_beta_p_tot_%s" % cloud_str].attrs["long_name"] = \
        "Backscatter coefficient from all %s hydrometeors" % cloud_str_full
    model.ds["sub_col_beta_p_tot_%s" % cloud_str].attrs["units"] = r"$m^{-1} sr^{-1}$"
    model.ds["sub_col_beta_p_tot_%s" % cloud_str].attrs["Processing method"] = method_str
    model.ds["sub_col_beta_p_tot_%s" % cloud_str].attrs["Ice scattering database"] = scat_str
    model.ds["sub_col_alpha_p_tot_%s" % cloud_str].attrs["long_name"] = \
        "Extinction coefficient from all %s hydrometeors" % cloud_str_full
    model.ds["sub_col_alpha_p_tot_%s" % cloud_str].attrs["units"] = r"$m^{-1}$"
    model.ds["sub_col_alpha_p_tot_%s" % cloud_str].attrs["Processing method"] = method_str
    model.ds["sub_col_alpha_p_tot_%s" % cloud_str].attrs["Ice scattering database"] = scat_str
    model.ds["sub_col_OD_tot_%s" % cloud_str].attrs["long_name"] = \
        "Cumulative optical depth at %s from all %s hydrometeors" % \
        (OD_str, cloud_str_full)
    model.ds["sub_col_OD_tot_%s" % cloud_str].attrs["units"] = "1"
    model.ds["sub_col_OD_tot_%s" % cloud_str].attrs["Processing method"] = method_str
    model.ds["sub_col_OD_tot_%s" % cloud_str].attrs["Ice scattering database"] = scat_str

    model.ds["sub_col_beta_att_tot_%s" % cloud_str] = (
        beta_m + model.ds["sub_col_beta_p_tot_%s" % cloud_str]) * \
        T * np.exp(-2 * eta * model.ds["sub_col_OD_tot_%s" % cloud_str])
    model.ds["sub_col_beta_att_tot_%s" % cloud_str].attrs["long_name"] = \
        "Total attenuated backscatter from all %s hydrometeors (including atmospheric extinction)" % cloud_str_full
    model.ds["sub_col_beta_att_tot_%s" % cloud_str].attrs["units"] = r"$m^{-1} sr^{-1}$"
    model.ds["sub_col_beta_att_tot_%s" % cloud_str].attrs["Processing method"] = method_str
    model.ds["sub_col_beta_att_tot_%s" % cloud_str].attrs["Ice scattering database"] = scat_str

    print("Done! total processing time = %.2fs" % (time() - t0))

    return model


def _calc_strat_lidar_properties(tt, N_0, lambdas, mu, p_diam, total_hydrometeor,
                                 hyd_type, sub_frac_arr, num_subcolumns, D, beta_p, alpha_p,
                                 mcphys_scheme, calc_kws=None):
    Dims = total_hydrometeor.shape
    beta_p_strat = np.zeros((num_subcolumns, Dims[2]))
    alpha_p_strat = np.zeros((num_subcolumns, Dims[2]))
    if (hyd_type == 'ci') & (mcphys_scheme == "P3"):
        if not calc_kws["mie_for_ice"]:
            tiled_arr = _set_p3_tiled_arrays(tt, calc_kws, Dims, p_diam, include_vt=False)
        zero_arr = np.zeros(p_diam.size)

    if tt % 50 == 0:
        print('Stratiform moment for class %s progress: %d/%d' % (hyd_type, tt, Dims[1]))
    for k in range(Dims[2]):
        if np.all(total_hydrometeor[:, tt, k] == 0):
            continue
        N_D = []
        if (hyd_type == 'ci') & (mcphys_scheme == "P3"):  # no N0 etc subcol dim (subcol q filter applies below)
            if not calc_kws["mie_for_ice"]:
                beta_p = tiled_arr["beta_p"][k, :]
                alpha_p = tiled_arr["alpha_p"][k, :]
            N_0_tmp = N_0[tt, k]
            lambda_tmp = lambdas[tt, k]
            mu_tmp = mu[tt, k]
            N_D_tmp = N_0_tmp * p_diam ** mu_tmp * np.exp(-lambda_tmp * p_diam)
        for i in range(num_subcolumns):
            if (hyd_type == 'ci') & (mcphys_scheme == "P3"):
                if sub_frac_arr[i, tt, k] == 1:
                    N_D.append(N_D_tmp)   # use subcol q to decide N_D population
                else:
                    N_D.append(zero_arr)  # no hydrometeors in subcolumn
            else:
                N_0_tmp = N_0[i, tt, k]
                lambda_tmp = lambdas[i, tt, k]
                mu_tmp = mu[i, tt, k]
                N_D.append(N_0_tmp * p_diam ** mu_tmp * np.exp(-lambda_tmp * p_diam))
        N_D = np.stack(N_D, axis=0)

        Calc_tmp = np.tile(beta_p, (num_subcolumns, 1)) * N_D
        beta_p_strat[:, k] = np.trapz(Calc_tmp, x=D, axis=1).astype('float64')
        Calc_tmp = np.tile(alpha_p, (num_subcolumns, 1)) * N_D
        alpha_p_strat[:, k] = np.trapz(Calc_tmp, x=D, axis=1).astype('float64')

    return beta_p_strat, alpha_p_strat
