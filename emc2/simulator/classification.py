import xarray as xr
import numpy as np

from ..core import Instrument, Model


def lidar_classify_phase(instrument, model, beta_p_phase_thresh=None,
                         convert_zeros_to_nan=False, remove_sub_detect=True):
    """
    Phase classification based on fixed thresholds of a lidar's LDR and
    tot beta_p variables.

    Parameters
    ----------
    instrument: Instrument
        The instrument to classify. The instrument must be a lidar.
    model: Model
        The model output to classify.
    beta_p_phase_thresh: list of dicts or None
        If a list, each element contains a dictionary with class name, class integer
        value (mask order starting at 1), LDR value bounds, and the corresponding beta_p threshold
        (thresholds are linearly interpolated in log10 scale between LDR values). In order for the
        method to operate properly, the list should be arranged from the lowest to
        highest beta_p threshold values for a given LDR, that is, (where i is the list element)
        beta_p[i+1 | LDR=x] >= beta_p[i | LDR=x]. Set class integer values of 1 or higher = clear
        (because of a very high beta_p value).
        When None, the default settings from the instrument object will be used (available only for
        beta resolving lidar classes).
    convert_zeros_to_nan: bool
        If True, saving the mask array as a float dtype (instead of uint8) and converting all
            zeros in the array to nans.
    remove_sub_detect: bool
        If True, remove hydrometeor-bearing grid cells with extinct lidar signal.

    Returns
    -------
    model: Model
        The model with the added simulated lidar phase classification mask.
    """

    if not instrument.instrument_class.lower() == "lidar":
        raise ValueError("Instrument must be a lidar!")

    if beta_p_phase_thresh is None:
        if not instrument.beta_p_phase_thresh:
            raise ValueError("no default threshold values for %s" % instrument.instrument_str)
        beta_p_phase_thresh = instrument.beta_p_phase_thresh

    if model.process_conv:
        mask_name_str = ["strat_phase_mask_%s" % instrument.instrument_str,
                         "conv_phase_mask_%s" % instrument.instrument_str,
                         "phase_mask_%s_all_hyd" % instrument.instrument_str]
        mask_long_name_str = ["%s phase classification mask (strat)" % instrument.instrument_str,
                              "%s phase classification mask (conv)" % instrument.instrument_str,
                              "%s phase classification mask (convective + stratiform)" %
                              instrument.instrument_str]
        LDR_fieldnames = ["sub_col_LDR_strat", "sub_col_LDR_conv", "sub_col_LDR_tot"]
        OD_fieldnames = ["sub_col_OD_tot_strat", "sub_col_OD_tot_conv", "sub_col_OD_tot"]
        beta_p_fieldnames = ["sub_col_beta_p_tot_strat", "sub_col_beta_p_tot_conv", "sub_col_beta_p_tot"]
    else:
        mask_name_str = ["phase_mask_%s_all_hyd" % instrument.instrument_str]
        mask_long_name_str = ["%s phase classification mask (convective + stratiform)" %
                              instrument.instrument_str]
        LDR_fieldnames = ["sub_col_LDR_tot"]
        OD_fieldnames = ["sub_col_OD_tot"]
        beta_p_fieldnames = ["sub_col_beta_p_tot"]

    for ii in range(len(mask_name_str)):
        mask_name = mask_name_str[ii]
        Class_legend = [""] * (len(beta_p_phase_thresh))
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["sub_col_LDR_tot"], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["sub_col_LDR_tot"], dtype=np.uint8)
        for class_type in range(len(beta_p_phase_thresh)):
            phase_mask = np.where(np.where(model.ds[beta_p_fieldnames[ii]].values > 0,
                                  np.log10(model.ds[beta_p_fieldnames[ii]].values), np.nan) >=
                                  np.interp(model.ds[LDR_fieldnames[ii]].values,
                                  beta_p_phase_thresh[class_type]["LDR"],
                                  np.log10(beta_p_phase_thresh[class_type]["beta_p"])),
                                  beta_p_phase_thresh[class_type]["class_ind"], phase_mask)
            Class_legend[beta_p_phase_thresh[class_type]["class_ind"] - 1] = \
                beta_p_phase_thresh[class_type]["class"]
        if remove_sub_detect:
            phase_mask = np.where(model.ds[OD_fieldnames[ii]] > instrument.ext_OD, 0, phase_mask)
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["sub_col_LDR_strat"].dims)
        model.ds[mask_name].attrs["long_name"] = mask_long_name_str[ii]
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = Class_legend

    return model


def radar_classify_phase(instrument, model, mask_height_rng=None, convert_zeros_to_nan=False):
    """
    Phase classification based on cloud occurrence and Ze_min threshold (equivalent
        to the KAZR-sounding dataset used in Silber et al., ACP, 2020).

    Parameters
    ----------
    instrument: Instrument
        The instrument to classify. The instrument must be a radar.
    model: Model
        The model output to classify.
    convert_zeros_to_nan: bool
        If True, saving the mask array as a float dtype (instead of uint8) and converting all
            zeros in the array to nans.
    mask_height_rng: tuple or list
        If None, using all altitudes. Otherwise, limiting to a specific range determined by
            a two-element tuple or list specifying the height range.

    Returns
    -------
    model: Model
        The model with the added simulated radar-sounding hydrometeor classification mask.
    """

    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Instrument must be a radar!")

    if model.process_conv:
        mask_name_str = ["strat_phase_mask_%s_sounding" % instrument.instrument_str,
                         "conv_phase_mask_%s_sounding" % instrument.instrument_str,
                         "phase_mask_%s_sounding_all_hyd" % instrument.instrument_str]
        mask_long_name_str = [
            "%s-sounding cloud and precipitation detection output (strat)" % instrument.instrument_str,
            "%s-sounding cloud and precipitation detection output (conv)" % instrument.instrument_str,
            "%s-sounding cloud and precipitation detection output (convective + stratiform)" %
            instrument.instrument_str]
        Ze_fieldnames = ["sub_col_Ze_att_tot_strat", "sub_col_Ze_att_tot_conv", "sub_col_Ze_att_tot"]
        cld_exist_cond = [model.ds["strat_frac_subcolumns_cl"].values,
                          model.ds["conv_frac_subcolumns_cl"].values,
                          np.logical_or(model.ds["strat_frac_subcolumns_cl"].values,
                                        model.ds["conv_frac_subcolumns_cl"].values)]
    else:
        mask_name_str = ["phase_mask_%s_sounding_all_hyd" % instrument.instrument_str]
        mask_long_name_str = [
            "%s-sounding cloud and precipitation detection output (convective + stratiform)" %
            instrument.instrument_str]
        Ze_fieldnames = ["sub_col_Ze_att_tot"]
        cld_exist_cond = [model.ds["strat_frac_subcolumns_cl"].values]

    for ii in range(len(mask_name_str)):
        mask_name = mask_name_str[ii]
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["strat_frac_subcolumns_cl"], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["strat_frac_subcolumns_cl"], dtype=np.uint8)
        phase_mask = np.where(model.ds[Ze_fieldnames[ii]].values >=
                              np.tile(model.ds['Ze_min'].values,
                                      (model.num_subcolumns, 1, 1)), 3, phase_mask)  # Precip
        phase_mask = np.where(np.logical_and(cld_exist_cond[ii], phase_mask != 3), 1, phase_mask)  # Cloud
        phase_mask = np.where(np.logical_and(cld_exist_cond[ii], phase_mask == 3), 2, phase_mask)  # Mixed
        if mask_height_rng is not None:
            phase_mask = np.where(
                np.logical_or(np.tile(model.ds[model.z_field], (model.num_subcolumns, 1, 1)) < mask_height_rng[0],
                              np.tile(model.ds[model.z_field], (model.num_subcolumns, 1, 1)) > mask_height_rng[1]),
                0, phase_mask)
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["strat_frac_subcolumns_cl"].dims)
        model.ds[mask_name].attrs["long_name"] = mask_long_name_str[ii]
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = ["Cloud", "Mixed", "Precip"]

    return model


def lidar_emulate_cosp_phase(instrument, model, eta=0.7, OD_from_sfc=True, phase_disc_curve=None,
                             atb_cross_coeff=None, cloud_SR_thresh=5., undef_SR_thresh=30.,
                             inc_precip_hyd_atb=True, convert_zeros_to_nan=False, output_ATB=False,
                             hyd_types=None):
    """
    Phase classification method to emulate COSP based on attenuated total backscatter
        (ATB) analysis following Cesana and Chepfer (2013).

    Parameters
    ----------
    instrument: Instrument
        The instrument to classify. The instrument must be a lidar.
    model: Model
        The model output to classify.
    eta: float
        Multiple scattering coefficient.
    OD_from_sfc: bool
        If True, optical depth will be calculated from the surface. If False,
        optical depth will be calculated from the top of the atmosphere.
    phase_disc_curves: list or None
        Phase discrimination curve polynomial coefficients (above - liquid, below - ice).
        When None, the default settings from this method will be used (following Cesana
        and Chepfer (2013).
    atb_cross_coeff: dict or None.
        Dictionary of polynomial coefficients for the estimation of ATB_cross for each
        hydrometeor type (dict keys).
        When None, the default coefficients from Cesana and Chepfer (2013) / COSP are used.
    cloud_SR_thresh: float
        Scattering ratio threshold for hydrometeor detection.
    undef_SR_thresh: float
        Scattering ratio threshold for the undefined phase (layers below a layer with
        SR values above this threshold are all set to set to "undefined").
    inc_precip_hyd_atb: bool
        If True, include precipitating classes in the calculation of ATB if specified in hyd_types
        or if hyd_types is None
    convert_zeros_to_nan: bool
        If True, saving the mask array as a float dtype (instead of uint8) and converting all
            zeros in the array to nans.
    output_ATB: bool
        If True, save the ATB and scattering ratio fields for each cloud type as well as for
        all hydrometeors.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """
    hyd_types = model.set_hyd_types(hyd_types)

    if model.process_conv:
        cld_classes = ["conv", "strat"]
    else:
        cld_classes = ["strat"]

    if instrument.instrument_str != "HSRL":
        raise ValueError("Instrument must be the 532 nm HSRL (to match CALIOP's operating wavelength)")

    if phase_disc_curve is None:
        phase_disc_curve = [9.032e3, -2.136e3, 173.394, -3.951, 0.256, -9.478e-4]

    if atb_cross_coeff is None:
        atb_cross_coeff = {'liq': [0.4099, 0.009, 0], 'ice': [0.2904, 0]}

    if inc_precip_hyd_atb:
        hyd_classes = {"liq": ["cl", "pl"], "ice": ["ci", "pi"]}
    else:
        hyd_classes = {"liq": ["cl"], "ice": ["ci"]}

    ATB_mol = np.tile(model.ds['sigma_180_vol'].values * (1. + 0.0284) * model.ds['tau'].values,
                      (model.num_subcolumns, 1, 1))

    beta_p_allhyd = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
    beta_p_cross_allhyd = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
    OD_allhyd = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
    for cloud_class in cld_classes:
        mask_name = "%s_COSP_phase_mask" % cloud_class
        phase_mask = np.zeros_like(model.ds["strat_q_subcolumns_cl"], dtype=np.uint8)
        ATB_co = {}
        ATB_cross = {}
        OD = {}
        beta_p = {}
        beta_p_cross = {}
        for hyd_class in hyd_classes.keys():
            OD[hyd_class] = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
            beta_p[hyd_class] = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
            for hyd_type in hyd_classes[hyd_class]:
                if hyd_type not in hyd_types:
                    print("'%s' not in hyd_types = %s; excluding from COSP calculations" % (hyd_type, hyd_types))
                    continue
                beta_p[hyd_class] += np.nan_to_num(
                    model.ds['sub_col_beta_p_%s_%s' % (hyd_type, cloud_class)].values)
                OD[hyd_class] += np.nan_to_num(model.ds['sub_col_OD_%s_%s' % (hyd_type, cloud_class)].values)
            ATB_co[hyd_class] = (np.tile(model.ds['sigma_180_vol'].values, (model.num_subcolumns, 1, 1)) +
                                 beta_p[hyd_class]) * np.tile(
                model.ds['tau'].values, (model.num_subcolumns, 1, 1)) * \
                np.exp(-2 * eta * OD[hyd_class])
            ATB_cross[hyd_class] = np.polyval(atb_cross_coeff[hyd_class], ATB_co[hyd_class] * 1e3) / 1e3
            beta_p_cross[hyd_class] = ATB_cross[hyd_class] / np.exp(-2 * eta * OD[hyd_class]) / \
                np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1)) - \
                np.tile(model.ds['sigma_180_vol'].values * (0.0284 / (1 + 0.0284)), (model.num_subcolumns, 1, 1))
            beta_p_allhyd += beta_p[hyd_class]
            beta_p_cross_allhyd += beta_p_cross[hyd_class]
            OD_allhyd += OD[hyd_class]
        ATB_tot = (beta_p["liq"] + beta_p_cross["liq"] + beta_p["ice"] + beta_p_cross["ice"] +
                   np.tile(model.ds['sigma_180_vol'].values * (1. + 0.0284), (model.num_subcolumns, 1, 1))) * \
            np.exp(-2 * eta * (OD["liq"] + OD["ice"])) * \
            np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
        ATB_cross_tot = (beta_p_cross["liq"] + beta_p_cross["ice"] + np.tile(model.ds['sigma_180_vol'].values *
                         (0.0284 / (1. + 0.0284)), (model.num_subcolumns, 1, 1))) * \
            np.exp(-2 * eta * (OD["liq"] + OD["ice"])) * \
            np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
        del beta_p_cross, ATB_cross, ATB_co, OD, beta_p

        # Begin cloud detection and phase classification
        SR = ATB_tot / ATB_mol
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["strat_q_subcolumns_cl"], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["strat_q_subcolumns_cl"], dtype=np.uint8)
        phase_mask = np.where(SR > cloud_SR_thresh, 1, phase_mask)
        phase_mask = np.where(np.logical_and(ATB_cross_tot > np.polyval(phase_disc_curve, ATB_tot * 1e3) / 1e3,
                              phase_mask == 1), 2, phase_mask)
        if OD_from_sfc:
            reflective_mask = np.cumsum(SR > undef_SR_thresh, axis=2)
        else:
            reflective_mask = np.flip(np.cumsum(np.flip(SR, axis=2) > undef_SR_thresh, axis=2), axis=2)
        reflective_mask = np.where(np.logical_and(SR > undef_SR_thresh, reflective_mask == 1),
                                   0, reflective_mask)

        phase_mask = np.where(np.logical_and(np.tile(model.ds[model.T_field].values,
                                                     (model.num_subcolumns, 1, 1)) > 273.15,
                              phase_mask > 0), 1, phase_mask)
        phase_mask = np.where(np.logical_and(np.tile(model.ds[model.T_field].values,
                                                     (model.num_subcolumns, 1, 1)) < 233.15,
                              phase_mask > 0), 2, phase_mask)
        phase_mask = np.where(np.logical_and(reflective_mask > 0, phase_mask > 0), 3, phase_mask)
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["strat_q_subcolumns_cl"].dims)
        model.ds[mask_name].attrs["long_name"] = "COSP emulation phase classification mask (%s)" % cloud_class
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = ["liquid", "ice", "undef"]

        # save ATB and SR fields for stratiform only
        if output_ATB is True:
            model.ds["COSP_ATBtot_%s" % cloud_class] = xr.DataArray(
                ATB_tot, dims=model.ds["strat_q_subcolumns_cl"].dims)
            model.ds["COSP_ATBtot_%s" % cloud_class].attrs["long_name"] = \
                "COSP emulation ATB_tot for %s clouds" % cloud_class
            model.ds["COSP_ATBtot_%s" % cloud_class].attrs["units"] = r"$m^{-1}\ sr^{-1}$"

            model.ds["COSP_ATBcross_%s" % cloud_class] = xr.DataArray(
                ATB_cross_tot, dims=model.ds["strat_q_subcolumns_cl"].dims)
            model.ds["COSP_ATBcross_%s" % cloud_class].attrs["long_name"] = \
                "COSP emulation ATB_cross for %s clouds" % cloud_class
            model.ds["COSP_ATBcross_%s" % cloud_class].attrs["units"] = r"$m^{-1}\ sr^{-1}$"
            model.ds["COSP_SR_%s" % cloud_class] = xr.DataArray(
                SR, dims=model.ds["strat_q_subcolumns_cl"].dims)
            model.ds["COSP_SR_%s" % cloud_class].attrs["long_name"] = \
                "COSP emulation scattering ratio for %s clouds" % cloud_class
            model.ds["COSP_SR_%s" % cloud_class].attrs["units"] = r"$m^{-1}\ sr^{-1}$"

    # determine phase_mask for all hydrometeors
    ATB_tot_allhyd = (beta_p_allhyd + beta_p_cross_allhyd +
                      np.tile(model.ds['sigma_180_vol'].values * (1. + 0.0284), (model.num_subcolumns, 1, 1))) * \
        np.exp(-2 * eta * OD_allhyd) * \
        np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
    ATB_cross_allhyd = (beta_p_cross_allhyd + np.tile(model.ds['sigma_180_vol'].values *
                        (0.0284 / (1. + 0.0284)), (model.num_subcolumns, 1, 1))) * \
        np.exp(-2 * eta * OD_allhyd) * \
        np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))

    # Begin cloud detection and phase classification
    mask_name = "COSP_phase_mask_all_hyd"
    SR_allhyd = ATB_tot_allhyd / ATB_mol
    if convert_zeros_to_nan:
        phase_mask = np.zeros_like(model.ds["strat_q_subcolumns_cl"], dtype=np.float)
    else:
        phase_mask = np.zeros_like(model.ds["strat_q_subcolumns_cl"], dtype=np.uint8)
    phase_mask = np.where(SR_allhyd > cloud_SR_thresh, 1, phase_mask)
    phase_mask = np.where(
        np.logical_and(ATB_cross_allhyd > np.polyval(phase_disc_curve, ATB_tot_allhyd * 1e3) / 1e3,
                       phase_mask == 1), 2, phase_mask)
    if OD_from_sfc:
        reflective_mask = np.cumsum(SR_allhyd > undef_SR_thresh, axis=2)
    else:
        reflective_mask = np.flip(np.cumsum(np.flip(SR_allhyd, axis=2) > undef_SR_thresh, axis=2), axis=2)
    reflective_mask = np.where(np.logical_and(SR_allhyd > undef_SR_thresh, reflective_mask == 1),
                               0, reflective_mask)

    phase_mask = np.where(np.logical_and(np.tile(model.ds[model.T_field].values,
                                                 (model.num_subcolumns, 1, 1)) > 273.15,
                          phase_mask > 0), 1, phase_mask)
    phase_mask = np.where(np.logical_and(np.tile(model.ds[model.T_field].values,
                                                 (model.num_subcolumns, 1, 1)) < 233.15,
                          phase_mask > 0), 2, phase_mask)
    phase_mask = np.where(np.logical_and(reflective_mask > 0, phase_mask > 0), 3, phase_mask)
    if convert_zeros_to_nan:
        phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

    model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["strat_q_subcolumns_cl"].dims)
    model.ds[mask_name].attrs["long_name"] = "COSP emulation phase classification mask (convective + stratiform)"
    model.ds[mask_name].attrs["units"] = "Unitless"
    model.ds[mask_name].attrs["legend"] = ["liquid", "ice", "undef"]
    if output_ATB is True:
        model.ds["COSP_ATBtot_all_hyd"] = xr.DataArray(
            ATB_tot_allhyd, dims=model.ds["strat_q_subcolumns_cl"].dims)
        model.ds["COSP_ATBtot_all_hyd"].attrs["long_name"] = \
            "COSP emulation ATB_tot (convective + stratiform)"
        model.ds["COSP_ATBtot_all_hyd"].attrs["units"] = r"$m^{-1}\ sr^{-1}$"

        model.ds["COSP_ATBcross_all_hyd"] = xr.DataArray(
            ATB_cross_allhyd, dims=model.ds["strat_q_subcolumns_cl"].dims)
        model.ds["COSP_ATBcross_all_hyd"].attrs["long_name"] = \
            "COSP emulation ATB_cross (convective + stratiform)"
        model.ds["COSP_ATBcross_all_hyd"].attrs["units"] = r"$m^{-1}\ sr^{-1}$"
        model.ds["COSP_SR_all_hyd"] = xr.DataArray(
            SR_allhyd, dims=model.ds["strat_q_subcolumns_cl"].dims)
        model.ds["COSP_SR_all_hyd"].attrs["long_name"] = \
            "COSP emulation scattering ratio (convective + stratiform)"
        model.ds["COSP_SR_all_hyd"].attrs["units"] = r"$m^{-1}\ sr^{-1}$"

    return model


def calculate_phase_ratio(model, variable, mask_class, mask_allhyd=None, mass_pr=False,
                          mpr_subcolmod=False, hyd_types=None):
    """
    Calculate time-height phase ratio field of subcolumn hydrometeor mask for a given class(es).

    Parameters
    ----------
    model: Model
        The model output to classify.
    variable: str
        The mask variable to process and plot.
    mask_class: int or list
        value(s) corresponding to hydrometeor class(es) to calculate the phase ratio for
        (numerator). Phase ratio is calculated relative to the sum of all hydrometeor classes
        in subcolumns (defined by mask_allhyd) per time-height grid cell.
    mask_allhyd: int, list, or None
        value(s) corresponding to all hydrometeor class(es) to calculate the phase ratio with
        (denominator). If None, using all non-zero values.
    mass_pr: bool
        If True, calcuating the mass phase ratio from the model output where hydrometeors exist
        in the classification mask. Otherwise, the frequency phase ratio (from all subcolumns).
    mpr_subcolmod: bool
        If True, doing subcolumn-based MPR calculation. Otherwise, simply using the model output
        mixing ratio data.
    hyd_types: list or None
        list of hydrometeor names to include in calcuation. using default Model subclass types if None.

    Returns
    -------
    model: Model
        The model with the added phase ratio field
    """
    hyd_types = model.set_hyd_types(hyd_types)

    if model.process_conv:
        cld_classes = ["conv", "strat"]
    else:
        cld_classes = ["strat"]

    if mass_pr is True:
        liq_classes = [x for x in ["cl", "pl"] if x in hyd_types]
        ice_classes = [x for x in ["ci", "pi"] if x in hyd_types]
        if mpr_subcolmod is True:
            mass_subcol_liq = np.zeros_like(model.ds["strat_frac_subcolumns_cl"], dtype=np.float)
            mass_subcol_ice = np.zeros_like(model.ds["strat_frac_subcolumns_cl"], dtype=np.float)
            for cloud_class in cld_classes:
                for hyd_class in liq_classes:
                    mass_subcol_liq += np.where(
                        model.ds[variable] > 0,
                        model.ds["%s_q_subcolumns_%s" % (cloud_class, hyd_class)], 0)
                for hyd_class in ice_classes:
                    mass_subcol_ice += np.where(
                        model.ds[variable] > 0,
                        model.ds["%s_q_subcolumns_%s" % (cloud_class, hyd_class)], 0)
            PR = mass_subcol_liq / (mass_subcol_liq + mass_subcol_ice)
            PR_sum = np.nansum(mass_subcol_liq, axis=0) / (np.nansum(mass_subcol_liq, axis=0) +
                                                           np.nansum(mass_subcol_ice, axis=0))
            model.ds[variable + "_mpr"] = xr.DataArray(PR, dims=model.ds["strat_frac_subcolumns_cl"].dims)
            model.ds[variable + "_mpr"].attrs["long_name"] = variable + "mass phase ratio"
            model.ds[variable + "_mpr"].attrs["units"] = ""
            model.ds[variable + "_mpr_sum"] = xr.DataArray(PR_sum, dims=model.ds[model.T_field].dims)
            model.ds[variable + "_mpr_sum"].attrs["long_name"] = variable + \
                "mass phase ratio with q summed over all hydrometeor containing mask grid cells"
            model.ds[variable + "_mpr_sum"].attrs["units"] = ""
        else:
            mass_liq = np.zeros_like(model.ds[model.q_names_stratiform["cl"]], dtype=np.float)
            mass_ice = np.zeros_like(model.ds[model.q_names_stratiform["cl"]], dtype=np.float)
            for hyd_class in liq_classes:
                mass_liq += model.ds[model.q_names_stratiform[hyd_class]].values
            for hyd_class in ice_classes:
                mass_ice += model.ds[model.q_names_stratiform[hyd_class]].values
            PR_sum = mass_liq / (mass_liq + mass_ice)
            model.ds["mpr_q"] = xr.DataArray(PR_sum, dims=model.ds[model.T_field].dims)
            model.ds["mpr_q"].attrs["long_name"] = variable + \
                "mass phase ratio using model output fields"
            model.ds["mpr_q"].attrs["units"] = ""
    else:
        mask_array = model.ds[variable].values.astype(np.int8)
        if mask_allhyd is None:
            mask_allhyd = [x for x in range(1, np.nanmax(mask_array) + 1)]

        numer_counts = np.nansum(np.isin(mask_array, mask_class), axis=0)
        denom_counts = np.nansum(np.isin(mask_array, mask_allhyd), axis=0)
        denom_counts = np.where(denom_counts == 0, np.nan, denom_counts)
        PR = np.nan_to_num(numer_counts) / denom_counts
        model.ds[variable + "_fpr"] = xr.DataArray(PR, dims=model.ds[model.T_field].dims)
        model.ds[variable + "_fpr"].attrs["long_name"] = variable + " frequency phase ratio"
        model.ds[variable + "_fpr"].attrs["units"] = ""
        model.ds[variable + "_fpr"].attrs["hyd_numer_val"] = mask_class
        model.ds[variable + "_fpr"].attrs["hyd_denom_val"] = mask_allhyd

    return model
