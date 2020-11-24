import xarray as xr
import numpy as np

from ..core import Instrument, Model


def lidar_classify_phase(instrument, model, beta_p_phase_thresh=None,
                        convert_zeros_to_nan=False):
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
        If a list, each index contains a dictionaly with class name, class integer
        value (mask order), LDR value bounds, and the corresponding beta_p threshold
        (thresholds are linearly interpolated between LDR values). In order for the
        method to operate properly, the list should be arranged from the lowest to
        highest beta_p threshold values for a given LDR, that is,
        beta_p[i+1 | LDR=x] >= beta_p[i | LDR=x]. class integer values of 0 = clear.
        When None, the default settings from the instrument will be used.
    convert_zeros_to_nan: bool
        If True, saving the mask array as a float dtype (instead of uint8) and converting all
            zeros in the array to nans.

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

    for cloud_class in ["conv", "strat"]:
        mask_name = "%s_phase_mask_%s" % (cloud_class, instrument.instrument_str)
        Class_legend = [""] * (len(beta_p_phase_thresh))
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.uint8)
        for class_type in range(len(beta_p_phase_thresh)):
            phase_mask = np.where(model.ds["sub_col_beta_p_tot_%s" % cloud_class].values >= \
                np.interp(model.ds["LDR_%s" % cloud_class].values, beta_p_phase_thresh[class_type]["LDR"], 
                beta_p_phase_thresh[class_type]["beta_p"]),
                beta_p_phase_thresh[class_type]["class_ind"], phase_mask)
            Class_legend[beta_p_phase_thresh[class_type]["class_ind"]-1] = \
                beta_p_phase_thresh[class_type]["class"]
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["LDR_%s" % cloud_class].dims)
        model.ds[mask_name].attrs["long_name"] = "%s phase classification mask" % instrument.instrument_str
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
        The instrument to classify. The instrument must be a lidar.
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

    for cloud_class in ["strat"]:
        mask_name = "%s_phase_mask_%s_sounding" % (cloud_class, instrument.instrument_str)
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["mu"], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["mu"], dtype=np.uint8)
        phase_mask = np.where(model.ds["sub_col_Ze_att_tot_%s" % cloud_class].values >=
            np.tile(model.ds['Ze_min'].values, (model.num_subcolumns, 1, 1)),
            2, phase_mask)
        phase_mask = np.where(np.logical_and(model.ds["%s_frac_subcolumns_cl" % cloud_class].values == True,
            phase_mask != 2), 1, phase_mask)
        phase_mask = np.where(np.logical_and(model.ds["%s_frac_subcolumns_cl" % cloud_class].values == True,
            phase_mask == 2), 3, phase_mask)
        if mask_height_rng is not None:
            phase_mask = np.where(np.logical_or(np.tile(model.ds[model.z_field], (model.num_subcolumns, 1, 1)) <
                            mask_height_rng[0],
                            np.tile(model.ds[model.z_field], (model.num_subcolumns, 1, 1)) > mask_height_rng[1]),
                            0, phase_mask)
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["mu"].dims)
        model.ds[mask_name].attrs["long_name"] = "%s-sounding cloud and precipitation detection output" % instrument.instrument_str
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = ["Cloud", "Precip", "Mixed"]

    return model


def lidar_emulate_cosp_phase(instrument, model, eta=0.7, OD_from_sfc=True, phase_disc_curve=None,
                            atb_cross_coeff=None, cloud_SR_thresh=5., undef_SR_thresh=30.,
                            inc_precip_hyd_atb=True, convert_zeros_to_nan=False):
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
        If True, include precipitating classes in the calculation of ATB (not specified in
            the referencedpaper, so likely false in actual COSP).
    convert_zeros_to_nan: bool
        If True, saving the mask array as a float dtype (instead of uint8) and converting all
            zeros in the array to nans.

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """

    if instrument.instrument_str is not "HSRL":
                raise Valuerror("Instrument must be the 532 nm HSRL (to match CALIOP's operating wavelength)")

    if phase_disc_curve is None:
        phase_disc_curve = [9.032e3, -2.136e3, 173.394, -3.951, 0.256, -9.478e-4]

    if atb_cross_coeff is None:
        atb_cross_coeff = {'liq': [0.4099, 0.009, 0], 'ice': [0.2904, 0]}

    if inc_precip_hyd_atb:
        hyd_groups = {"liq": ["cl", "pl"], "ice": ["ci", "pi"]}
    else:
        hyd_groups = {"liq": ["cl"], "ice": ["ci"]}

    ATB_mol = np.tile(model.ds['sigma_180_vol'].values * (1.+0.0284) * model.ds['tau'].values, \
               (model.num_subcolumns, 1, 1))

    for cloud_class in ["conv", "strat"]:
        mask_name = "%s_COSP_phase_mask" % cloud_class
        Class_legend = ["liquid", "ice", "undefined"]
        phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.uint8)
        ATB_co = {}
        ATB_cross = {}
        OD = {}
        beta_p = {}
        beta_p_cross = {}
        ATB = {}
        for hyd_class in hyd_groups.keys():
            OD[hyd_class] = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
            beta_p[hyd_class] = np.zeros_like(model.ds['sub_col_beta_p_tot_strat'].values)
            for class_name in hyd_groups[hyd_class]:
                beta_p[hyd_class] += model.ds['sub_col_beta_p_%s_%s' % (class_name, cloud_class)].values
                OD[hyd_class] += model.ds['sub_col_OD_%s_%s' % (class_name, cloud_class)].values
            ATB_co[hyd_class] =  (np.tile(model.ds['sigma_180_vol'].values, (model.num_subcolumns, 1, 1)) + \
                    beta_p[hyd_class]) * np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1)) * \
                    np.exp(-2 * eta * OD[hyd_class])
            ATB_cross[hyd_class] = np.polyval(atb_cross_coeff[hyd_class], ATB_co[hyd_class] * 1e3) / 1e3
            beta_p_cross[hyd_class] = ATB_cross[hyd_class] / np.exp(-2 * eta * OD[hyd_class]) / \
                    np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1)) - \
                    np.tile(model.ds['sigma_180_vol'].values * (0.0284/(1+0.0284)), (model.num_subcolumns, 1, 1))
            ATB[hyd_class] = (beta_p[hyd_class] + beta_p_cross[hyd_class] + \
                    np.tile(model.ds['sigma_180_vol'].values * (1.+0.0284), (model.num_subcolumns, 1, 1))) * \
                    np.exp(-2 * eta * OD[hyd_class]) * \
                    np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
        ATB_tot = (beta_p["liq"] + beta_p_cross["liq"] + beta_p["ice"] + beta_p_cross["ice"] + \
                    np.tile(model.ds['sigma_180_vol'].values * (1.+0.0284), (model.num_subcolumns, 1, 1))) * \
                    np.exp(-2 * eta * (OD["liq"] + OD["ice"])) * \
                    np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
        ATB_cross_tot = (beta_p_cross["liq"] + beta_p_cross["ice"] + np.tile(model.ds['sigma_180_vol'].values * \
                    (0.0284/(1.+0.0284)), (model.num_subcolumns, 1, 1))) * \
                    np.exp(-2 * eta * (OD["liq"] + OD["ice"])) * \
                    np.tile(model.ds['tau'].values, (model.num_subcolumns, 1, 1))
        del ATB, beta_p_cross, ATB_cross, ATB_co, OD, beta_p

        # Begin cloud detection and phase classification
        SR = ATB_tot / ATB_mol
        if convert_zeros_to_nan:
            phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.float)
        else:
            phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.uint8)
        phase_mask = np.where(SR > cloud_SR_thresh, 1, phase_mask)
        phase_mask = np.where(np.logical_and(ATB_cross_tot > np.polyval(phase_disc_curve, ATB_tot * 1e3) / 1e3,
                phase_mask == 1), 2, phase_mask)
        if OD_from_sfc:
            reflective_mask = np.cumsum(SR > undef_SR_thresh, axis=2)
        else:
            reflective_mask = np.flip(np.cumsum(np.flip(SR, axis=2) > undef_SR_thresh, axis=2), axis=2)
        reflective_mask = np.where(np.logical_and(SR > undef_SR_thresh, reflective_mask == 1),
            0, reflective_mask)

        phase_mask = np.where(np.logical_and(np.tile(model.ds['t'].values, (model.num_subcolumns, 1, 1)) > 273.15,
                        phase_mask > 0), 1, phase_mask)
        phase_mask = np.where(np.logical_and(np.tile(model.ds['t'].values, (model.num_subcolumns, 1, 1)) < 233.15,
                        phase_mask > 0), 2, phase_mask)
        phase_mask = np.where(np.logical_and(reflective_mask > 0, phase_mask > 0), 3, phase_mask)
        if convert_zeros_to_nan:
            phase_mask = np.where(phase_mask == 0, np.nan, phase_mask)

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["LDR_%s" % cloud_class].dims)
        model.ds[mask_name].attrs["long_name"] = "COSP emulation phase classification mask"
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = ["liquid", "ice", "undef"]

        # save ATB and SR fields for stratiform only
        if cloud_class is "strat":
            model.ds["COSP_ATBtot_strat"] = xr.DataArray(ATB_tot, dims=model.ds["LDR_%s" % cloud_class].dims)
            model.ds["COSP_ATBtot_strat"].attrs["long_name"] = "COSP emulation ATB_tot"
            model.ds["COSP_ATBtot_strat"].attrs["units"] = "m^-1sr^-1"

            model.ds["COSP_ATBcross_strat"] = xr.DataArray(ATB_cross_tot, dims=model.ds["LDR_%s" % cloud_class].dims)
            model.ds["COSP_ATBcross_strat"].attrs["long_name"] = "COSP emulation ATB_cross"
            model.ds["COSP_ATBcross_strat"].attrs["units"] = "m^-1sr^-1"

            model.ds["COSP_SR_strat"] = xr.DataArray(SR, dims=model.ds["LDR_%s" % cloud_class].dims)
            model.ds["COSP_SR_strat"].attrs["long_name"] = "COSP emulation scattering ratio"
            model.ds["COSP_SR_strat"].attrs["units"] = "m^-1sr^-1"

    return model

def calculate_phase_ratio(model, variable, mask_class, mask_all_hyd=None):
    """
    calculate time-height phase ratio field of subcolumn hydrometeor mask for a given class(es).

    Parameters
    ----------
    model: Model
        The model output to classify.
    variable: str
        The mask variable to process and plot.
    mask_class: int or list
        value(s) corresponding to hydrometeor class(es) to calculate the phase ratio for
        (numerator). Phase ratio is calculated relative to the sum of all hydrometeor classes
        in subcolumns (defined by mask_all_hyd) per time-height grid cell.
    mask_all_hyd: int, list, or None
        value(s) corresponding to all hydrometeor class(es) to calculate the phase ratio with
        (denominator). If None, using all non-zero values.

    Returns
    -------
    model: Model
        The model with the added phase ratio field
    """
    if mask_all_hyd is None:
       mask_all_hyd = [x for x in range(1, np.nanmax(model.ds[variable].values)+1)]

    numer_counts = np.nansum(np.isin(model.ds[variable].values, mask_class), axis=0)
    denom_counts = np.nansum(np.isin(model.ds[variable].values, mask_all_hyd), axis=0)
    denom_counts = np.where(denom_counts == 0, np.nan, denom_counts)
    PR = numer_counts / denom_counts

    model.ds[variable+"_pr"] = xr.DataArray(PR, dims=model.ds[model.T_field].dims)
    model.ds[variable+"_pr"].attrs["long_name"] = variable + "phase ratio"
    model.ds[variable+"_pr"].attrs["units"] = ""
    model.ds[variable+"_pr"].attrs["hyd_numer_val"] = mask_class
    model.ds[variable+"_pr"].attrs["hyd_denom_val"] = mask_all_hyd

    return model
