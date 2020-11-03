import xarray as xr
import numpy as np

from ..core import Instrument, Model


def lidar_classify_phase(instrument, model, beta_p_phase_thresh=None):
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

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
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
        phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.uint8)
        for class_type in range(len(beta_p_phase_thresh)):
            phase_mask = np.where(model.ds["sub_col_beta_p_tot_%s" % cloud_class].values >= \
                np.interp(model.ds["LDR_%s" % cloud_class].values, beta_p_phase_thresh[class_type]["LDR"], 
                beta_p_phase_thresh[class_type]["beta_p"]),
                beta_p_phase_thresh[class_type]["class_ind"], phase_mask)
            Class_legend[beta_p_phase_thresh[class_type]["class_ind"]-1] = \
                beta_p_phase_thresh[class_type]["class"]

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["LDR_%s" % cloud_class].dims)
        model.ds[mask_name].attrs["long_name"] = "%s phase classification mask" % instrument.instrument_str
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = Class_legend

    return model

def lidar_emulate_cosp_phase(instrument, model, phase_disc_curves=None):
    """
    Phase classification method to emulate COSP based on attenuated total backscatter
        (ATB) analysis following Cesana and Chepfer (2013).

    Parameters
    ----------
    instrument: Instrument
        The instrument to classify. The instrument must be a lidar.
    model: Model
        The model output to classify.
    beta_p_phase_thresh: list of dicts or None
        If a list, each index contains a dictionaly with class name, class integer
        value (mask order), and the phase discrimination polynomial coefficients.
        In order for the method to operate properly, the list should be arranged
        from the lowest to highest ATB_cross per ATB_tot, that is,
        ATB_cross[i+1 | ATB_tot=x] >= ATB_cross[i | ATB_tot=x]. 
        class integer values of 0 = clear.
        When None, the default settings from the instrument will be used.

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
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
        phase_mask = np.zeros_like(model.ds["LDR_%s" % cloud_class], dtype=np.uint8)
        for class_type in range(len(beta_p_phase_thresh)):
            phase_mask = np.where(model.ds["sub_col_beta_p_tot_%s" % cloud_class].values >= \
                np.interp(model.ds["LDR_%s" % cloud_class].values, beta_p_phase_thresh[class_type]["LDR"],
                beta_p_phase_thresh[class_type]["beta_p"]),
                beta_p_phase_thresh[class_type]["class_ind"], phase_mask)
            Class_legend[beta_p_phase_thresh[class_type]["class_ind"]-1] = \
                beta_p_phase_thresh[class_type]["class"]

        model.ds[mask_name] = xr.DataArray(phase_mask, dims=model.ds["LDR_%s" % cloud_class].dims)
        model.ds[mask_name].attrs["long_name"] = "%s phase classification mask" % instrument.instrument_str
        model.ds[mask_name].attrs["units"] = "Unitless"
        model.ds[mask_name].attrs["legend"] = Class_legend

    return model

