from .subcolumn import set_convective_sub_col_frac, set_precip_sub_col_frac
from .subcolumn import set_stratiform_sub_col_frac, set_q_n
from .lidar_moments import calc_lidar_moments, calc_LDR_and_ext, calc_total_alpha_beta
from .radar_moments import calc_radar_moments, calc_total_reflectivity
from .attenuation import calc_radar_Ze_min
from .classification import lidar_classify_phase, lidar_emulate_cosp_phase, radar_classify_phase


def make_simulated_data(model, instrument, N_columns, do_classify=False, **kwargs):
    """
    This procedure will make all of the subcolumns and simulated data for each model column.

    Parameters
    ----------
    model: :func:`emc2.core.Model`
        The model to make the simulated parameters for.
    instrument: :func:`emc2.core.Instrument`
        The instrument to make the simulated parameters for.
    N_columns: int
        The number of subcolumns to generate.
    do_classify: bool
        run hydrometeor classification routines when True.

    Additional keyword arguments are passed into :func:`emc2.simulator.calc_lidar_moments` or
    :func:`emc2.simulator.calc_radar_moments`

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with all of the simulated parameters generated.
    """
    print("## Creating subcolumns...")
    hydrometeor_classes = model.conv_frac_names.keys()
    for hyd_type in hydrometeor_classes:
        model = set_convective_sub_col_frac(model, hyd_type, N_columns=N_columns)

    model = set_stratiform_sub_col_frac(model)
    model = set_precip_sub_col_frac(model, convective=False)
    model = set_precip_sub_col_frac(model, convective=True)
    for hyd_type in hydrometeor_classes:
        if hyd_type != 'cl':
            model = set_q_n(model, hyd_type, is_conv=False, qc_flag=False)
            model = set_q_n(model, hyd_type, is_conv=True, qc_flag=False)
        else:
            model = set_q_n(model, hyd_type, is_conv=False, qc_flag=True)
            model = set_q_n(model, hyd_type, is_conv=True, qc_flag=False)

    if 'OD_from_sfc' in kwargs.keys():
        OD_from_sfc = kwargs['OD_from_sfc']
        del kwargs['OD_from_sfc']
    else:
        OD_from_sfc = instrument.OD_from_sfc

    if 'parallel' in kwargs.keys():
        parallel = kwargs['parallel']
        del kwargs['parallel']
    else:
        parallel = True
    if 'chunk' in kwargs.keys():
        chunk = kwargs['chunk']
        del kwargs['chunk']
    else:
        chunk = None
    if 'convert_zeros_to_nan' in kwargs.keys():
        convert_zeros_to_nan = kwargs['convert_zeros_to_nan']
        del kwargs['convert_zeros_to_nan']
    else:
        convert_zeros_to_nan = False
    if 'mask_height_rng' in kwargs.keys():
        mask_height_rng = kwargs['mask_height_rng']
        del kwargs['mask_height_rng']
    else:
        mask_height_rng = None

    if instrument.instrument_class.lower() == "radar":
        print("Generating radar moments...")
        if 'reg_rng' in kwargs.keys():
            ref_rng = kwargs['ref_rng']
            del kwargs['ref_rng']
        else:
            ref_rng = 1000
        model = calc_radar_moments(instrument, model, False, OD_from_sfc=OD_from_sfc, parallel=parallel,
                                   chunk=chunk, **kwargs)
        model = calc_radar_moments(instrument, model, True, OD_from_sfc=OD_from_sfc, parallel=parallel,
                                   chunk=chunk, **kwargs)
        model = calc_total_reflectivity(model)

        model = calc_radar_Ze_min(instrument, model, ref_rng)

        if do_classify is True:
            model = radar_classify_phase(instrument, model, mask_height_rng=mask_height_rng,
                                         convert_zeros_to_nan=convert_zeros_to_nan)

    elif instrument.instrument_class.lower() == "lidar":
        print("Generating lidar moments...")
        if 'ext_OD' in kwargs.keys():
            ext_OD = kwargs['ext_OD']
            del kwargs['ext_OD']
        else:
            ext_OD = instrument.ext_OD
        if 'eta' in kwargs.keys():
            eta = kwargs['eta']
            del kwargs['eta']
        else:
            eta = instrument.eta
        model = calc_lidar_moments(instrument, model, False, OD_from_sfc=OD_from_sfc,
                                   parallel=parallel, eta=eta, chunk=chunk, **kwargs)
        model = calc_lidar_moments(instrument, model, True, OD_from_sfc=OD_from_sfc,
                                   parallel=parallel, eta=eta, chunk=chunk, **kwargs)
        model = calc_total_alpha_beta(model, OD_from_sfc=OD_from_sfc, eta=eta)
        model = calc_LDR_and_ext(model, ext_OD=ext_OD, OD_from_sfc=OD_from_sfc)

        if do_classify is True:
            model = lidar_classify_phase(instrument, model, convert_zeros_to_nan=convert_zeros_to_nan)
            model = lidar_emulate_cosp_phase(instrument, model, eta=eta, OD_from_sfc=OD_from_sfc,
                                             convert_zeros_to_nan=convert_zeros_to_nan)
    else:
        raise ValueError("Currently, only lidars and radars are supported as instruments.")
    return model
