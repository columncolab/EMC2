from .subcolumn import set_convective_sub_col_frac, set_precip_sub_col_frac
from .subcolumn import set_stratiform_sub_col_frac, set_q_n
from .lidar_moments import calc_lidar_moments, calc_LDR
from .radar_moments import calc_radar_moments


def make_simulated_data(model, instrument, N_columns, **kwargs):
    """
    This procedure will make all of the subcolumns and simulated data for each subcolumn.

    Parameters
    ----------
    model: :func:`emc2.core.Model`
        The model to make the simulated parameters for.
    instrument: :func:`emc2.core.Instrument`
        The instrument to make the simulated parameters for.
    N_columns: int
        The number of subcolumns to generate.

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
        if hyd_type is not 'cl':
            model = set_q_n(model, hyd_type, is_conv=False, qc_flag=False)
            model = set_q_n(model, hyd_type, is_conv=True, qc_flag=False)
        else:
            model = set_q_n(model, hyd_type, is_conv=False, qc_flag=True)
            model = set_q_n(model, hyd_type, is_conv=True, qc_flag=False)

    if 'OD_from_sfc' in kwargs.keys():
        OD_from_sfc = kwargs['OD_from_sfc']
        del kwargs['OD_from_sfc']
    else:
        OD_from_sfc = True

    if 'parallel' in kwargs.keys():
        parallel = kwargs['parallel']
        del kwargs['parallel']
    else:
        parallel = True

    if instrument.instrument_class.lower() == "radar":
        print("Generating radar moments...")
        model = calc_radar_moments(instrument, model, False, OD_from_sfc=OD_from_sfc, parallel=parallel, **kwargs)
        model = calc_radar_moments(instrument, model, True, OD_from_sfc=OD_from_sfc, parallel=parallel, **kwargs)
    elif instrument.instrument_class.lower() == "lidar":
        print("Generating lidar moments...")
        model = calc_lidar_moments(instrument, model, False, OD_from_sfc=OD_from_sfc, parallel=parallel, **kwargs)
        model = calc_lidar_moments(instrument, model, True, OD_from_sfc=OD_from_sfc, parallel=parallel, **kwargs)
        if 'ext_OD' in kwargs.keys():
            ext_OD = kwargs['ext_OD']
        else:
            ext_OD = 10.

        model = calc_LDR(model, ext_OD=ext_OD, OD_from_sfc=OD_from_sfc)
    else:
        raise ValueError("Currently, only lidars and radars are supported as instruments.")
    return model
