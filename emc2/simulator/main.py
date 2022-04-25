import numpy as np
from .subcolumn import set_convective_sub_col_frac, set_precip_sub_col_frac
from .subcolumn import set_stratiform_sub_col_frac, set_q_n
from .lidar_moments import calc_lidar_moments, calc_LDR_and_ext, calc_total_alpha_beta
from .radar_moments import calc_radar_moments, calc_total_reflectivity
from .attenuation import calc_radar_Ze_min
from .classification import lidar_classify_phase, lidar_emulate_cosp_phase, radar_classify_phase
from .psd import calc_re_thompson


def make_simulated_data(model, instrument, N_columns, do_classify=False, unstack_dims=False,
                        calc_re=False, skip_subcol_gen=False, finalize_fields=False, 
                        **kwargs):
    """
    This procedure will make all of the subcolumns and simulated data for each model column.

    NOTE:
    When starting a parallel task (in microphysics approach), it is recommended
    to wrap the top-level python script calling the EMC^2 processing ('lines_of_code')
    with the following command (just below the 'import' statements):
    
    .. code-block:: python
    
        if __name__ == “__main__”:
            lines_of_code

    Parameters
    ----------
    model: :func:`emc2.core.Model`
        The model to make the simulated parameters for.
    instrument: :func:`emc2.core.Instrument`
        The instrument to make the simulated parameters for.
    N_columns: int or None
        The number of subcolumns to generate. Set to None to automatically
        detect from LES 4D data.
    do_classify: bool
        run hydrometeor classification routines when True.
    unstack_dims: bool
        True - unstack the time, lat, and lon dimensions after processing in cases
        of regional model output.
    calc_re: bool
        True - calculating effective radius (e.g., for when it is not provided.
        Note that re is always calculated when WRF output is used.
    skip_subcol_gen: bool
        True - skip the subcolumn generator (e.g., in case subcolumn were already generated).
    finalize_fields: bool
        True - set absolute 0 values in"sub_col"-containing fields to np.nan enabling analysis
        and visualization.
    Additional keyword arguments are passed into :func:`emc2.simulator.calc_lidar_moments` or
    :func:`emc2.simulator.calc_radar_moments`

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with all of the simulated parameters generated.
    """
    print("## Creating subcolumns...")
    hydrometeor_classes = model.conv_frac_names.keys()

    if 'use_rad_logic' in kwargs.keys():
        use_rad_logic = kwargs['use_rad_logic']
        del kwargs['use_rad_logic']
    else:
        use_rad_logic = True

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

    if 'hyd_types' in kwargs.keys():
        hyd_types = kwargs['hyd_types']
        del kwargs['hyd_types']
    else:
        hyd_types = None

    if 'mie_for_ice' in kwargs.keys():
        mie_for_ice = {"conv": kwargs['mie_for_ice'],
                       "strat": kwargs['mie_for_ice']}
        del kwargs['mie_for_ice']
    else:
        if use_rad_logic:
            mie_for_ice = {"conv": False, "strat": False}
        else:
            mie_for_ice = {"conv": False, "strat": True}  # use True for strat (micro), False for conv (rad)
    if 'use_empiric_calc' in kwargs.keys():
        use_empiric_calc = kwargs['use_empiric_calc']
        del kwargs['use_empiric_calc']
    else:
        use_empiric_calc = False

    if skip_subcol_gen:
        print('Skipping subcolumn generator (make sure subcolumns were already generated).')
    else:
        if model.process_conv:
            for hyd_type in hydrometeor_classes:
                model = set_convective_sub_col_frac(
                    model, hyd_type, N_columns=N_columns,
                    use_rad_logic=use_rad_logic)
        else:
            print("No convective processing for %s" % model.model_name)

        # Subcolumn Generator
        model = set_stratiform_sub_col_frac(
            model, use_rad_logic=use_rad_logic, N_columns=N_columns, parallel=parallel, chunk=chunk)
        model = set_precip_sub_col_frac(
            model, is_conv=False, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)
        if model.process_conv:
            model = set_precip_sub_col_frac(
                model, is_conv=True, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)
        for hyd_type in hydrometeor_classes:
            if hyd_type != 'cl':
                model = set_q_n(
                    model, hyd_type, is_conv=False,
                    qc_flag=False, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)
                if model.process_conv:
                    model = set_q_n(
                        model, hyd_type, is_conv=True,
                        qc_flag=False, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)
            else:
                model = set_q_n(
                    model, hyd_type, is_conv=False,
                    qc_flag=True, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)
                if model.process_conv:
                    model = set_q_n(
                        model, hyd_type, is_conv=True,
                        qc_flag=False, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)

    # Calcualte r_eff if requested
    if np.logical_or(calc_re, model.model_name == "WRF"):
        model_vars = [x for x in model.ds.variables.keys()]
        for hyd_type in hydrometeor_classes:
            if not model.strat_re_fields[hyd_type] in model_vars:
                model = calc_re_thompson(model, hyd_type,
                                         is_conv=False, subcolumns=False,
                                         **kwargs)
    if model.process_conv:
        model_vars = [x for x in model.ds.variables.keys()]
        if not model.conv_re_fields[hyd_type] in model_vars:
                model = calc_re_thompson(model, hyd_type,
                                         is_conv=True, subcolumns=True,
                                         **kwargs)


    # Radar Simulator
    if instrument.instrument_class.lower() == "radar":
        print("Generating radar moments...")
        if 'reg_rng' in kwargs.keys():
            ref_rng = kwargs['ref_rng']
            del kwargs['ref_rng']
        else:
            ref_rng = 1000

        model = calc_radar_moments(
            instrument, model, False, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
            parallel=parallel, chunk=chunk, mie_for_ice=mie_for_ice["strat"],
            use_rad_logic=use_rad_logic,
            use_empiric_calc=use_empiric_calc, **kwargs)
        if model.process_conv:
            model = calc_radar_moments(
                instrument, model, True, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                parallel=parallel, chunk=chunk, mie_for_ice=mie_for_ice["conv"],
                use_rad_logic=use_rad_logic,
                use_empiric_calc=use_empiric_calc, **kwargs)

        model = calc_radar_Ze_min(instrument, model, ref_rng)
        model = calc_total_reflectivity(model, detect_mask=True)

        if do_classify is True:
            model = radar_classify_phase(
                instrument, model, mask_height_rng=mask_height_rng,
                convert_zeros_to_nan=convert_zeros_to_nan)

    # Lidar Simulator
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
        model = calc_lidar_moments(
            instrument, model, False, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
            parallel=parallel, eta=eta, chunk=chunk,
            mie_for_ice=mie_for_ice["strat"], use_rad_logic=use_rad_logic,
            use_empiric_calc=use_empiric_calc, **kwargs)
        if model.process_conv:
            model = calc_lidar_moments(
                instrument, model, True, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                parallel=parallel, eta=eta, chunk=chunk,
                mie_for_ice=mie_for_ice["conv"], use_rad_logic=use_rad_logic,
                use_empiric_calc=use_empiric_calc, **kwargs)
        model = calc_total_alpha_beta(model, OD_from_sfc=OD_from_sfc, eta=eta)
        model = calc_LDR_and_ext(model, ext_OD=ext_OD, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types)

        if do_classify is True:
            model = lidar_classify_phase(
                instrument, model, convert_zeros_to_nan=convert_zeros_to_nan)
            model = lidar_emulate_cosp_phase(
                instrument, model, eta=eta, OD_from_sfc=OD_from_sfc,
                convert_zeros_to_nan=convert_zeros_to_nan, hyd_types=hyd_types)
    else:
        raise ValueError("Currently, only lidars and radars are supported as instruments.")

    if finalize_fields:
        model.finalize_subcol_fields()

    # Unstack dims in case of regional model output (typically done at the end of all EMC^2 processing)
    if np.logical_and(model.stacked_time_dim is not None, unstack_dims):
        print("Unstacking the %s dimension (time, lat, and lon dimensions)" % model.stacked_time_dim)
        model.unstack_time_lat_lon()
    return model
