import emc2
import numpy as np


def test_radar_moments_all_convective():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, True)
    assert np.nanmax(my_model.ds["sub_col_Ze_tot_conv"].values) > 20
    assert np.nanmax(my_model.ds["sub_col_Ze_cl_conv"].values) > -10.
    assert np.nanmax(my_model.ds["sub_col_Ze_pl_conv"].values) > 20.

    # Ze_min should increase with height
    my_model = emc2.simulator.attenuation.calc_radar_Ze_min(instrument, my_model)
    assert np.all(np.logical_or(np.diff(my_model.ds["Ze_min"].values) > 0,
                                np.isnan(np.diff(my_model.ds['Ze_min'].values))))


def test_radar_classification():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, True)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, False)
    assert np.nanmax(my_model.ds["sub_col_Ze_tot_conv"].values) > 20
    assert np.nanmax(my_model.ds["sub_col_Ze_cl_conv"].values) > -10.
    assert np.nanmax(my_model.ds["sub_col_Ze_pl_conv"].values) > 20.

    # Ze_min should increase with height
    my_model = emc2.simulator.attenuation.calc_radar_Ze_min(instrument, my_model)
    assert np.all(np.logical_or(np.diff(my_model.ds["Ze_min"].values) > 0,
                                np.isnan(np.diff(my_model.ds['Ze_min'].values))))
    my_model = emc2.simulator.radar_moments.calc_total_reflectivity(my_model)
    my_model = emc2.simulator.classification.radar_classify_phase(instrument, my_model)
    assert np.sum(my_model.ds.phase_mask_KAZR_sounding_all_hyd.values) > 10


def test_radar_moments_all_stratiform():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, False)
    assert np.nanmax(my_model.ds["sub_col_Ze_tot_strat"].values) < 80.
    assert np.nanmax(my_model.ds["sub_col_Ze_pi_strat"].values) < 80.
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, False, use_rad_logic=False)
    assert np.all(np.nanmax(np.abs(my_model.ds["sub_col_Vd_cl_strat"].values)) < 1.5)
    assert np.all(np.nanmax(my_model.ds["sub_col_Vd_pl_strat"].values) <= 1)

    # Ze_min should increase with height
    my_model = emc2.simulator.attenuation.calc_radar_Ze_min(instrument, my_model)
    assert np.all(np.logical_or(np.diff(my_model.ds["Ze_min"].values) > 0,
                                np.isnan(np.diff(my_model.ds['Ze_min'].values))))
