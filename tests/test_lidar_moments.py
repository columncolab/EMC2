import emc2
import numpy as np


def test_lidar_moments_all_convective():
    instrument = emc2.core.instruments.HSRL()
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False)
    my_model = emc2.simulator.lidar_moments.calc_total_alpha_beta(my_model)
    # Check to see if the signal goes extinct. We should have thick enough cloud for this. OD should
    # increase with height
    assert np.all(np.logical_or(np.diff(my_model.ds['sub_col_OD_tot_conv'].values, axis=1) > 0,
                                np.isnan(np.diff(my_model.ds['sub_col_OD_tot_conv'].values, axis=1))))

    # Maximum extinction mask value should be 2
    my_model = emc2.simulator.lidar_moments.calc_LDR_and_ext(my_model)
    assert my_model.ds['ext_mask'].max() == 2

    # We should have all zeros
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False)
    assert np.nanmax(my_model.ds['sub_col_OD_tot_strat'].values) == 0


def test_lidar_moments_all_stratiform():
    instrument = emc2.core.instruments.HSRL()
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True)
    my_model = emc2.simulator.lidar_moments.calc_total_alpha_beta(my_model)
    # OD should increase with height
    assert np.all(np.logical_or(np.diff(my_model.ds['sub_col_OD_tot_strat'].values, axis=1) >= 0,
                                np.isnan(np.diff(my_model.ds['sub_col_OD_tot_strat'].values, axis=1))))

    # Maximum extinction mask value should be 2
    my_model = emc2.simulator.lidar_moments.calc_LDR_and_ext(my_model)
    assert my_model.ds['ext_mask'].max() == 2

    # We should have all zeros in convection
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True)
    assert np.all(my_model.ds['sub_col_OD_tot_conv'].values == 0)


def test_lidar_classification():
    instrument = emc2.core.instruments.HSRL()
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True)
    my_model = emc2.simulator.lidar_moments.calc_total_alpha_beta(my_model)
    my_model = emc2.simulator.lidar_moments.calc_LDR_and_ext(my_model)
    my_model = emc2.simulator.classification.lidar_classify_phase(instrument, my_model)
    assert np.sum(my_model.ds.strat_phase_mask_HSRL.values) == 8


def test_cosp_emulator():
    instrument = emc2.core.instruments.CALIOP()
    my_model = emc2.core.model.TestAllStratiform()
    my_model_top = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model_top.ds = my_model.ds.copy()
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True)
    my_model_top = emc2.simulator.lidar_moments.calc_lidar_moments(
        instrument, my_model_top, False, OD_from_sfc=False)
    my_model_top = emc2.simulator.lidar_moments.calc_lidar_moments(
        instrument, my_model_top, True, OD_from_sfc=False)
    my_model = emc2.simulator.lidar_moments.calc_total_alpha_beta(my_model)
    my_model_top = emc2.simulator.lidar_moments.calc_total_alpha_beta(my_model_top)
    my_model = emc2.simulator.classification.lidar_emulate_cosp_phase(instrument, my_model)
    my_model_top = emc2.simulator.classification.lidar_emulate_cosp_phase(instrument, my_model_top)
    # Only liquid (value == 1) should be detected when observing from sfc (detected in all subcolumns).
    assert np.all(my_model.ds.COSP_phase_mask_all_hyd.values[my_model.ds.COSP_phase_mask_all_hyd.values
                  > 0] == 1)
    assert np.sum(my_model.ds.COSP_phase_mask_all_hyd.values[my_model.ds.COSP_phase_mask_all_hyd.values
                  > 0] == 1) == 8
    # Only ice (value == 2) should be detected when observing from TOA (detected in all subcolumns).
    assert np.all(my_model_top.ds.COSP_phase_mask_all_hyd.values[my_model_top.ds.COSP_phase_mask_all_hyd.values
                  > 0] == 2)
    assert np.sum(my_model_top.ds.COSP_phase_mask_all_hyd.values[my_model_top.ds.COSP_phase_mask_all_hyd.values
                  > 0] == 2) == 8
