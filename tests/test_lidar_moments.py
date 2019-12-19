import emc2
import numpy as np


def test_radar_moments_all_convective():
    instrument = emc2.core.instruments.HSRL()
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=True, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, True, 10)

    # Check to see if all OD's > 10 are masked. We should have thick enough cloud for this.
    assert np.nanmax(my_model.ds['sub_col_OD_tot_conv'].values) == 10
    assert np.all(np.diff(my_model.ds['sub_col_OD_tot_conv'].values, axis=1) > 0)
    assert my_model.ds['ext_max'].max() == 2

    # We should have all zeros
    my_model = emc2.simulator.lidar_moments.calc_lidar_moments(instrument, my_model, False, 10)
    assert np.nanmax(my_model.ds['sub_col_OD_tot_strat'].values) == 0