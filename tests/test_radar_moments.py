import emc2
import numpy as np


def test_convective_reflectivity():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "cl")
    assert np.nanmax(my_model.ds["Ze"].values) > 30.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "pl")
    assert np.nanmax(my_model.ds["Ze"].values) > 25.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "ci")
    assert np.nanmax(my_model.ds["Ze"].values) < 10
    assert np.nanmax(my_model.ds["Ze"].values) > -10.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "pi")
    assert np.nanmax(my_model.ds["Ze"].values) < 10
    assert np.nanmax(my_model.ds["Ze"].values) > -10


def test_radar_moments_all_convective():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=True)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, True)
    assert np.nanmax(my_model.ds["sub_col_Ze_tot_conv"].values) > 40
    assert np.nanmax(my_model.ds["sub_col_Ze_cl_conv"].values) > 40.
    assert np.nanmax(my_model.ds["sub_col_Ze_pl_conv"].values) > 40.


def test_radar_moments_all_stratiform():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, False)
    assert np.nanmax(my_model.ds["sub_col_Ze_tot_strat"].values) > 40
    assert my_model.ds["sub_col_Ze_cl_strat"].values.max() > 30.
    assert my_model.ds["sub_col_Ze_pl_strat"].values.max() > 10.
    assert np.all(~np.isfinite(my_model.ds["sub_col_Ze_pi_strat"].values))
    assert np.all(~np.isfinite(my_model.ds["sub_col_Ze_ci_strat"].values))
    assert np.all(np.abs(my_model.ds["sub_col_Vd_cl_strat"].values) < 1)
    assert np.all(my_model.ds["sub_col_Vd_pl_strat"].values <= 0)
