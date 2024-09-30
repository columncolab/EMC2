import emc2
import numpy as np


def test_get_SR():
    KAZR = emc2.core.instruments.KAZR('nsa')
    HSRL = emc2.core.instruments.HSRL()
    my_e3sm = emc2.core.model.E3SMv1(emc2.test_files.TEST_E3SM_FILE, all_appended_in_lat=True, appended_str=True)
    N_sub = 20
    my_e3sm = emc2.simulator.main.make_simulated_data(my_e3sm, HSRL, N_sub, do_classify=False, 
                                                      convert_zeros_to_nan=True, skip_subcol_gen=False)
    my_e3sm = emc2.simulator.main.make_simulated_data(my_e3sm, KAZR, N_sub,do_classify=False, convert_zeros_to_nan=True,
                                                      unstack_dims=True, finalize_fields=True,use_rad_logic=True)
    atb_total_4D, atb_mol_4D, z_full_km_3D, z_half_km_3D, Ze_att_total_4D =\
    emc2.statistics_LLNL.statistical_aggregation.get_radar_lidar_signals(
        my_e3sm)
    
    # Check for realistic ranges of attenuation, reflectivity, and height of beam attenuation
    assert np.all(Ze_att_total_4D[np.isfinite(Ze_att_total_4D)] > -120)
    assert np.all(Ze_att_total_4D[np.isfinite(Ze_att_total_4D)] < 30)
    assert np.all(z_full_km_3D[np.isfinite(z_full_km_3D)] < 12000)
    assert np.all(z_half_km_3D[np.isfinite(z_half_km_3D)] < 12000)
    assert np.all(atb_total_4D[np.isfinite(atb_total_4D)] > 0)
    assert np.all(atb_total_4D[np.isfinite(atb_total_4D)] < 1)

    subcolum_num = len(my_e3sm.ds.subcolumn)
    time_num = len(my_e3sm.ds.time)
    col_num = len(my_e3sm.ds.ncol)
    lev_num = len(my_e3sm.ds.lev)

    Ncolumns = subcolum_num  # subcolumn
    Npoints = time_num  # (time and col)
    Nlevels = lev_num

    Nglevels = 40
    zstep = 0.480
    levStat_km = np.arange(Nglevels)*zstep+zstep/2.
    newgrid_bot = (levStat_km)-0.24
    newgrid_top = (levStat_km)+0.24
    newgrid_mid = (newgrid_bot+newgrid_top)/2.


    SR_4D = emc2.statistics_LLNL.statistical_aggregation.calculate_SR(
        atb_total_4D, atb_mol_4D, subcolum_num, time_num, z_full_km_3D, z_half_km_3D,
        Ncolumns, Npoints, Nlevels, Nglevels, col_num, newgrid_bot, newgrid_top)
    assert np.all(SR_4D[np.isfinite(SR_4D)] > 0)
    SR_EDGES = np.array([
        -1., 0.01, 1.2, 3.0, 5.0, 7.0, 10.0,
        15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
        60.0, 80.0, 999.])
    SR_BINS_GR_ground = np.array([
        -4.950e-01, 6.050e-01, 2.100e+00, 4.000e+00, 6.000e+00, 
        8.500e+00,1.250e+01, 1.750e+01, 2.250e+01, 2.750e+01,
        3.500e+01,  4.500e+01, 5.500e+01, 7.000e+01, 5.395e+02])
    col_index = 2
    cfadSR_cal_alltime = emc2.statistics_LLNL.statistical_aggregation.get_cfad_SR(
        SR_EDGES, newgrid_mid, Npoints, Ncolumns, SR_4D, col_index)

    cfadSR_cal_alltime_col = np.empty(
        (Nglevels, len(SR_BINS_GR_ground), len(my_e3sm.ds.ncol.values)))

    for i in my_e3sm.ds.ncol.values:
        cfadSR_cal_alltime_col[:, :, i] = emc2.statistics_LLNL.statistical_aggregation.get_cfad_SR(
            SR_EDGES, newgrid_mid, Npoints, Ncolumns, SR_4D, i)

    cfadSR_cal_alltime = np.nanmean(cfadSR_cal_alltime_col, axis=2)

    # CFAD frequency cannot be greater than 1
    assert np.all(cfadSR_cal_alltime[np.isfinite(cfadSR_cal_alltime)] <= 1)

def test_get_CF():
    KAZR = emc2.core.instruments.KAZR('nsa')
    HSRL = emc2.core.instruments.HSRL()
    my_e3sm = emc2.core.model.E3SMv1(emc2.test_files.TEST_E3SM_FILE, all_appended_in_lat=True, appended_str=True)
    N_sub = 20
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, HSRL, N_sub, do_classify=False, 
        convert_zeros_to_nan=True, skip_subcol_gen=False)
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, KAZR, N_sub, do_classify=False, convert_zeros_to_nan=True,
        unstack_dims=True, finalize_fields=True, use_rad_logic=True)
    
    atb_total_4D, atb_mol_4D, z_full_km_3D, z_half_km_3D, Ze_att_total_4D =\
    emc2.statistics_LLNL.statistical_aggregation.get_radar_lidar_signals(
        my_e3sm)
    
    subcolum_num = len(my_e3sm.ds.subcolumn)
    time_num = len(my_e3sm.ds.time)
    col_num = len(my_e3sm.ds.ncol)
    lev_num = len(my_e3sm.ds.lev)

    Ncolumns = subcolum_num  # subcolumn
    Npoints = time_num  # (time and col)
    Nlevels = lev_num

    Nglevels = 40
    zstep = 0.480
    levStat_km = np.arange(Nglevels) * zstep + zstep / 2.
    newgrid_bot = (levStat_km) - 0.24
    newgrid_top = (levStat_km) + 0.24
    
    SR_4D = emc2.statistics_LLNL.statistical_aggregation.calculate_SR(
        atb_total_4D, atb_mol_4D, subcolum_num, time_num, z_full_km_3D, z_half_km_3D,
        Ncolumns, Npoints, Nlevels, Nglevels, col_num, newgrid_bot, newgrid_top)
    CF_3D = emc2.statistics_LLNL.statistical_aggregation.calculate_lidar_CF(
        SR_4D, time_num, Nglevels, col_num)
    
    # Cloud fraction must be between 0 and 1
    assert np.all(CF_3D[np.isfinite(CF_3D)] >= 0)
    assert np.all(CF_3D[np.isfinite(CF_3D)] <= 1)
    
