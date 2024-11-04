import emc2
import pytest
import matplotlib.pyplot as plt
import numpy as np

@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_timeseries():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", subplot_shape=(2, 2), figsize=(30, 20))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 1, subplot_index=(0, 0))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 2, subplot_index=(1, 0))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 3, subplot_index=(0, 1))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 4, subplot_index=(1, 1))
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_single_time():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_single_profile('sub_col_Ze_cl_strat', time='2016-08-16T09:30:00')
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_profile():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_subcolumn_mean_profile('sub_col_Ze_cl_strat', time='2016-08-16T09:30:00')
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_instrument():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)
    HSRL = emc2.core.instruments.HSRL()
    HSRL.read_arm_netcdf_file(emc2.test_files.TEST_INST_PLOT_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, subplot_shape=(1, 2), figsize=(10, 10))
    model_display.plot_instrument_timeseries(HSRL, "linear_depol", log_plot=False, cmap='magma',
                                             subplot_index=(0, 0), vmin=0.0, vmax=0.5)
    model_display.plot_instrument_timeseries(HSRL, "beta_a_backscat", log_plot=True, cmap='magma',
                                             subplot_index=(0, 1), vmin=1e-8, vmax=1e-3)
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_instrument_profile():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)
    HSRL = emc2.core.instruments.HSRL()
    HSRL.read_arm_netcdf_file(emc2.test_files.TEST_INST_PLOT_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_instrument_mean_profile(HSRL, 'linear_depol', pressure_coords=False)
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_classification():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    _, cb = model_display.plot_subcolumn_timeseries('phase_mask_KAZR_sounding_all_hyd', 1)
    model_display.change_plot_to_class_mask(cb, variable="phase_mask_KAZR_sounding_all_hyd",
                                            class_legend=["Cloud", "precip", "mixed"])
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_SR_subcol_timeseries():
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
    
    subcolum_num = len(my_e3sm.ds.subcolumn)
    time_num = len(my_e3sm.ds.time)
    col_num = len(my_e3sm.ds.ncol)
    lev_num = len(my_e3sm.ds.lev)
    col_index = 2

    Ncolumns = subcolum_num  # subcolumn
    Npoints = time_num  # (time and col)
    Nlevels = lev_num

    Nglevels = 40
    zstep = 0.480
    levStat_km = np.arange(Nglevels)*zstep+zstep/2.
    newgrid_bot = (levStat_km)-0.24
    newgrid_top = (levStat_km)+0.24

    SR_4D = emc2.statistics_LLNL.statistical_aggregation.calculate_SR(
        atb_total_4D, atb_mol_4D, subcolum_num, time_num, z_full_km_3D, z_half_km_3D,
        Ncolumns, Npoints, Nlevels, Nglevels, col_num, newgrid_bot, newgrid_top)
    emc2.statistics_LLNL.statistical_plots.plot_every_subcolumn_timeseries_SR(
        my_e3sm, atb_total_4D, atb_mol_4D, col_index, '', '', 'addpl_rad')
    assert plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_get_CFAD_SR():
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
    emc2.statistics_LLNL.statistical_plots.plot_lidar_SR_CFAD(
        SR_EDGES, newgrid_mid, cfadSR_cal_alltime, '', '', 'addpl_rad')
    assert plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_regridded_CF_timeseries():
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
    
    subcolum_num = len(my_e3sm.ds.subcolumn)
    time_num = len(my_e3sm.ds.time)
    col_num = len(my_e3sm.ds.ncol)
    lev_num = len(my_e3sm.ds.lev)

    Ncolumns = subcolum_num  # subcolumn
    Npoints = time_num  # (time and col)
    Nlevels = lev_num

    Nglevels = 40
    zstep = 0.480
    col_index = 2
    levStat_km = np.arange(Nglevels)*zstep+zstep/2.
    newgrid_bot = (levStat_km)-0.24
    newgrid_top = (levStat_km)+0.24
    newgrid_mid = (newgrid_bot+newgrid_top)/2.


    SR_4D = emc2.statistics_LLNL.statistical_aggregation.calculate_SR(
        atb_total_4D, atb_mol_4D, subcolum_num, time_num, z_full_km_3D, z_half_km_3D,
        Ncolumns, Npoints, Nlevels, Nglevels, col_num, newgrid_bot, newgrid_top)
    # Set input parameters.
    cmap = 'Spectral_r'
    field_to_plot = ["CF"]
    vmin_max = [(0., 1.)]
    log_plot = [False]
    is_radar_field = [False]
    y_range = (0., 15)  # in km
    subcol_ind = 0
    NSA_coords = {"lat": 71.32, "lon": -156.61}
    cbar_label = ['CF']
    model_display3 = emc2.plotting.SubcolumnDisplay(
        my_e3sm, figsize=(24*0.5, 6),
        lat_sel=NSA_coords["lat"],
        lon_sel=NSA_coords["lon"], tight_layout=True) 

    CF_3D = emc2.statistics_LLNL.statistical_aggregation.calculate_lidar_CF(
        SR_4D, time_num, Nglevels, col_num)
    for ii in range(len(field_to_plot)):
        model_display3.plot_regridded_CF_timeseries(
            CF_3D, newgrid_mid, col_index, y_range=y_range,
            cmap=cmap, title='',
            vmin=vmin_max[ii][0], vmax=vmin_max[ii][1], cbar_label=cbar_label[ii])
    assert model_display3.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_subcolumn_timeseries():
    KAZR = emc2.core.instruments.KAZR('nsa')
    HSRL = emc2.core.instruments.HSRL()
    my_e3sm = emc2.core.model.E3SMv1(emc2.test_files.TEST_E3SM_FILE, all_appended_in_lat=True, appended_str=True)
    N_sub = 20
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, HSRL, N_sub, do_classify=False, 
        convert_zeros_to_nan=True, skip_subcol_gen=False)
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, KAZR, N_sub,do_classify=False, convert_zeros_to_nan=True,
        unstack_dims=True, finalize_fields=True,use_rad_logic=True)
    cmap = "Spectral_r"
    field_to_plot = ["sub_col_beta_p_tot", "sub_col_Ze_att_tot"]
    vmin_max = [(1e-8, 1e-3),  (-50., 10.)]
    log_plot = [True,  False]
    is_radar_field = [False,  True]
    # y_range = (200., 1e3)  # in hPa
    y_range = (0, 15000)  # in m
    subcol_ind = 0
    NSA_coords = {"lat": 71.32, "lon": -156.61}

    # Generate a SubcolumnDisplay object for coords closest to the NSA site
    model_display = emc2.plotting.SubcolumnDisplay(
        my_e3sm, subplot_shape=(1, len(field_to_plot)), figsize=(15, 5),
        lat_sel=NSA_coords["lat"],
        lon_sel=NSA_coords["lon"], tight_layout=True)


    # Plot variables
    for ii in range(len(field_to_plot)):
        model_display.plot_subcolumn_timeseries(
            field_to_plot[ii], subcol_ind, log_plot=log_plot[ii], y_range=y_range,
            subplot_index=(0, ii),  cmap=cmap, title='',
            vmin=vmin_max[ii][0], vmax=vmin_max[ii][1], pressure_coords=False)
    assert model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plotting_every_subcolumn_timeseries_radarlidarsignal():
    KAZR = emc2.core.instruments.KAZR('nsa')
    HSRL = emc2.core.instruments.HSRL()
    my_e3sm = emc2.core.model.E3SMv1(emc2.test_files.TEST_E3SM_FILE, all_appended_in_lat=True, appended_str=True)
    N_sub = 20
    col_index = 2
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, HSRL, N_sub, do_classify=False, 
        convert_zeros_to_nan=True, skip_subcol_gen=False)
    my_e3sm = emc2.simulator.main.make_simulated_data(
        my_e3sm, KAZR, N_sub,do_classify=False, convert_zeros_to_nan=True,
        unstack_dims=True, finalize_fields=True, use_rad_logic=True)
    emc2.statistics_LLNL.statistical_plots.plot_every_subcolumn_timeseries_radarlidarsignal(
        my_e3sm, col_index, '', '', 'addpl_radiation')
    assert plt.gcf()
