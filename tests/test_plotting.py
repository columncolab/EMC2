import emc2
import pytest


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_timeseries():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", subplot_shape=(2, 2), figsize=(30, 20))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 1, subplot_index=(0, 0))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 2, subplot_index=(1, 0))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 3, subplot_index=(0, 1))
    model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 4, subplot_index=(1, 1))
    return model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_single_time():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_single_profile('sub_col_Ze_cl_strat', time='2016-08-16T09:30:00')
    return model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_profile():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_subcolumn_mean_profile('sub_col_Ze_cl_strat', time='2016-08-16T09:30:00')
    return model_display.fig


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
    return model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_instrument_profile():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)
    HSRL = emc2.core.instruments.HSRL()
    HSRL.read_arm_netcdf_file(emc2.test_files.TEST_INST_PLOT_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    model_display.plot_instrument_mean_profile(HSRL, 'linear_depol', pressure_coords=False)
    return model_display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_classification():
    model = emc2.core.model.ModelE(emc2.test_files.TEST_SUBCOL_FILE)

    model_display = emc2.plotting.SubcolumnDisplay(model, ds_name="ModelE", figsize=(10, 10))
    _, cb = model_display.plot_subcolumn_timeseries('phase_mask_KAZR_sounding_all_hyd', 1)
    model_display.change_plot_to_class_mask(cb, variable="phase_mask_KAZR_sounding_all_hyd",
                                            class_legend=["Cloud", "precip", "mixed"])
    return model_display.fig
