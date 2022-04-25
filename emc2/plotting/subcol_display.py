import numpy as np
from scipy.stats import gmean, gstd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.dates as mdates
import warnings
import copy

from act.plotting import Display


class SubcolumnDisplay(Display):
    """
    This class contains modules for displaying the generated subcolumn parameters as quicklook
    plots. It is inherited from `ACT <https://arm-doe.github.io/ACT>`_'s Display object. For more
    information on the Display object and its attributes and parameters, click `here
    <https://arm-doe.github.io/ACT/API/generated/act.plotting.plot.Display.html>`_. In addition to the
    methods in :code:`Display`, :code:`SubcolumnDisplay` has the following attributes and methods:
    The plotting dataset is automatically cropped to include only single lat and lon coordinates in case
    of a regional output, but can always be replaced with different coordinates using the internal methods.
    Note that there is no dedicated option to plot subcolumns vs. lat or lon, since subcolumns at a given
    grid cell are independent of subcolumns attributed to other neighboring grid cells.
    Note: if older version of ACT are installed (e.g., 0.2.4), "_obj" should be replaced with "_arm".

    Attributes
    ----------
    model: emc2.core.Model
        The model object containing the subcolumn data to plot.

    Examples
    --------
    This example makes a four panel plot of 4 subcolumns of EMC^2 simulated reflectivity::

    $ model_display = emc2.plotting.SubcolumnDisplay(my_model, subplot_shape=(2, 2), figsize=(30, 20))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 1, subplot_index=(0, 0))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 2, subplot_index=(1, 0))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 3, subplot_index=(0, 1))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 4, subplot_index=(1, 1))

    """
    def __init__(self, model, lat_sel=None, lon_sel=None, **kwargs):
        """

        Parameters
        ----------
        model: emc2.core.Model
            The model containing the subcolumn data to plot.
        lat_sel: float, int, or None
            Relevant only if a latitude dimension exists in dataset (model output file).
            if float, then specifying the latitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.
        lon_sel: float, int, or None
            Relevant only if a lonitude dimension exists in dataset (model output file).
            if float, then specifying the lonitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.

        Additional keyword arguments are passed into act.plotting.plot.Display's constructor.
        """
        if 'ds_name' not in kwargs.keys():
            ds_name = model.model_name
        else:
            ds_name = kwargs.pop('ds_name')
        if np.any([x in model.ds.dims for x in [model.lat_dim, model.lon_dim]]):
            model_c = copy.deepcopy(model)
        else:
            model_c = model
        self.model = model_c
        self._crop_lat_lon(model_c, lat_sel, lon_sel, deep=True)
        super().__init__(model_c.ds, ds_name=ds_name, **kwargs)

    def _crop_lat_lon(self, model, lat_sel=None, lon_sel=None, deep=False):
        """
        cropping lat and/or lon coordinates enabling robust use of plotting routines.

        Parameters
        ----------
        model: emc2.core.Model
            The model containing the subcolumn data to plot.
        lat_sel: float, int, or None
            Relevant only if a latitude dimension exists in dataset (model output file).
            if float, then specifying the latitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.
        lon_sel: float, int, or None
            Relevant only if a lonitude dimension exists in dataset (model output file).
            if float, then specifying the lonitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.
        deep: bool
            If True, create a deep copy of the dataset in case of cropping
        """
        if lat_sel is None:
            lat_sel = 0
        if lon_sel is None:
            lon_sel = 0
        self._switch_lat_lon(model, lat_sel, lon_sel, deep)

    def _switch_lat_lon(self, model, lat_sel=None, lon_sel=None, deep=False):
        """
        switching cropped lat and/or lon coordinates assuming a fully processed dataset exists.

        Parameters
        ----------
        model: emc2.core.Model
            The model containing the subcolumn data to plot.
        lat_sel: float, int, or None
            Relevant only if a latitude dimension exists in dataset (model output file).
            if float, then specifying the latitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.
        lon_sel: float, int, or None
            Relevant only if a lonitude dimension exists in dataset (model output file).
            if float, then specifying the lonitude value for which to crop the model xr.Dataset for
            plotting (using the nearest value).
            if int, then specifying the index to crop.
            if None, then using index 0 to prevent issues.
        deep: bool
            If True, create a deep copy of the dataset in case of cropping.
        """
        if not np.logical_and(lat_sel is None, lon_sel is None):
            self.cropped_lat_lon = [lat_sel, lon_sel]
            if np.any([x in self.model.ds.dims for x in [self.model.lat_dim, self.model.lon_dim]]):
                if deep:
                    self.model.ds = copy.deepcopy(model.ds)
                if self.model.lat_dim in self.model.ds.dims:
                    if isinstance(lat_sel, float):
                        print("cropping lat dim (lat requested = %.2f)" % lat_sel)
                        self.model.ds = self.model.ds.sel({self.model.lat_dim: lat_sel}, method='nearest')
                    elif isinstance(lat_sel, int):
                        print("cropping lat dim (lat index requested = %d)" % lat_sel)
                        self.model.ds = self.model.ds.isel({self.model.lat_dim: lat_sel})
                if self.model.lon_dim in self.model.ds.dims:
                    if isinstance(lon_sel, float):
                        print("cropping lon dim (lon requested = %.2f)" % lon_sel)
                        self.model.ds = self.model.ds.sel({self.model.lon_dim: lon_sel}, method='nearest')
                    elif isinstance(lon_sel, int):
                        print("cropping lon dim (lon index requested = %d)" % lon_sel)
                        self.model.ds = self.model.ds.isel({self.model.lon_dim: lon_sel})
        else:
            print("no alternative lat and lon coords provided - keeping processed dataset as is.")

    def _switch_model(self, model):
        """
        Replace the processed model data in the SubcolumnDisplay object with a deep copy of
        processed model data allowing full compitability with Display object (e.g., plotting
        output from multiple processing methods using the SubcolumnDisplay plotting routines).

        Parameters
        ----------
        model: emc2.core.Model
            A model containing the subcolumn data to relpace the current model data.

        """
        self._obj.pop(self.model.model_name)
        self.model = model
        self._obj.update({self.model.model_name: self.model.ds})

    def calc_mean_and_sd(self, variable, use_geom_mean=False, axis=None):
        """
        This function calculates geometric or arithmetic mean and SD of arrays.

        Parameters
        ----------
        variable: np.ndarray
            array to use for calculation.
        axis: int, tuple, or None
            axis along which to calculate the mean and SD. None for calcualtion over the
            full array (single output value).
        use_geom_mean: str or bool
            if True, then using geometric mean and SD, If False, using arithmetic mean and SD.

        Returns
        -------
        Mean: np.ndarray
            array of calculated mean.
        SD: np.ndarray
            array of calculated SD.
        """
        if use_geom_mean:
            if np.any(variable <= 0):
                print("Negative values exist in array - will be ignored in geomteric SD calculation")
            finite_arr = np.logical_and(np.isfinite(variable), variable > 0)
            n_finite_elem = np.sum(finite_arr, axis=axis)
            variable_tmp = np.where(finite_arr, variable, np.nan)
            Mean = np.exp((1 / n_finite_elem) * np.nansum(np.log(variable_tmp), axis=axis))
            SD = np.exp(np.nanstd(np.log(variable_tmp), axis=axis))
        else:
            Mean = np.nanmean(variable, axis=axis)
            SD = np.nanstd(variable, axis=axis)
        return Mean, SD

    @staticmethod
    def set_axis_label(ds, variable, max_long_name=20, use_prespec=True):
        """
        Setting the axis label based on whether a field name is one of prespecified fields,
        its attributes exist, and whether these attributes are not "too long".

        Parameters
        ----------
        ds: xr.Dataset
            dataset containing the field for which to generate the label.
        variable: str
            Name of variable for which to generate axis title.
        max_long_name: int
            maximum length of the "long_name" attribute to include in plot
        use_prespec: bool
            If True, then setting axis label based on pre-specified variable names (if exist in variable name).

        Returns
        -------
        axis_title: str
            axis title
        """
        axis_label = None

        if use_prespec:
            # Choose label based on variable
            variables = ['Ze', 'Vd', 'sigma_d', 'od', 'OD', 'beta', 'alpha', 'LDR']
            hyd_met = ''
            hyd_types = ['cl', 'ci', 'pl', 'pi', 'tot']
            for hyd in hyd_types:
                if hyd in variable:
                    hyd_met = hyd

            for var in variables:
                if var in variable:
                    if var == 'Ze':
                        axis_label = r'$Z_{e, %s}$ [dBZ]' % hyd_met
                    elif var == 'Vd':
                        axis_label = r'$V_{d, %s}$ [m/s]' % hyd_met
                    elif var == 'sigma_d':
                        axis_label = r'$\sigma_{d, %s}$ [m/s]' % hyd_met
                    elif var in ['od', 'OD']:
                        axis_label = r'$\tau_{%s}$' % hyd_met
                    elif var == 'beta':
                        axis_label = r'$\beta_{%s}$ [$m^{-1} sr^{-1}$]' % hyd_met
                    elif var == 'alpha':
                        axis_label = r'$\alpha_{%s}$ [$m^{-1}$]' % hyd_met
                    elif var == 'LDR':
                        axis_label = 'LDR'

        if axis_label is None:
            if "units" in ds[variable].attrs:
                axis_label = "[%s]" % ds[variable].attrs["units"]
                if "long_name" in ds[variable].attrs:
                    if len(ds[variable].attrs["long_name"]) <= max_long_name:
                        axis_label = '%s [%s]' % (ds[variable].attrs["long_name"],
                                                  ds[variable].attrs["units"])
            else:
                axis_label = variable
        return axis_label

    def set_yrng(self, subplot_index, y_range):
        """
        Set the Y axes limits of the subplot

        Parameters
        ----------
        subplot_index: tuple
            The index of the subplot to set the y axes limits to.
        y_range: tuple
            The y range of the plot.

        Returns
        -------

        """
        self.axes[subplot_index].set_ylim(y_range)

    def set_xrng(self, subplot_index, x_range):
        """
        Set the Y axes limits of the subplot

        Parameters
        ----------
        subplot_index: tuple
            The index of the subplot to set the y axes limits to.
        x_range: tuple
            The y range of the plot.

        Returns
        -------

        """
        self.axes[subplot_index].set_xlim(x_range)

    def change_plot_to_class_mask(self, cbar, class_legend=None, variable=None, cbar_label="",
                                  cmap=None, convert_zeros_to_nan=False, **kwargs):
        """
        Updates the colorbar to show phase classification.

        Parameters
        ----------
        cbar: Matplotlib axes handle
            colorbar handle.
        class_legend: list
            Class type strings in order corresponding to mask integer values.
            If None, using the "legend" attributes from the mask variable.
        variable: str
            The classification mask variable to use assuming it has a "legend" attribute
            Raises an error when both variable and class_legend are both None.
        cbar_label: str
            The colorbar label. Empty string by default.
        cmap: ListedColormap object
            colormap to use in the colorbar. If None, using tab20c(N), where N is the number of
            classes.
        convert_zeros_to_nan: bool
            If True, assuming that the plotted classification mask has all the zeros converted
            to nans, i.e., 'convert_zeros_to_nan' was True when the classification method was called.

        Returns
        -------
        cbar: Matplotlib axes handle
            The matplotlib colorbar handle of the plot.
        """
        if np.logical_and(class_legend is None, variable is None):
            raise ValueError("both the class_legend and the mask variable are None")

        if class_legend is None:
            class_legend = self.model.ds[variable].attrs["legend"]
        l_legend = len(class_legend)
        if cmap is None:
            if convert_zeros_to_nan:
                cmap = cm.get_cmap("tab20c", lut=l_legend)
            else:
                cmap = cm.get_cmap("tab20c", lut=l_legend + 1)

        if convert_zeros_to_nan:
            cm.ScalarMappable.set_clim(cbar.mappable, vmin=0.5, vmax=l_legend + 0.5)
        else:
            cm.ScalarMappable.set_clim(cbar.mappable, vmin=-0.5, vmax=l_legend + 0.5)
        cm.ScalarMappable.set_cmap(cbar.mappable, cmap=cmap)
        cbar.set_ticks([x for x in np.arange(1, l_legend + 1)])
        if convert_zeros_to_nan:
            cbar.set_ticks([x for x in np.arange(1, l_legend + 1)])
            cbar.set_ticklabels(class_legend)
        else:
            cbar.set_ticks([x for x in np.arange(0, l_legend + 1)])
            cbar.set_ticklabels(["clear"] + class_legend)
        cbar.set_label(cbar_label)

        return cbar

    def plot_subcolumn_timeseries(self, variable, column_no=0, pressure_coords=True, title=None,
                                  subplot_index=(0, ), colorbar=True, cbar_label=None,
                                  log_plot=False, Mask_array=None, hatched_mask=False,
                                  x_range=None, y_range=None, x_dateformat="%b%d-%H",
                                  x_rotation=30., **kwargs):
        """
        Plots timeseries of subcolumn parameters for a given variable and subcolumn.
        In the case of a 2D (time x height) field, plotting a time-height curtain.

        Parameters
        ----------
        variable: str
            The subcolumn variable to plot.
        column_no: int
            The subcolumn number to plot. By default, using the first subcolumn.
        pressure_coords: bool
            Set to true to plot in pressure coordinates, false to height coordinates.
        title: str or None
            The title of the plot. Set to None to have EMC^2 generate a title for you.
        subplot_index: tuple
            The index of the subplot to make the plot in.
        colorbar: bool
            If true, plot the colorbar.
        cbar_label: None or str
            The colorbar label. Set to None to provide a default label.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        Mask_array: bool, int, or float (same dims as "variable")
            Set to true or to other values greater than 0 in grid cells to make them transparent.
        hatched_mask: bool or str
            True - masked areas show masked '/' pattern, False - Masked area is transparent,
            str - use the str as the hatch pattern (see:
            https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html).
        x_range: tuple, list, or None
            The x range of the plot (also accepts datetime64 format).
        y_range: tuple, list, or None
            The y range of the plot.
        x_dateformat: str
            Date format for the x-axis.
        x_rotation: float
            x-axis label rotation for a date axis.
        Additional keyword arguments are passed into matplotlib's matplotlib.pyplot.pcolormesh.

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        cbar: Matplotlib axes handle
            The matplotlib colorbar handle of the plot.
        """
        ds_name = [x for x in self._obj.keys()][0]
        if len(self.model.ds[variable].dims) == 3:
            my_ds = self._obj[ds_name].sel(subcolumn=column_no)
        else:
            my_ds = self._obj[ds_name]
        x_variable = self.model.time_dim
        if pressure_coords:
            y_variable = self.model.height_dim
        else:
            y_variable = self.model.z_field

        x_label = 'Time [UTC]'
        y_label = self.set_axis_label(my_ds, y_variable)

        if cbar_label is None:
            cbar_label = self.set_axis_label(my_ds, variable)

        if pressure_coords:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values
            x, y = np.meshgrid(x, y)
        else:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values.T
            p = my_ds[self.model.height_dim].values
            x, p = np.meshgrid(x, p)

        var_array = my_ds[variable].values.T
        if Mask_array is not None:
            Mask_array = Mask_array.T
            if Mask_array.shape == var_array.shape:
                if not hatched_mask:
                    var_array = np.where(Mask_array <= 0, var_array, np.nan)
            else:
                print("Mask dimensions " + str(Mask_array.shape) +
                      " are different than in the requested field " +
                      str(var_array.shape) + " - ignoring mask")
        if y_range is not None:
            self.axes[subplot_index].set_ylim(y_range)
        if x_range is not None:
            self.axes[subplot_index].set_xlim(x_range)

        if np.issubdtype(x.dtype, np.datetime64):
            date_xaxis = True
            x = mdates.date2num([y for y in x])
        else:
            date_xaxis = False

        if log_plot is True:
            if 'vmin' in kwargs.keys():
                vmin = kwargs['vmin']
                del kwargs['vmin']
            if 'vmax' in kwargs.keys():
                vmax = kwargs['vmax']
                del kwargs['vmax']
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                                       **kwargs)
        else:
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, **kwargs)
        if isinstance(hatched_mask, str):
            hatch = hatched_mask
            hatched_mask = True
        else:
            hatch = '\\/...'
        if hatched_mask:
            self.axes[subplot_index].pcolor(x, y, np.ma.masked_where(Mask_array == 0, np.ones_like(var_array)),
                                            hatch=hatch, alpha=0.)

        if date_xaxis:
            self.axes[subplot_index].xaxis.set_major_formatter(mdates.DateFormatter(x_dateformat))
            for label in self.axes[subplot_index].get_xticklabels(which='major'):
                label.set(rotation=x_rotation, horizontalalignment='right')

        if title is None:
            self.axes[subplot_index].set_title(self.model.model_name + ' ' +
                                               np.datetime_as_string(self.model.ds[x_variable][0].values))
        else:
            self.axes[subplot_index].set_title(title)

        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        if colorbar:
            cbar = plt.colorbar(mesh, ax=self.axes[subplot_index])
            cbar.set_label(cbar_label)
            return self.axes[subplot_index], cbar

        return self.axes[subplot_index]

    def plot_instrument_timeseries(self, instrument, variable, title=None,
                                   subplot_index=(0, ), colorbar=True, cbar_label=None,
                                   log_plot=False, Mask_array=None, hatched_mask=False,
                                   x_range=None, y_range=None, x_dateformat="%b%d-%H",
                                   x_rotation=30., **kwargs):
        """
        Plots timeseries of a given instrument variable.

        Parameters
        ---------
        instrument: :py:mod:`emc2.core.Instrument`
            The Instrument class that you wish to plot.
        variable: str
            The variable to plot.
        title: str or None
            The title of the plot. Set to None to have EMC^2 generate a title for you.
        subplot_index: tuple
            The index of the subplot to make the plot in.
        colorbar: bool
            If true, plot the colorbar.
        cbar_label: None or str
            The colorbar label. Set to None to provide a default label.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        Mask_array: bool, int, or float (same dims as "variable")
            Set to true or to other values greater than 0 in grid cells to make them transparent.
        hatched_mask: bool or str
            True - masked areas show masked '/' pattern, False - Masked area is transparent,
            str - use the str as the hatch pattern (see:
            https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html).
        x_range: tuple, list, or None
            The x range of the plot (also accepts datetime64 format).
        y_range: tuple, list, or None
            The y range of the plot.
        x_dateformat: str
            Date format for the x-axis.
        x_rotation: float
            x-axis label rotation for a date axis.
        Additional keyword arguments are passed into matplotlib's matplotlib.pyplot.pcolormesh.

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        cbar: Matplotlib axes handle
            The matplotlib colorbar handle of the plot.
        """
        my_ds = instrument.ds
        x_variable = "time"
        if 'range' in my_ds.keys():
            y_variable = "range"
        elif 'altitude' in my_ds.keys():
            y_variable = "altitude"
        elif 'height' in my_ds.keys():
            y_variable = "height"

        x_label = 'Time [UTC]'
        y_label = self.set_axis_label(my_ds, y_variable)

        if cbar_label is None:
            cbar_label = self.set_axis_label(my_ds, variable)

        x = my_ds[x_variable].values
        y = my_ds[y_variable].values
        x, y = np.meshgrid(x, y)
        var_array = my_ds[variable].values.T
        if Mask_array is not None:
            Mask_array = Mask_array.T
            if Mask_array.shape == var_array.shape:
                if not hatched_mask:
                    var_array = np.where(Mask_array <= 0, var_array, np.nan)
            else:
                print("Mask dimensions " + str(Mask_array.shape) +
                      " are different than in the requested field " +
                      str(var_array.shape) + " - ignoring mask")
        if y_range is not None:
            self.axes[subplot_index].set_ylim(y_range)
        if x_range is not None:
            self.axes[subplot_index].set_xlim(x_range)

        if np.issubdtype(x.dtype, np.datetime64):
            date_xaxis = True
            x = mdates.date2num([y for y in x])
        else:
            date_xaxis = False

        if log_plot is True:
            if 'vmin' in kwargs.keys():
                vmin = kwargs['vmin']
                del kwargs['vmin']
            if 'vmax' in kwargs.keys():
                vmax = kwargs['vmax']
                del kwargs['vmax']
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                                       **kwargs)
        else:
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, **kwargs)
        if isinstance(hatched_mask, str):
            hatch = hatched_mask
            hatched_mask = True
        else:
            hatch = '\\/...'
        if hatched_mask:
            self.axes[subplot_index].pcolor(x, y, np.ma.masked_where(Mask_array == 0, np.ones_like(var_array)),
                                            hatch=hatch, alpha=0.)

        if date_xaxis:
            self.axes[subplot_index].xaxis.set_major_formatter(mdates.DateFormatter(x_dateformat))
            for label in self.axes[subplot_index].get_xticklabels(which='major'):
                label.set(rotation=x_rotation, horizontalalignment='right')

        if title is None:
            self.axes[subplot_index].set_title(instrument.instrument_str + ' ' +
                                               np.datetime_as_string(my_ds.time[0].values))
        else:
            self.axes[subplot_index].set_title(title)

        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        if colorbar:
            cbar = plt.colorbar(mesh, ax=self.axes[subplot_index])
            cbar.set_label(cbar_label)
            return self.axes[subplot_index], cbar

        return self.axes[subplot_index]

    def plot_single_profile(self, variable, time, pressure_coords=True, title=None,
                            subplot_index=(0,), colorbar=True, cbar_label=None,
                            log_plot=False, Mask_array=None, hatched_mask=False,
                            x_range=None, y_range=None, **kwargs):
        """
        Plots the single profile of subcolumns for a given time period.

        Parameters
        ----------
        variable: str
            The subcolumn variable to plot.
        time: tuple of Datetime or str
            The time step to plot. If a string, specify in the format '%Y-%m-%dT%H:%M:%S'
        pressure_coords: bool
            Set to true to plot in pressure coordinates, false to height coordinates.
        title: str or None
            The title of the plot. Set to None to have EMC^2 generate a title for you.
        subplot_index: tuple
            The index of the subplot to make the plot in.
        colorbar: bool
            If true, then plot the colorbar.
        cbar_label: None or str
            The colorbar label. Set to None to provide a default label.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        Mask_array: bool, int, or float (same dims as "variable")
            Set to true or to other values greater than 0 in grid cells to make them transparent
            or hatched.
        hatched_mask: bool or str
            True - masked areas show masked '/' pattern, False - Masked area is transparent,
            str - use the str as the hatch pattern (see:
            https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html).
        x_range: tuple, list, or None
            The x range of the plot (also accepts datetime64 format).
        y_range: tuple, list, or None
            The y range of the plot.
        Additional keyword arguments are passed into matplotlib's matplotlib.pyplot.pcolormesh.

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        cbar: Matplotlib axes handle
            The matplotlib colorbar handle of the plot.
        """
        ds_name = [x for x in self._obj.keys()][0]
        my_ds = self._obj[ds_name].sel({self.model.time_dim: time}, method='nearest')

        if pressure_coords:
            y_variable = self.model.height_dim
        else:
            y_variable = self.model.z_field

        x_label = 'Subcolumn #'
        y_label = self.set_axis_label(my_ds, y_variable)

        if cbar_label is None:
            cbar_label = self.set_axis_label(my_ds, variable)

        if pressure_coords:
            x = np.arange(0, self.model.num_subcolumns, 1)
            y = my_ds[y_variable].values
            x, y = np.meshgrid(x, y)
        else:
            x = np.arange(0, self.model.num_subcolumns, 1)
            y = my_ds[y_variable].values.T
            p = my_ds[self.model.height_dim].values
            x, p = np.meshgrid(x, p)

        var_array = my_ds[variable].values.T
        if Mask_array is not None:
            Mask_array = Mask_array.T
            if Mask_array.shape == var_array.shape:
                if not hatched_mask:
                    var_array = np.where(Mask_array <= 0, var_array, np.nan)
            else:
                print("Mask dimensions " + str(Mask_array.shape) +
                      " are different than in the requested field " +
                      str(var_array.shape) + " - ignoring mask")
        if y_range is not None:
            self.axes[subplot_index].set_ylim(y_range)
        if x_range is not None:
            self.axes[subplot_index].set_xlim(x_range)
        if log_plot is True:
            if 'vmin' in kwargs.keys():
                vmin = kwargs['vmin']
                del kwargs['vmin']
            if 'vmax' in kwargs.keys():
                vmax = kwargs['vmax']
                del kwargs['vmax']
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                                       **kwargs)
        else:
            mesh = self.axes[subplot_index].pcolormesh(x, y, var_array, **kwargs)
        if isinstance(hatched_mask, str):
            hatch = hatched_mask
            hatched_mask = True
        else:
            hatch = '\\/...'
        if hatched_mask:
            self.axes[subplot_index].pcolor(x, y, np.ma.masked_where(Mask_array == 0, np.ones_like(var_array)),
                                            hatch=hatch, alpha=0.)
        if title is None:
            time_title = ""
            if isinstance(time, str):
                time_title = time
            elif isinstance(time, np.datetime64):
                time_title = np.datetime_as_string(time)

            self.axes[subplot_index].set_title(self.model.model_name + ' ' +
                                               time_title)
        else:
            self.axes[subplot_index].set_title(title)

        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()

        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        if colorbar:
            cbar = plt.colorbar(mesh, ax=self.axes[subplot_index])
            cbar.set_label(cbar_label)
            return self.axes[subplot_index], cbar

        return self.axes[subplot_index]

    def plot_subcolumn_mean_profile(self, variable, time=None, pressure_coords=True, title=None,
                                    subplot_index=(0,), log_plot=False, plot_SD=True, Xlabel=None,
                                    Mask_array=None, x_range=None, y_range=None, use_geom_mean=False, **kwargs):
        """
        This function will plot a mean vertical profile of a subcolumn variable for a given time period. The
        thick line will represent the mean profile along the subcolumns, and the shading represents one
        standard deviation about the mean.

        Parameters
        ----------
        variable: str
            The name of the variable to plot.
        time: tuple of Datetime or str
            The time period to plot. If a string, specify in the format '%Y-%m-%dT%H:%M:%S'
            If a 2-element array using the values within range.
        pressure_coords: bool
            Set to true to plot in pressure coordinates.
        title: str or None
            Set the title of the plot to this string. Set to None to provide a default title
        subplot_index: tuple
            The index of the subplot to make the plot in.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        plot_SD: bool
            Set to  True (default) in order to plot a shaded patch for mean +- SD.
        Xlabel: None or str
            X-axis label. Set to None to provide a default label.
        Mask_array: bool, int, or float (same dims as "variable")
            Set to true or to other values greater than 0 in grid cells to exclude them from
            mean and SD calculations.
        x_range: tuple, list, or None
            The x range of the plot.
        y_range: tuple, list, or None
            The y range of the plot.
        use_geom_mean: str or bool
            if True, then using geometric mean and SD, If False, using arithmetic mean and SD, if "auto"
            then choosing based on typical variable scales (e.g., geometric for reflectivity and backscatter,
            and arithmetic for V_D.
        kwargs

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.

        """
        vars_for_gmean = ["alpha", "beta", "OD", "Ze", "backscatter", "extinction", "reflectivity"]

        if isinstance(use_geom_mean, str):
            if use_geom_mean == "auto":
                if np.any([x in variable for x in vars_for_gmean]):
                    use_geom_mean = True
                else:
                    use_geom_mean = False
            else:
                print("'use_geom_mean' is an unknown string. Using arithmetic mean and SD")
                use_geom_mean = False

        ds_name = [x for x in self._obj.keys()][0]
        x_variable = self.model.time_dim
        if time is not None:
            if np.logical_or(type(time) is tuple, type(time) is str):
                time = np.array(time)
            if time.size == 1:
                my_ds = self._obj[ds_name].sel({x_variable: time}, method='nearest')
            else:
                time_ind = np.logical_and(self._obj[ds_name][x_variable] >= time[0],
                                          self._obj[ds_name][x_variable] < time[1])
                my_ds = self._obj[ds_name].isel({x_variable: time_ind})
        else:
            my_ds = self._obj[ds_name]

        if pressure_coords:
            y_variable = my_ds[self.model.p_field]
            y_label = 'Pressure [hPa]'
        else:
            y_variable = my_ds[self.model.z_field]
            y_label = 'Height [m]'
        if len(y_variable.shape) > 1:
            y_variable = np.nanmean(y_variable, axis=0)

        x_variable = my_ds[variable].squeeze().values
        x_variable = np.ma.masked_where(~np.isfinite(x_variable), x_variable)
        if Mask_array is not None:
            Mask_array = Mask_array.squeeze()  # prevent singleton dimension issues
            if Mask_array.shape == x_variable.shape:
                x_variable = np.where(Mask_array <= 0, x_variable, np.nan)
            else:
                print("Mask dimensions " + str(Mask_array.shape) +
                      " are different than in the requested field " +
                      str(x_variable.shape) + " - ignoring mask")

        if 'Ze' in variable:
            with warnings.catch_warnings():  # Ignore "mean of slice" warning common with nan values.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if len(x_variable.shape) == 2:
                    x_var, x_err = self.calc_mean_and_sd(10**(x_variable / 10), use_geom_mean, axis=0)
                elif len(x_variable.shape) == 3:
                    x_var, x_err = self.calc_mean_and_sd(10**(x_variable / 10), use_geom_mean, axis=(0, 1))
            x_label = ''
            Xscale = 'linear'  # treating dBZ as linear for plotting
            x_fill = np.array(10 * np.log10([x_var - x_err, x_var + x_err]))
            x_fill[0] = np.where(x_var > x_err, x_fill[0], 10 * np.log10(np.finfo(float).eps))
        else:
            with warnings.catch_warnings():  # Ignore "mean of slice" warning common with nan values.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if len(x_variable.shape) == 2:
                    x_var, x_err = self.calc_mean_and_sd(x_variable, use_geom_mean, axis=0)
                elif len(x_variable.shape) == 3:
                    x_var, x_err = self.calc_mean_and_sd(x_variable, use_geom_mean, axis=(0, 1))
            x_fill = np.array([x_var - x_err, x_var + x_err])
            if log_plot:
                x_label = 'log '
                Xscale = 'log'
            else:
                x_label = ''
                Xscale = 'linear'
        x_lim = np.array([np.nanmin(x_fill[0]) * 0.95,
                          np.nanmax(x_fill[1] * 1.05)])

        # Choose label based on variable
        hyd_met = ''
        hyd_types = ['cl', 'ci', 'pl', 'pi', 'tot']
        for hyd in hyd_types:
            if hyd in variable:
                hyd_met = hyd

        if Xlabel is None:
            x_label = self.set_axis_label(my_ds, variable)
        else:
            x_label = Xlabel

        if plot_SD is True:
            self.axes[subplot_index].fill_betweenx(y_variable, x_fill[0], x_fill[1],
                                                   **kwargs)
        if 'alpha' in kwargs.keys():
            kwargs['alpha'] = 1
        if 'Ze' in variable:
            self.axes[subplot_index].plot(10 * np.log10(x_var), y_variable, **kwargs)
        else:
            self.axes[subplot_index].plot(x_var, y_variable, **kwargs)

        if title is None:
            self.axes[subplot_index].set_title(time)
        else:
            self.axes[subplot_index].set_title(title)

        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_xscale(Xscale)
        if y_range is not None:
            self.axes[subplot_index].set_ylim(y_range)
        if x_range is not None:
            self.axes[subplot_index].set_xlim(x_range)
        else:
            self.axes[subplot_index].set_xlim(x_lim)

        return self.axes[subplot_index]

    def plot_instrument_mean_profile(self, instrument, variable, time_range=None, pressure_coords=True,
                                     title=None, subplot_index=(0,), log_plot=False, plot_SD=True,
                                     Xlabel=None, Mask_array=None, x_range=None, y_range=None,
                                     use_geom_mean=False, **kwargs):
        """
        This function will plot a mean vertical profile of an instrument variable averaged over a given
        time period. The thick line will represent the mean profile along the given period, and the
        shading represents one standard deviation about the mean.

        Parameters
        ----------
        instrument: :py:mod:`emc2.core.Instrument`
            The Instrument class that you wish to plot.
        variable: str
            The name of the variable to plot.
        time_range: datetime64 or None
            Two-element array with starting and ending of time range; use the full data range when None.
        pressure_coords: bool
            Set to true to plot in pressure coordinates.
        title: str or None
            Set the title of the plot to this string. Set to None to provide a default title
        subplot_index: tuple
            The index of the subplot to make the plot in.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        plot_SD: bool
            Set to  True (default) in order to plot a shaded patch for mean +- SD.
        Xlabel: None or str
            X-axis label. Set to None to provide a default label.
        Mask_array: bool, int, or float (same dims as "variable")
            Set to true or to other values greater than 0 in grid cells to exclude them from
            mean and SD calculations.
        x_range: tuple, list, or Non
            The x range of the plot.
        y_range: tuple, list, or None
            The y range of the plot.
        use_geom_mean: str or bool
            if True, then using geometric mean and SD, If False, using arithmetic mean and SD, if "auto"
            then choosing based on typical variable scales (e.g., geometric for reflectivity and backscatter,
            and arithmetic for V_D.
        kwargs

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        """
        vars_for_gmean = ["alpha", "beta", "OD", "Ze", "backscatter", "extinction", "reflectivity"]
        if isinstance(use_geom_mean, str):
            if use_geom_mean == "auto":
                if np.any([x in variable for x in vars_for_gmean]):
                    use_geom_mean = True
                else:
                    use_geom_mean = False
            else:
                print("'use_geom_mean' is an unknown string. Using arithmetic mean and SD")
                use_geom_mean = False

        my_ds = instrument.ds
        if 'range' in my_ds.keys():
            y_variable = "range"
        elif 'altitude' in my_ds.keys():
            y_variable = "altitude"
        elif 'height' in my_ds.keys():
            y_variable = "height"
        y_variable = my_ds[y_variable]
        y_label = 'Height [m]'

        if time_range is None:
            x_variable = my_ds[variable].values
        else:
            time_ind = np.logical_and(my_ds.time >= time_range[0], my_ds.time < time_range[1])
            x_variable = my_ds[variable].isel(time=time_ind)
        x_variable = np.ma.masked_where(~np.isfinite(x_variable), x_variable)
        if Mask_array is not None:
            if Mask_array.shape == x_variable.shape:
                x_variable = np.where(Mask_array <= 0, x_variable, np.nan)
            else:
                print("Mask dimensions " + str(Mask_array.shape) +
                      " are different than in the requested field " +
                      str(x_variable.shape) + " - ignoring mask")

        if 'Ze' in variable:
            with warnings.catch_warnings():  # Ignore "mean of slice" warning common with nan values.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_var, x_err = self.calc_mean_and_sd(10**(x_variable / 10), use_geom_mean, axis=0)
            x_label = ''
            Xscale = 'linear'  # treating dBZ as linear for plotting
            x_fill = np.array(10 * np.log10([x_var - x_err, x_var + x_err]))
            x_fill[0] = np.where(x_var > x_err, x_fill[0], 10 * np.log10(np.finfo(float).eps))
        else:
            with warnings.catch_warnings():  # Ignore "mean of slice" warning common with nan values.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_var, x_err = self.calc_mean_and_sd(x_variable, use_geom_mean, axis=0)
            x_fill = np.array([x_var - x_err, x_var + x_err])
            if log_plot:
                Xscale = 'log'
            else:
                Xscale = 'linear'
        x_lim = np.array([np.nanmin(x_fill[0]) * 0.95,
                          np.nanmax(x_fill[1] * 1.05)])

        if Xlabel is None:
            x_label = self.set_axis_label(my_ds, variable)
        else:
            x_label = Xlabel

        if plot_SD is True:
            self.axes[subplot_index].fill_betweenx(y_variable, x_fill[0], x_fill[1],
                                                   **kwargs)
        if 'alpha' in kwargs.keys():
            kwargs['alpha'] = 1
        if 'Ze' in variable:
            self.axes[subplot_index].plot(10 * np.log10(x_var), y_variable, **kwargs)
        else:
            self.axes[subplot_index].plot(x_var, y_variable, **kwargs)

        if title is None:
            self.axes[subplot_index].set_title('%s' % time_range)
        else:
            self.axes[subplot_index].set_title(title)

        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_xscale(Xscale)
        if y_range is not None:
            self.axes[subplot_index].set_ylim(y_range)
        if x_range is not None:
            self.axes[subplot_index].set_xlim(x_range)
        else:
            self.axes[subplot_index].set_xlim(x_lim)
