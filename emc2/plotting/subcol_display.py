import numpy as np
import matplotlib.pyplot as plt

from act.plotting import Display


class SubcolumnDisplay(Display):
    """
    This class contains modules for displaying the generated subcolumn parameters as quicklook
    plots. It is inherited from `ACT <https://arm-doe.github.io/ACT>`_'s Display object. For more
    information on the Display object and its attributes and parameters, click `here
    <https://arm-doe.github.io/ACT/API/generated/act.plotting.plot.Display.html>`_. In addition to the
    methods in :code:`Display`, :code:`SubcolumnDisplay` has the following attributes and methods:

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
    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model: emc2.core.Model
            The model containing the subcolumn data to plot.

        Additional keyword arguments are passed into act.plotting.plot.Display's constructor.
        """
        if 'ds_name' not in kwargs.keys():
            ds_name = model.model_name
        else:
            ds_name = kwargs.pop('ds_name')
        super().__init__(model.ds, ds_name=ds_name, **kwargs)
        self.model = model

    def plot_subcolumn_timeseries(self, variable,
                                  column_no, pressure_coords=True, title=None,
                                  subplot_index=(0, ), **kwargs):
        """
        Plots timeseries of subcolumn parameters for a given variable and subcolumn.

        Parameters
        ----------
        variable: str
            The subcolumn variable to plot.
        column_no: int
            The subcolumn number to plot.
        pressure_coords: bool
            Set to true to plot in pressure coordinates, false to height coordinates.
        title: str or None
            The title of the plot. Set to None to have EMC^2 generate a title for you.
        subplot_index: tuple
            The index of the subplot to make the plot in.

        Additional keyword arguments are passed into matplotlib's matplotlib.pyplot.pcolormesh.

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        """
        ds_name = [x for x in self._arm.keys()][0]
        my_ds = self._arm[ds_name].sel(subcolumn=column_no)
        x_variable = self.model.time_dim
        if pressure_coords:
            y_variable = self.model.height_dim
        else:
            y_variable = self.model.z_field

        x_label = 'Time [UTC]'
        if "long_name" in my_ds[y_variable].attrs and "units" in my_ds[y_variable].attrs:
            y_label = '%s [%s]' % (my_ds[y_variable].attrs["long_name"],
                                   my_ds[y_variable].attrs["units"])
        else:
            y_label = y_variable

        cbar_label = '%s [%s]' % (my_ds[variable].attrs["long_name"], my_ds[variable].attrs["units"])
        if pressure_coords:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values
            x, y = np.meshgrid(x, y)
        else:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values.T
            p = my_ds[self.model.height_dim].values
            x, p = np.meshgrid(x, p)
        mesh = self.axes[subplot_index].pcolormesh(x, y, my_ds[variable].values.T, **kwargs)
        if title is None:
            self.axes[subplot_index].set_title(self.model.model_name + ' ' +
                                               np.datetime_as_string(self.model.ds.time[0].values))
        else:
            self.axes[subplot_index].set_title(title)

        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        cbar = plt.colorbar(mesh, ax=self.axes[subplot_index])
        cbar.set_label(cbar_label)
        return self.axes[subplot_index]

    def plot_single_profile(self, variable, time, pressure_coords=True, title=None,
                               subplot_index=(0,), log_plot=False):
        return

    def plot_subcolumn_mean_profile(self, variable, time, pressure_coords=True, title=None,
                               subplot_index=(0,), log_plot=False, **kwargs):
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
        pressure_coords: bool
            Set to true to plot in pressure coordinates.
        title: str or None
            Set the title of the plot to this string. Set to None to provide a default title
        subplot_index: tuple
            The index of the subplot to make the plot in.
        log_plot: bool
            Set to true to plot variable in logarithmic space.
        kwargs

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.

        """

        ds_name = [x for x in self._arm.keys()][0]
        my_ds = self._arm[ds_name].sel(time=time, method='nearest')
        if pressure_coords:
            y_variable = my_ds[self.model.p_field]
            y_label = 'Pressure [hPa]'
        else:
            y_variable = my_ds[self.model.height_dim]
            y_label = 'Height [hPa]'

        x_variable = my_ds[variable].values
        x_variable = np.where(np.isfinite(x_variable), x_variable, np.nan)

        if 'Ze' in variable:
            x_var = np.nanmean(x_variable, axis=0)
            x_err = np.nanstd(x_variable, ddof=0, axis=0)
            x_lim = np.array([np.nanmin(np.floor(x_var - x_err)),
                              np.nanmax(np.ceil(x_var + x_err))])
            x_label = ''
        elif log_plot:
            x_var = np.nanmean(np.log10(x_variable), axis=0)
            x_err = np.nanstd(np.log10(x_variable), ddof=0, axis=0)
            x_lim = np.array([np.nanmin(10 ** (np.floor(x_var - x_err))).min(),
                              np.nanmax(10 ** (np.ceil(x_var + x_err))).max()])
            x_label = 'log '
        else:
            x_var = np.nanmean(x_variable, axis=0)
            x_err = np.nanstd(x_variable, ddof=0, axis=0)
            x_lim = np.array([np.nanmin(np.floor(x_var - x_err)), np.nanmax(np.ceil(x_var + x_err))])
            x_label = ''

        # Choose label based on variable
        hyd_met = ''
        hyd_types = ['cl', 'ci', 'pl', 'pi', 'tot']
        for hyd in hyd_types:
            if hyd in variable:
                hyd_met = hyd

        variables = ['Ze', 'Vd', 'sigma_d', 'od', 'beta', 'alpha']

        for var in variables:
            if var in variable:
                if var == 'Ze':
                    x_label += '$Z_{e, %s}$ [dBZ]' % hyd_met
                elif var == 'Vd':
                    x_label += '$V_{d, %s}$ [m/s]' % hyd_met
                elif var == 'sigma_d':
                    x_label += '$\sigma_{d, %s}$ [m/s]' % hyd_met
                elif var == 'od':
                    x_label += '$\tau_{%s}$' % hyd_met
                elif var == 'beta':
                    x_label += '$\beta_{%s}$ [$m^{-2}$]' % hyd_met
                elif var == 'alpha':
                    x_label += '$\alpha_{%s}$ [$m^{-2}$]' % hyd_met

        if x_label == '' or x_label == 'log':
            x_label = variable

        self.axes[subplot_index].plot(x_var, y_variable)
        self.axes[subplot_index].fill_betweenx(y_variable, x_var - x_err, x_var + x_err,
                                               alpha=0.5, color='deepskyblue')
        if title is None:
            self.axes[subplot_index].set_title(time)
        else:
            self.axes[subplot_index].set_title(title)

        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        self.axes[subplot_index].set_xlim(x_lim)
        return self.axes[subplot_index]