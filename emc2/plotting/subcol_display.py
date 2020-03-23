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
            y_variable = self.model.time_dim

        x_label = 'Time [UTC]'
        if "long_name" in my_ds[y_variable].attrs and "units" in my_ds[y_variable].attrs:
            y_label = '%s [%s]' % (my_ds[y_variable].attrs["long_name"],
                                   my_ds[y_variable].attrs["units"])
        else:
            y_label = y_variable

        cbar_label = '%s [%s]' % (my_ds[variable].attrs["long_name"], my_ds[variable].attrs["units"])
        x = my_ds[x_variable].values
        y = my_ds[y_variable].values
        x, y = np.meshgrid(x, y)
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
