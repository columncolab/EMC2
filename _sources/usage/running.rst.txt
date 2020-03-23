=================================
Creating your simulated variables
=================================

Once you have created the :py:mod:`emc2.core.Model` and :py:mod:`emc2.core.Instrument` objects creating
the simulated variables is as easy as::

$ my_model = emc2.simulator.main.make_simulated_data(my_model, my_instrument, num_columns)

++++++++++++++++++++++++++++++
Cloud microphysical parameters
++++++++++++++++++++++++++++++

Once you have done this, the model will make several variables related to the subcolumn microphysical properties
as well as the simulated moments. The microphysical properties output by EMC^2 consist of the number concentration,
liquid water mixing ratio, and cloud fraction in each subcolumn.

+--------------------------+-------------------------------------------------------------+----------------+
| Variable name            | Variable description                                        | Units          |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_q_subcolumns_cl    | | The cloud liquid water mixing                             |                |
|                          | | ratio in stratiform clouds.                               | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_q_subcolumns_ci    | | The cloud ice mixing ratio in                             |                |
|                          | | stratiform clouds.                                        | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_q_subcolumns_pl    | | The precipitating liquid mixing                           |                |
|                          | | ratio in stratiform clouds.                               | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_q_subcolumns_pi    | | The precipitating ice mixing ratio                        |                |
|                          | | in stratiform clouds.                                     | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_n_subcolumns_cl    | | The cloud liquid water particle number                    |                |
|                          | | concentration in stratiform clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_n_subcolumns_ci    | | The cloud ice particle number                             |                |
|                          | | concentration in stratiform clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_n_subcolumns_pl    | | The precipitating liquid particle concentration           |                |
|                          | | in stratiform clouds.                                     | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_n_subcolumns_pi    | | The precipitating ice particle                            |                |
|                          | | concentration in stratiform clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_frac_subcolumns_cl | | The presence of liquid water particles in                 | 0 = No, 1 = Yes|
|                          | | stratiform clouds.                                        |                |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_frac_subcolumns_ci | | The presence of ice particles from stratiform             | 0 = No, 1 = Yes|
|                          | | clouds.                                                   |                |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_frac_subcolumns_pl | | The presence of liquid precipitation from                 | 0 = No, 1 = Yes|
|                          | | stratiform clouds.                                        |                |
+--------------------------+-------------------------------------------------------------+----------------+
| strat_frac_subcolumns_pi | | The presence of ice precipitation from stratiform         | 0 = No, 1 = Yes|
|                          | | clouds.                                                   |                |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_q_subcolumns_cl     | | The cloud liquid water mixing                             |                |
|                          | | ratio in convective clouds.                               | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_q_subcolumns_ci     | | The cloud ice water mixing ratio in convective clouds     | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_q_subcolumns_pl     | | The precipitating liquid particle mixing                  |                |
|                          | | ratio in convective clouds.                               | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_q_subcolumns_pi     | | The precipitating ice particle mixing ratio               |                |
|                          | | in convective clouds.                                     | kg/kg          |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_n_subcolumns_cl     | | The cloud liquid water particle                           |                |
|                          | | concentration in convective clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_n_subcolumns_ci     | | The cloud ice particle number                             |                |
|                          | | concentration in convective clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_n_subcolumns_pl     | | The precipitation liquid particle                         |                |
|                          | | concentration in convective clouds.                       | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_n_subcolumns_pi     | | The precipitation ice particle                            |                |
|                          | | concentration in ice clouds.                              | :math:`m^{-3}` |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_frac_subcolumns_cl  | | The presence of liquid water particles in                 | 0 = No, 1 = Yes|
|                          | | convective clouds.                                        |                |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_frac_subcolumns_ci  | | The presence of ice particles from convection.            | 0 = No, 1 = Yes|
|                          |                                                             |                |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_frac_subcolumns_pl  | | The presence of the volume covered by liquid              | 0 = No, 1 = Yes|
|                          | | precipitation.                                            |                |
+--------------------------+-------------------------------------------------------------+----------------+
| conv_frac_subcolumns_pi  | | The presence of the volume covered by ice                 | 0 = No, 1 = Yes|
|                          | | precipitation.                                            |                |
+--------------------------+-------------------------------------------------------------+----------------+

+++++++++++++++++++++++++
Simulated radar variables
+++++++++++++++++++++++++

In addition, if you are simulating a radar, EMC^2 will output the following parameters for each
model generated subcolumn.

+---------------------------+-------------------------------------------------------------+----------------+
| Variable name             | Variable description                                        | Units          |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_tot_strat      | | The total radar reflectivity factor from stratiform       |                |
|                           | | clouds.                                                   | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_tot_strat      | | The doppler Velocity from all hydrometeors in stratiform  |                |
|                           | | clouds.                                                   | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_tot_strat | | The spectral width from all hydrometeors in stratiform    |                |
|                           | | clouds.                                                   | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_tot_conv       | | The total radar reflectivity factor from convective       |                |
|                           | | clouds.                                                   | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_tot_conv       | | The doppler Velocity from all hydrometeors in convective  |                |
|                           | | clouds.                                                   | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_tot_conv  | | The spectral width from all hydrometeors in convective    |                |
|                           | | clouds.                                                   | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+

For cloud liquid particles:

+---------------------------+-------------------------------------------------------------+----------------+
| Variable name             | Variable description                                        | Units          |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_cl_strat       | | The radar reflectivity factor of cloud liquid particles   |                |
|                           | | in clouds.                                                | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_cl_strat       | | The doppler Velocity from cloud liquid particles in       |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_cl_strat  | | The spectral width from cloud liquid particles in         |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_cl_conv        | | The total radar reflectivity factor of cloud liquid       |                |
|                           | | particles in convective clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_cl_conv        | | The doppler Velocity from cloud liquid particles          |                |
|                           | | in convective clouds.                                     | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_cl_conv   | | The spectral width from cloud liquid particles in         |                |
|                           | | convective clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+

For cloud ice particles:

+---------------------------+-------------------------------------------------------------+----------------+
| Variable name             | Variable description                                        | Units          |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_ci_strat       | | The radar reflectivity factor of cloud ice particles      |                |
|                           | | in clouds.                                                | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_ci_strat       | | The doppler Velocity from cloud ice particles in          |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_ci_strat  | | The spectral width from cloud ice particles in            |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_ci_conv        | | The total radar reflectivity factor of cloud ice          |                |
|                           | | particles in convective clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_ci_conv        | | The doppler Velocity from cloud ice particles             |                |
|                           | | in convective clouds.                                     | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_ci_conv   | | The spectral width from cloud ice particles in            |                |
|                           | | convective clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+

For liquid precipitation particles:

+---------------------------+-------------------------------------------------------------+----------------+
| Variable name             | Variable description                                        | Units          |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_pl_strat       | | The radar reflectivity factor of liquid precipitation     |                |
|                           | | particles in stratiform clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_pl_strat       | | The doppler Velocity of liquid precipitation              |                |
|                           | | particles in stratiform clouds.                           | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_pl_strat  | | The spectral width of liquid precipitation in             |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_pl_conv        | | The radar reflectivity factor of liquid precipitation     |                |
|                           | | particles in convective clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_pl_conv        | | The doppler Velocity of liquid precipitation particles    |                |
|                           | | in convective clouds.                                     | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_pl_conv   | | The spectral width of liquid precipitation particles      |                |
|                           | | convective clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+

For ice precipitation particles:

+---------------------------+-------------------------------------------------------------+----------------+
| Variable name             | Variable description                                        | Units          |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_pl_strat       | | The radar reflectivity factor of ice precipitation        |                |
|                           | | particles in stratiform clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_pl_strat       | | The doppler Velocity of ice precipitation                 |                |
|                           | | particles in stratiform clouds.                           | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_pl_strat  | | The spectral width of ice precipitation in                |                |
|                           | | stratiform clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Ze_pl_conv        | | The radar reflectivity factor of ice precipitation        |                |
|                           | | particles in convective clouds.                           | dBZ            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_Vd_pl_conv        | | The doppler Velocity of ice precipitation particles       |                |
|                           | | in convective clouds.                                     | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+
| sub_col_sigma_d_pl_conv   | | The spectral width of ice precipitation particles         |                |
|                           | | convective clouds.                                        | m/s            |
+---------------------------+-------------------------------------------------------------+----------------+

+++++++++++++++++++++++++
Simulated lidar variables
+++++++++++++++++++++++++

If you are simulating a lidar, EMC^2 will output the following variables:

+---------------------------+-----------------------------------------------------------+-------+
| Variable name             | Variable description                                      | Units |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_alpha_p_tot_conv  | The extinction coefficient from all convective clouds.    | /m    |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_beta_p_tot_conv   | The backscatter coefficient from all convective clouds    | /m    |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_OD_tot_conv       | The optical depth from all convective clouds              |       |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_alpha_p_tot_strat | The extinction coefficient from all stratiform clouds.    | /m    |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_beta_p_tot_strat  | The backscatter coefficient from all stratiform clouds    | /m    |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_OD_tot_strat      | The optical depth from all stratiform clouds              |       |
|                           |                                                           |       |
+---------------------------+-----------------------------------------------------------+-------+

For the cloud liquid particles:

+---------------------------+-------------------------------------------------------------+-------+
| Variable name             | Variable description                                        | Units |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_cl_conv   | | The extinction coefficient from the cloud liquid          | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_cl_conv    | | The backscatter coefficient from the cloud liquid         | /m    |
|                           | | particles in convective clouds                            |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_cl_conv        | | The optical depth from the cloud liquid particles in      |       |
|                           | | convective clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_cl_strat  | | The extinction coefficient from the cloud liquid          | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_cl_strat   | | The backscatter coefficient from all the cloud liquid     | /m    |
|                           | | particles in stratiform clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_cl_strat       | | The optical depth from cloud liquid particles in          |       |
|                           | | stratiform clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+

For the cloud ice particles:

+---------------------------+-------------------------------------------------------------+-------+
| Variable name             | Variable description                                        | Units |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_ci_conv   | | The extinction coefficient from the cloud ice             | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_ci_conv    | | The backscatter coefficient from the cloud ice            | /m    |
|                           | | particles in convective clouds                            |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_ci_conv        | | The optical depth from the cloud ice particles in         |       |
|                           | | convective clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_ci_strat  | | The extinction coefficient from the cloud ice             | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_ci_strat   | | The backscatter coefficient from all the cloud ice        | /m    |
|                           | | particles in stratiform clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_ci_strat       | | The optical depth from cloud ice particles in             |       |
|                           | | stratiform clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+

For the liquid precipitation particles:

+---------------------------+-------------------------------------------------------------+-------+
| Variable name             | Variable description                                        | Units |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_pl_conv   | | The extinction coefficient from the liquid precipitation  | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_pl_conv    | | The backscatter coefficient from the liquid precipitation | /m    |
|                           | | particles in convective clouds                            |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_pl_conv        | | The optical depth from liquid precipitation in            |       |
|                           | | convective clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_alpha_p_pl_strat  | | The extinction coefficient from liquid precipitation      | /m    |
|                           | | particles in convective clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_beta_p_pl_strat   | | The backscatter coefficient from liquid precipitation     | /m    |
|                           | | particles in stratiform clouds.                           |       |
+---------------------------+-------------------------------------------------------------+-------+
| sub_col_OD_pl_strat       | | The optical depth from liquid precipitation particles in  |       |
|                           | | stratiform clouds.                                        |       |
+---------------------------+-------------------------------------------------------------+-------+

For the ice precipitation particles:

+---------------------------+-----------------------------------------------------------+-------+
| Variable name             | Variable description                                      | Units |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_alpha_p_pl_conv   | | The extinction coefficient from the ice precipitation   | /m    |
|                           | | particles in convective clouds.                         |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_beta_p_pl_conv    | | The backscatter coefficient from the ice precipitation  | /m    |
|                           | | particles in convective clouds                          |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_OD_pl_conv        | | The optical depth from ice precipitation in             |       |
|                           | | convective clouds.                                      |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_alpha_p_pl_strat  | | The extinction coefficient from ice precipitation       | /m    |
|                           | | particles in convective clouds.                         |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_beta_p_pl_strat   | | The backscatter coefficient from ice precipitation      | /m    |
|                           | | particles in stratiform clouds.                         |       |
+---------------------------+-----------------------------------------------------------+-------+
| sub_col_OD_pl_strat       | | The optical depth from ice precipitation particles in   |       |
|                           | | stratiform clouds.                                      |       |
+---------------------------+-----------------------------------------------------------+-------+

========================
Visualization of results
========================

:code:`model.ds` is a standard xarray dataset that includes all of xarray's plotting capabilities. Therefore,
you can plot any of these parameters for a given subcolumn simply by doing::

$ model_display = act.plotting.TimeSeriesDisplay(my_model.ds.sel(subcolumn=0), figsize=(15,5))
$ model_display.plot('sub_col_Ze_cl_strat', cmap='act_HomeyerRainbow', vmin=-30, vmax=-15)
$ model_display.axes[0].invert_yaxis()

.. image:: Model_Ze.png

If we want to compare against observations, you can harness the power of the `Atmospheric Community Toolkit
<https://arm-doe.github.io/ACT>`_ to make time series displays of the outputs. For example, to plot the
reflectivity from KAZR, simply do::

$ display = act.plotting.TimeSeriesDisplay(KAZR.ds, figsize=(15,5))
$ display.plot('reflectivity_copol', cmap='act_HomeyerRainbow', subplot_index=(0, ))

.. image:: Kazr_refl.png

For more information on ACT's plotting routines click `here <https://arm-doe.github.io/ACT/API/plotting.html>`_.
In the future we plan on integrating ACT's Display module to make this process even easier.