import xarray as xr
import emc2
import matplotlib.pyplot as plt

model_data_path = '/nfs/gce/projects/digr/emc2_data/allvars.SCM_AWR_linft_BT0_unNa_noaer.nc'
model_column = xr.open_dataset(model_data_path, decode_times=False)
print(model_column)
my_instrument = emc2.core.instruments.KAZR('nsa')
model_column = emc2.simulator.attenuation.calc_radar_atm_attenuation(
    my_instrument, model_column, t_field='t')
model_column = emc2.simulator.reflectivity.calc_radar_reflectivity_conv(
    my_instrument, model_column, hyd_type="ci", t_field='t')
plt.pcolormesh(model_column['Ze'].values[:, :, 0, 0].T, vmin=-10, vmax=10)
plt.colorbar()
plt.show()
model_column.close()
