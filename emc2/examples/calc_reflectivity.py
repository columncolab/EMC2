import emc2
import matplotlib.pyplot as plt
import numpy as np

model_path = '/nfs/gce/projects/digr/emc2_data/allvars.SCM_AWR_linft_BT0_unNa_noaer.nc'
my_model = emc2.core.model.ModelE(model_path)
HSRL = emc2.core.instruments.KAZR('nsa')
my_model = emc2.simulator.main.make_simulated_data(my_model, HSRL, 4)

fig, ax = plt.subplots(1, 1)
my_model.ds["sub_col_OD_cl_strat"].sel(subcolumn=2).plot(
    ax=ax, cbar_kwargs={'ticks': np.linspace(-30, -15, 15), 'label': 'Reflectivity [dBZ]'},
    cmap='act_HomeyerRainbow', vmin=-30, vmax=-15)
ax.invert_yaxis()
plt.show()
