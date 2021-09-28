"""
===============
emc2.core.Model
===============

This module contains the Model class and example Models for your use.

"""
import xarray as xr
import numpy as np

from act.io.armfiles import read_netcdf
from .instrument import ureg, quantity
from netCDF4 import Dataset

try:
    from wrf import tk, getvar, ALL_TIMES
    WRF_PYTHON_AVAILABLE = True
except ImportError:
    WRF_PYTHON_AVAILABLE = False


class Model():
    """
    This class stores the model specific parameters for the radar simulator.

    Attributes
    ----------
    Rho_hyd: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the density of said hydrometeors in :math:`kg\ m^{-3}`
    fluffy: dict
       A dictionary whose keys are the names of the model's ice hydrometeor classes and
       whose values are the ice fluffiness factor for the fwd calculations using r_e,
       where values of 0 - equal volume sphere, 1 - fluffy sphere i.e., diameter = maximum dimension.
    lidar_ratio: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the lidar_ratio of said hydrometeors.
    vel_param_a: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the :math:`a` parameters to the equation :math:`V = aD^b` used to
       calculate terminal velocity corresponding to each hydrometeor.
    vel_param_b: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the :math:`b` parameters to the equation :math:`V = aD^b` used to
       calculate terminal velocity corresponding to each hydrometeor.
    N_field: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the number concentrations in :math:`cm^{-3}` corresponding to
       each hydrometeor class.
    T_field: str
       A string containing the name of the temperature field in the model.
    q_field: str
       A string containing the name of the water vapor mixing ratio field (in kg/kg) in the model.
    p_field: str
       A string containing the name of the pressure field (in mbar) in the model.
    z_field: str
       A string containing the name of the height field (in m) in the model.
    conv_frac_names: dict
       A dictionary containing the names of the convective fraction corresponding to each
       hydrometeor class in the model.
    strat_frac_names: dict
       A dictionary containing the names of the stratiform fraction corresponding to each
       hydrometeor class in the model.
    conv_frac_names_for_rad: dict
       A dictionary containing the names of the convective fraction corresponding to each
       hydrometeor class in the model for the radiation scheme.
    strat_frac_names_for_rad: dict
       A dictionary containing the names of the stratiform fraction corresponding to each
       hydrometeor class in the model for the radiation scheme.
    conv_re_fields: dict
       A dictionary containing the names of the effective radii of each convective
       hydrometeor class
    strat_re_fields: dict
       A dictionary containing the names of the effective radii of each stratiform
       hydrometeor class
    time_dim: str
       The name of the time dimension in the model.
    height_dim: str
       The name of the height dimension in the model.
    model_name: str
       The name of the model (used for plotting).
    x_dim: str
       The name of the x dimension of the model.
    y_dim: str
       The name of the y dimension of the model.
    """

    def __init__(self):
        self.Rho_hyd = {}
        self.fluffy = {}
        self.lidar_ratio = {}
        self.LDR_per_hyd = {}
        self.vel_param_a = {}
        self.vel_param_b = {}
        self.q_names_convective = {}
        self.q_names_stratiform = {}
        self.N_field = {}
        self.T_field = ""
        self.q_field = ""
        self.p_field = ""
        self.z_field = ""
        self.qp_field = {}
        self.conv_frac_names = {}
        self.strat_frac_names = {}
        self.conv_frac_names_for_rad = {}
        self.strat_frac_names_for_rad = {}
        self.conv_re_fields = {}
        self.strat_re_fields = {}
        self.ds = None
        self.time_dim = "time"
        self.height_dim = "height"
        self.model_name = "empty_model"
        self.x_dim = None
        self.y_dim = None
        self.lat_name = None
        self.lon_name = None
        self.consts = {"c": 299792458.0,  # m/s
                       "R_d": 287.058,  # J K^-1 Kg^-1
                       "g": 9.80665,  # m/s^2
                       "Avogadro_c": 6.022140857e23,
                       "R": 8.3144598}  # J K^-1 mol^-1

    def _add_vel_units(self):
        for my_keys in self.vel_param_a.keys():
            self.vel_param_a[my_keys] = self.vel_param_a[my_keys] * (
                ureg.meter ** (1 - self.vel_param_b[my_keys].magnitude) / ureg.second)

    def _prepare_variables(self):
        for variable in self.ds.variables.keys():
            attrs = self.ds[variable].attrs
            try:
                self.ds[variable] = self.ds[variable].astype('float64')
            except TypeError:
                continue
            self.ds[variable].attrs = attrs

    def _crop_time_range(self, time_range, alter_coord=None):
        """
        Crop model output time range (time coords must be of np.datetime64 datatype).
        Can significantly cut subcolumn processing time.

        Parameters
        ----------
        time_range: tuple, list, or array, typically in datetime64 format
            Two-element array with starting and ending of time range.
        alter_coord: str or None
            Alternative time coords to use for cropping.
        """
        if alter_coord is None:
            t_coords = self.time_dim
        else:
            t_coords = alter_coord
        time_ind = np.logical_and(self.ds[t_coords] >= time_range[0],
                                  self.ds[t_coords] < time_range[1])
        if np.sum(time_ind) == 0:
            self.ds.close()
            print("The requested time range: {0} to {1} is out of the \
            model output range; Ignoring crop request.".format(time_range[0], time_range[1]))
        else:
            self.ds = self.ds.isel({self.time_dim: time_ind})

    @property
    def hydrometeor_classes(self):
        """
        The list of hydrometeor classes.
        """
        return list(self.N_field.keys())

    @property
    def num_hydrometeor_classes(self):
        """
        The number of hydrometeor classes
        """
        return len(list(self.N_field.keys()))

    @property
    def num_subcolumns(self):
        """
        Gets the number of subcolumns in the model. Will
        return 0 if the number of subcolumns has not yet been set.
        """
        if 'subcolumn' in self.ds.dims.keys():
            return self.ds.dims['subcolumn']
        else:
            return 0

    @num_subcolumns.setter
    def num_subcolumns(self, a):
        """
        This will set the number of subcolumns in the simulated radar output.
        This is a handy shortcut for setting the number of subcolumns if you
        do not want to use any of the functions in the simulator module to
        do so.
        """
        subcolumn = xr.DataArray(np.arange(a), dims='subcolumn')
        self.ds['subcolumn'] = subcolumn

    def subcolumns_to_netcdf(self, file_name):
        """
        Saves all of the simulated subcolumn parameters to a netCDF file.

        Parameters
        ----------
        file_name: str
            The name of the file to save to.
        """
        # Set all relevant variables to save:
        vars_to_keep = ["sub_col", "subcol", "strat_", "conv_", "_tot", "_ext", "_mask", "_min", "mpr", "fpr"]
        var_dict = {}
        for my_var in self.ds.variables.keys():
            if np.any([x in my_var for x in vars_to_keep]):
                var_dict[my_var] = self.ds[my_var]
        out_ds = xr.Dataset(var_dict)
        out_ds.to_netcdf(file_name)

    def load_subcolumns_from_netcdf(self, file_name):
        """
        Load all of the subcolumn data from a previously saved netCDF file.
        The dataset being loaded must match the current number of subcolumns if there are any
        generated.

        Parameters
        ----------
        file_name: str
            Name of the file to save.

        """
        my_file = xr.open_dataset(file_name)
        self.ds = xr.merge([self.ds, my_file])
        my_file.close()


class ModelE(Model):
    def __init__(self, file_path, time_range=None):
        """
        This loads a ModelE simulation with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to a ModelE simulation.
        """
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3), 'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3), 'pi': 250. * ureg.kg / (ureg.m**3)}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.height_dim = "p"
        self.time_dim = "time"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmcr', 'ci': 'cldmcr',
                                        'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldssr', 'ci': 'cldssr',
                                         'pl': 'cldssr', 'pi': 'cldssr'}
        self.conv_re_fields = {'cl': 're_mccl', 'ci': 're_mcci', 'pi': 're_mcpi', 'pl': 're_mcpl'}
        self.strat_re_fields = {'cl': 're_sscl', 'ci': 're_ssci', 'pi': 're_sspi', 'pl': 're_sspl'}
        self.q_names_convective = {'cl': 'QCLmc', 'ci': 'QCImc', 'pl': 'QPLmc', 'pi': 'QPImc'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.ds = read_netcdf(file_path)

        # Check to make sure we are loading a single column
        if 'lat' in [x for x in self.ds.dims.keys()]:
            if self.ds.dims['lat'] != 1 or self.ds.dims['lon'] != 1:
                self.ds.close()
                raise RuntimeError("%s is not an SCM run. EMC^2 will only work with SCM runs." % file_path)

            # No need for lat and lon dimensions
            self.ds = self.ds.squeeze(dim=('lat', 'lon'))

        # crop specific model output time range (if requested)
        if time_range is not None:
            if np.issubdtype(time_range.dtype, np.datetime64):
                super()._crop_time_range(time_range)
            else:
                raise RuntimeError("input time range is not in the required datetime64 data type")

        # ModelE has pressure units in mb, but pint only supports hPa
        self.ds["p_3d"].attrs["units"] = "hPa"
        self.model_name = "ModelE"


class E3SM(Model):
    def __init__(self, file_path, time_range=None):
        """
        This loads an E3SM simulation output with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to an E3SM simulation.
        """
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3), 'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3), 'pi': 250. * ureg.kg / (ureg.m**3)}
        self.fluffy = {'ci': 1.0 * ureg.dimensionless, 'pi': 1.0 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_field = "Q"
        self.N_field = {'cl': 'NUMLIQ', 'ci': 'NUMICE', 'pl': 'NUMRAI', 'pi': 'NUMSNO'}
        self.p_field = "p_3d"
        self.z_field = "Z3"
        self.T_field = "T"
        self.height_dim = "lev"
        self.time_dim = "ncol"
        self.conv_frac_names = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.strat_frac_names = {'cl': 'CLOUD', 'ci': 'CLOUD', 'pl': 'CLOUD', 'pi': 'CLOUD'}
        self.conv_frac_names_for_rad = {'cl': 'zeros_cf', 'ci': 'zeros_cf',
                                        'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.strat_frac_names_for_rad = {'cl': 'CLOUD', 'ci': 'CLOUD',
                                         'pl': 'CLOUD', 'pi': 'CLOUD'}
        self.conv_re_fields = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pi': 'zeros_cf', 'pl': 'zeros_cf'}
        self.strat_re_fields = {'cl': 'AREL', 'ci': 'AREI', 'pi': 'ADSNOW', 'pl': 'ADRAIN'}
        self.q_names_convective = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.q_names_stratiform = {'cl': 'CLDLIQ', 'ci': 'CLDICE', 'pl': 'RAINQM', 'pi': 'SNOWQM'}
        self.ds = read_netcdf(file_path)
        time_datetime64 = np.array([x.strftime('%Y-%m-%dT%H:%M') for x in self.ds["time"].values],
                                   dtype='datetime64')
        self.ds = self.ds.assign_coords(time=('ncol', time_datetime64))  # add additional time coords

        # Check to make sure we are loading a single column
        if 'lat' in [x for x in self.ds.dims.keys()]:
            if self.ds.dims['lat'] != 1 or self.ds.dims['lon'] != 1:
                self.ds.close()
                raise RuntimeError("%s is not a column dataset. EMC^2 will currently works with column data." %
                                   file_path)

            # No need for lat and lon dimensions
            self.ds = self.ds.squeeze(dim=('lat', 'lon'))

        # crop specific model output time range (if requested)
        if time_range is not None:
            if np.issubdtype(time_range.dtype, np.datetime64):
                super()._crop_time_range(time_range, alter_coord="time")
            else:
                raise RuntimeError("input time range is not in the required datetime64 data type")

        self.ds[self.p_field] = (self.ds["P0"] * self.ds["hyam"] + self.ds["PS"] * self.ds["hybm"]).T / 1e2  # hPa
        self.ds[self.p_field].attrs["units"] = "hPa"
        self.ds["zeros_cf"] = xr.DataArray(np.zeros_like(self.ds[self.p_field].values),
                                           dims=self.ds[self.p_field].dims)
        self.ds["zeros_cf"].attrs["long_name"] = "An array of zeros as only strat output is used for this model"
        for hyd in ["pl", "pi"]:
            self.ds[self.strat_re_fields[hyd]].values /= 2  # Assuming effective diameter was provided
        self.ds["rho_a"] = self.ds[self.p_field] * 1e2 / (self.consts["R_d"] * self.ds[self.T_field])
        self.ds["rho_a"].attrs["units"] = "kg / m ** 3"
        for hyd in ["cl", "ci", "pl", "pi"]:
            self.ds[self.N_field[hyd]].values *= self.ds["rho_a"].values  # convert from mass number to number
        self.model_name = "E3SM"


class WRF(Model):
    def __init__(self, file_path,
                 z_range=None, time_range=None, w_thresh=1,
                 t=None):
        """
        This load a WRF simulation and all of the necessary parameters from
        the simulation.

        Parameters
        ----------
        file_path: str
            Path to WRF simulation.
        time_range: tuple or None
            Start and end time to include. If this is None, the entire
            simulation will be included.
        z_range: numpy array or None
            The z levels of the vertical grid you want to use. By default,
            the levels are 0 m to 15000 m, increasing by 500 m.
        w_thresh: float
            The threshold of vertical velocity for defining a grid cell
            as convective.
        t: int or None
            The timestep number to subset the WRF data into. Set to None to
            load all of the data
        """
        if not WRF_PYTHON_AVAILABLE:
            raise ModuleNotFoundError("wrf-python must be installed in " +
                                      "order to read WRF data.")

        if z_range is None:
            z_range = np.arange(0., 15000., 500.)

        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3),
                        'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3),
                        'pi': 100. * ureg.kg / (ureg.m**3)}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names = {'cl': 'QCLOUD', 'ci': 'QICE',
                        'pl': 'QRAIN', 'qpi': 'QSNOW'}
        self.q_field = "QVAPOR"
        self.N_field = {'cl': 'QNCLOUD', 'ci': 'QNICE',
                        'pl': 'QNRAIN', 'pi': 'QNSNOW'}
        self.p_field = "pressure"
        self.z_field = "Z"
        self.T_field = "T"
        self.conv_frac_names = {'cl': 'conv_frac', 'ci': 'conv_frac',
                                'pl': 'conv_frac', 'pi': 'conv_frac'}
        self.strat_frac_names = {'cl': 'strat_frac', 'ci': 'strat_frac',
                                 'pl': 'strat_frac', 'pi': 'strat_frac'}
        self.conv_frac_names_for_rad = {
            'cl': 'conv_frac', 'ci': 'conv_frac',
            'pl': 'conv_frac', 'pi': 'conv_frac'}
        self.strat_frac_names_for_rad = {
            'cl': 'strat_frac', 'ci': 'strat_frac',
            'pl': 'strat_frac', 'pi': 'strat_frac'}
        self.re_fields = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                          'pi': 'strat_pi_frac', 'pl': 'strat_pl_frac'}
        self.strat_re_fields = {'cl': 'strat_cl_re', 'ci': 'strat_ci_frac',
                                'pi': 'strat_pi_re', 'pl': 'strat_pl_frac'}
        self.conv_re_fields = {'cl': 'conv_cl_re', 'ci': 'conv_ci_re',
                               'pi': 'conv_pi_re', 'pl': 'conv_pl_re'}
        self.q_names_convective = {'cl': 'qclc', 'ci': 'qcic',
                                   'pl': 'qplc', 'pi': 'qpic'}
        self.q_names_stratiform = {'cl': 'qcls', 'ci': 'qcis',
                                   'pl': 'qpls', 'pi': 'qpis'}

        ds = xr.open_dataset(file_path)
        wrfin = Dataset(file_path)
        self.ds = {}
        self.ds["pressure"] = ds["P"] + ds["PB"]
        self.ds["pressure"].attrs["units"] = "hPa"
        self.ds["Z"] = getvar(wrfin, "z", units="m", timeidx=ALL_TIMES)
        self.ds["T"] = getvar(wrfin, "tk", timeidx=ALL_TIMES)
        self.ds["T"] = self.ds["T"] + 273.15
        self.ds["T"].attrs["units"] = "K"
        W = getvar(wrfin, "wa", units="m s-1", timeidx=ALL_TIMES)
        shp = W.values.shape
        W = W.values.max(axis=1)
        W = np.transpose(np.tile(W, (shp[1], 1, 1, 1)), [1, 0, 2, 3])
        where_conv = np.where(W > w_thresh, 1, 0)
        self.ds["conv_frac"] = xr.DataArray(
            where_conv,
            dims=('Time', 'bottom_top', 'north_south', 'east_west'))
        self.ds["strat_frac"] = xr.DataArray(
            1 - where_conv,
            dims=('Time', 'bottom_top', 'north_south', 'east_west'))
        self.ds["qclc"] = ds["QCLOUD"] * where_conv
        self.ds["qcic"] = ds["QICE"] * where_conv
        self.ds["qplc"] = ds["QRAIN"] * where_conv
        self.ds["qpic"] = ds["QSNOW"] * where_conv
        self.ds["qcls"] = ds["QCLOUD"] * (1 - where_conv)
        self.ds["qcis"] = ds["QICE"] * (1 - where_conv)
        self.ds["qpls"] = ds["QRAIN"] * (1 - where_conv)
        self.ds["qpis"] = ds["QSNOW"] * (1 - where_conv)
        self.ds["QNCLOUD"] = ds["QNCLOUD"]
        self.ds["QNRAIN"] = ds["QNRAIN"]
        self.ds["QNSNOW"] = ds["QNSNOW"]
        self.ds["QNICE"] = ds["QNICE"]
        self.ds["QVAPOR"] = ds["QVAPOR"]
        self.time_dim = "Time"
        self.height_dim = "bottom_top"
        self.model_name = "WRF"
        self.lat_name = "XLAT"
        self.lon_name = "XLONG"
        wrfin.close()
        for keys in self.ds.keys():
            try:
                self.ds[keys] = self.ds[keys].drop("XTIME")
            except KeyError:
                continue
        self.ds = xr.Dataset(self.ds)
        # crop specific model output time range (if requested)
        if time_range is not None:
            super()._crop_time_range(time_range)


class DHARMA(Model):
    def __init__(self, file_path, time_range=None):
        """
        This loads a DHARMA simulation with all of the necessary parameters
        for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to a ModelE simulation.
        time_range: tuple or None
            Start and end time to include. If this is None, the entire
            simulation will be included.
        """
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3), 'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3), 'pi': 100. * ureg.kg / (ureg.m**3)}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p"
        self.z_field = "z"
        self.T_field = "t"
        self.height_dim = "hgt"
        self.time_dim = "dom_col"
        self.conv_frac_names = {'cl': 'conv_dat', 'ci': 'conv_dat',
                                'pl': 'conv_dat', 'pi': 'conv_dat'}
        self.strat_frac_names = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                                 'pl': 'strat_pl_frac', 'pi': 'strat_pi_frac'}
        self.conv_frac_names_for_rad = {'cl': 'conv_dat', 'ci': 'conv_dat',
                                        'pl': 'conv_dat', 'pi': 'conv_dat'}
        self.strat_frac_names_for_rad = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                                         'pl': 'strat_pl_frac', 'pi': 'strat_pi_frac'}
        self.conv_re_fields = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                               'pi': 'strat_pi_frac', 'pl': 'strat_pl_frac'}
        self.strat_re_fields = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                                'pi': 'strat_pi_frac', 'pl': 'strat_pl_frac'}
        self.q_names_convective = {'cl': 'conv_dat', 'ci': 'conv_dat', 'pl': 'conv_dat', 'pi': 'conv_dat'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.ds = read_netcdf(file_path)

        for variable in self.ds.variables.keys():
            my_attrs = self.ds[variable].attrs
            self.ds[variable] = self.ds[variable].astype('float64')
            self.ds[variable].attrs = my_attrs
        # es.keys():

        # Check to make sure we are loading a single column
        if 'lat' in [x for x in self.ds.dims.keys()]:
            if self.ds.dims['lat'] != 1 or self.ds.dims['lon'] != 1:
                self.ds.close()
                raise RuntimeError("%s is not an SCM run. EMC^2 will only work with SCM runs." % file_path)

            # No need for lat and lon dimensions
            self.ds = self.ds.squeeze(dim=('lat', 'lon'))

        # crop specific model output time range (if requested)
        if time_range is not None:
            super()._crop_time_range(time_range)

        self.model_name = "DHARMA"


class TestModel(Model):
    """
    This is a test Model structure used only for unit testing. It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 1, 1000) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) / (ureg.centimeter ** 3)
        heights = np.linspace(0, 11000., 1000) * ureg.meter
        temp = 15.04 * ureg.kelvin - quantity(0.00649, 'kelvin/meter') * heights + 273.15 * ureg.kelvin
        temp_c = temp.to('degC')
        p = 1012.9 * ureg.hPa * (temp / (288.08 * ureg.kelvin)) ** 5.256
        es = 0.6112 * ureg.hPa * np.exp(17.67 * temp_c.magnitude / (temp_c.magnitude + 243.5))
        qv = 0.622 * es * 1e3 / (p * 1e2 - es * 1e3)
        times = xr.DataArray(np.array([0]), dims=('time'))
        times.attrs["units"] = "seconds"
        heights = xr.DataArray(heights.magnitude[np.newaxis, :], dims=('time', 'height'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[np.newaxis, :], dims=('time', 'height'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[np.newaxis, :], dims=('time', 'height'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = temp_c.units
        temp = xr.DataArray(temp_c.magnitude[np.newaxis, :], dims=('time', 'height'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q = xr.DataArray(q.magnitude[np.newaxis, :], dims=('time', 'height'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % qv_units

        N = xr.DataArray(N.magnitude[np.newaxis, :], dims=('time', 'height'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % qv_units

        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': q, 'ncl': N, 'qpl': q, 'qci': q, 'qpi': q,
                            'time': times})
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 500. * ureg.kg / (ureg.m ** 3),
                        'pl': 1000. * ureg.kg / (ureg.m ** 3), 'pi': 250. * ureg.kg / (ureg.m ** 3)}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m ** 3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names_convective = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"


class TestConvection(Model):
    """
    This is a test Model structure used only for unit testing.
    This model has a 100% convective column from 1 km to 11 km.
    It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 1, 1000) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) * (ureg.centimeter ** -3)
        Npl = 0.001 * np.ones_like(1) * (ureg.centimeter ** -3)
        heights = np.linspace(0, 11000., 1000) * ureg.meter
        temp = 15.04 * ureg.kelvin - 0.00649 * (ureg.kelvin / ureg.meter) * heights + 273.15 * ureg.kelvin
        temp_c = temp.to('degC')
        p = 1012.9 * ureg.hPa * (temp / (288.08 * ureg.kelvin)) ** 5.256
        re_cl = 10 * np.ones_like(q) * ureg.micrometer
        re_pl = 100 * np.ones_like(q) * ureg.micrometer
        es = 0.6112 * ureg.hPa * np.exp(17.67 * temp_c.magnitude / (temp_c.magnitude + 243.5))
        qv = 0.622 * es * 1e3 / (p * 1e2 - es * 1e3) * q.units
        convective_liquid = np.logical_and(heights > 1000. * ureg.meter,
                                           temp >= 273.15 * ureg.kelvin)
        convective_ice = np.logical_and(heights > 1000. * ureg.meter,
                                        temp < 273.15 * ureg.kelvin)
        Nci = np.where(convective_ice, Npl.magnitude, 0)
        Npi = np.where(convective_ice, Npl.magnitude, 0)
        Npl = np.where(convective_liquid, Npl.magnitude, 0)
        cldmccl = np.where(convective_liquid, 1, 0.) * ureg.dimensionless
        cldmcci = np.where(convective_ice, 1, 0.) * ureg.dimensionless
        cldsscl = np.zeros_like(heights) * ureg.dimensionless
        cldssci = np.zeros_like(heights) * ureg.dimensionless
        times = xr.DataArray(np.array([0]), dims=('time'))
        times.attrs["units"] = "seconds"
        heights = xr.DataArray(heights.magnitude[np.newaxis, :], dims=('time', 'height'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[np.newaxis, :], dims=('time', 'height'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[np.newaxis, :], dims=('time', 'height'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = temp_c.units
        temp = xr.DataArray(temp_c.magnitude[np.newaxis, :], dims=('time', 'height'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[np.newaxis, :], dims=('time', 'height'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[np.newaxis, :], dims=('time', 'height'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units

        re_cl = xr.DataArray(re_cl.magnitude[np.newaxis, :], dims=('time', 'height'))
        re_cl.attrs["units"] = "micrometer"
        re_cl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        re_pl = xr.DataArray(re_pl.magnitude[np.newaxis, :], dims=('time', 'height'))
        re_pl.attrs["units"] = "micrometer"
        re_pl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        cldmccl = xr.DataArray(cldmccl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldssci.attrs["units"] = 'g kg-1'
        cldssci.attrs["long_name"] = "Stratiform cloud ice mixing ratio"

        Nci = xr.DataArray(Nci[np.newaxis, :], dims=('time', 'height'))
        Nci.attrs["units"] = "cm-3"
        Nci.attrs["long_name"] = "cloud ice particle number concentration"

        Npl = xr.DataArray(Npl[np.newaxis, :], dims=('time', 'height'))
        Npl.attrs["units"] = "cm-3"
        Npl.attrs["long_name"] = "liquid precipitation particle number concentration"

        Npi = xr.DataArray(Npi[np.newaxis, :], dims=('time', 'height'))
        Npi.attrs["units"] = "cm-3"
        Npi.attrs["long_name"] = "ice precipitation particle number concentration"
        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': q, 'ncl': N, 'nci': Nci, 'npl': Npl, 'npi': Npi,
                            'qpl': q, 'qci': q, 'qpi': q,
                            'cldmccl': cldmccl, 'cldmcci': cldmcci,
                            'cldsscl': cldsscl, 'cldssci': cldssci,
                            'cldmcpl': cldmccl, 'cldmcpi': cldmcci,
                            'cldsspl': cldsscl, 'cldsspi': cldssci,
                            'time': times, 're_cl': re_cl, 're_pl': re_pl})
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 500. * ureg.kg / (ureg.m ** 3),
                        'pl': 1000. * ureg.kg / (ureg.m ** 3), 'pi': 250. * ureg.kg / (ureg.m ** 3)}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m ** 3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names_convective = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.conv_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.strat_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"


class TestAllStratiform(Model):
    """
    This is a test Model structure used only for unit testing.
    This model has a 100% stratiform column from 1 km to 11 km.
    It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 2, 1000) * ureg.gram / ureg.kilogram
        N = 300 * np.ones_like(q) * (ureg.centimeter ** -3)
        heights = np.linspace(0, 11000., 1000) * ureg.meter
        temp = 15.04 * ureg.kelvin - 0.00649 * (ureg.kelvin / ureg.meter) * heights + 273.15 * ureg.kelvin
        temp_c = temp.to('degC').magnitude
        p = 1012.9 * ureg.hPa * (temp / (288.08 * ureg.kelvin)) ** 5.256
        es = 0.6112 * ureg.hPa * np.exp(17.67 * temp_c / (temp_c + 243.5))
        qv = 0.622 * es * 1e3 / (p * 1e2 - es * 1e3) * q.units
        re_cl = 10 * np.ones_like(q)
        re_pl = 100 * np.ones_like(q)
        stratiform_liquid = np.logical_and(heights > 1000. * ureg.meter,
                                           temp >= 273.15 * ureg.kelvin)
        stratiform_ice = np.logical_and(heights > 1000. * ureg.meter,
                                        temp < 273.15 * ureg.kelvin)
        cldsscl = np.where(stratiform_liquid, 1, 0.) * ureg.dimensionless
        cldssci = np.where(stratiform_ice, 1, 0.) * ureg.dimensionless
        cldmccl = np.zeros_like(heights) * ureg.dimensionless
        cldmcci = np.zeros_like(heights) * ureg.dimensionless
        qcl = np.where(stratiform_liquid, q, 0)
        qci = np.where(stratiform_ice, q, 0)
        times = xr.DataArray(np.array([0]), dims=('time'))
        times.attrs["units"] = "seconds"
        heights = xr.DataArray(heights.magnitude[np.newaxis, :], dims=('time', 'height'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[np.newaxis, :], dims=('time', 'height'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[np.newaxis, :], dims=('time', 'height'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = "degC"
        temp = xr.DataArray(temp_c[np.newaxis, :], dims=('time', 'height'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[np.newaxis, :], dims=('time', 'height'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[np.newaxis, :], dims=('time', 'height'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units
        qcl = xr.DataArray(qcl[np.newaxis, :], dims=('time', 'height'))
        qcl.attrs["units"] = "g kg-1"
        qcl.attrs["long_name"] = "Cloud liquid water mixing ratio"
        qci = xr.DataArray(qci[np.newaxis, :], dims=('time', 'height'))
        qci.attrs["units"] = "g kg-1"
        qci.attrs["long_name"] = "Cloud ice water mixing ratio"

        re_cl = xr.DataArray(re_cl[np.newaxis, :], dims=('time', 'height'))
        re_cl.attrs["units"] = "micrometer"
        re_cl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        re_pl = xr.DataArray(re_pl[np.newaxis, :], dims=('time', 'height'))
        re_pl.attrs["units"] = "micrometer"
        re_pl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        nci = 0. * N
        npi = 0. * N
        npl = 1e-3 * N

        nci.attrs["units"] = "cm-3"
        nci.attrs["long_name"] = "cloud ice particle number concentration"
        npl.attrs["units"] = "cm-3"
        npl.attrs["long_name"] = "liquid precipitation particle number concentration"
        npi.attrs["units"] = "cm-3"
        npi.attrs["long_name"] = "ice precipitation particle number concentration"
        cldmccl = xr.DataArray(cldmccl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldssci.attrs["units"] = 'g kg-1'
        cldssci.attrs["long_name"] = "Stratiform cloud ice mixing ratio"
        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': qcl, 'ncl': N, 'nci': nci, 'npi': npi,
                            'npl': npl, 'qpl': qcl, 'qci': qci, 'qpi': qci,
                            'cldmccl': cldmccl, 'cldmcci': cldmcci,
                            'cldsscl': cldsscl, 'cldssci': cldssci,
                            'cldmcpl': cldmccl, 'cldmcpi': cldmcci,
                            'cldsspl': cldsscl, 'cldsspi': cldssci,
                            'time': times, 're_cl': re_cl, 're_pl': re_pl})
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 500. * ureg.kg / (ureg.m ** 3),
                        'pl': 1000. * ureg.kg / (ureg.m ** 3), 'pi': 250. * ureg.kg / (ureg.m ** 3)}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m ** 3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names_convective = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.conv_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.strat_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"


class TestHalfAndHalf(Model):
    """
    This is a test Model structure used only for unit testing.
    This model has a 50% stratiform, 50% convective column from 1 km to 11 km.
    It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 1, 1000) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) * (ureg.centimeter ** -3)
        heights = np.linspace(0, 11000., 1000) * ureg.meter
        temp = 15.04 * ureg.kelvin - 0.00649 * (ureg.kelvin / ureg.meter) * heights + 273.15 * ureg.kelvin
        temp_c = temp.to('degC').magnitude
        p = 1012.9 * ureg.hPa * (temp / (288.08 * ureg.kelvin)) ** 5.256
        es = 0.6112 * ureg.hPa * np.exp(17.67 * temp_c / (temp_c + 243.5))
        qv = 0.622 * es * 1e3 / (p * 1e2 - es * 1e3) * q.units
        stratiform_liquid = np.logical_and(heights > 1000. * ureg.meter,
                                           temp >= 273.15 * ureg.kelvin)
        stratiform_ice = np.logical_and(heights > 1000. * ureg.meter,
                                        temp < 273.15 * ureg.kelvin)
        cldsscl = 0.5 * np.where(stratiform_liquid, 1, 0.) * ureg.dimensionless
        cldssci = 0.5 * np.where(stratiform_ice, 1, 0.) * ureg.dimensionless
        cldmccl = 0.5 * np.where(stratiform_liquid, 1, 0.) * ureg.dimensionless
        cldmcci = 0.5 * np.where(stratiform_ice, 1, 0.) * ureg.dimensionless
        qcl = np.where(stratiform_liquid, q, 0)
        qci = np.where(stratiform_ice, q, 0)
        times = xr.DataArray(np.array([0]), dims=('time'))
        times.attrs["units"] = "seconds"
        heights = xr.DataArray(heights.magnitude[np.newaxis, :], dims=('time', 'height'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[np.newaxis, :], dims=('time', 'height'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[np.newaxis, :], dims=('time', 'height'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = "degC"
        temp = xr.DataArray(temp_c[np.newaxis, :], dims=('time', 'height'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[np.newaxis, :], dims=('time', 'height'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[np.newaxis, :], dims=('time', 'height'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units

        qcl = xr.DataArray(qcl[np.newaxis, :], dims=('time', 'height'))
        qcl.attrs["units"] = "g kg-1"
        qcl.attrs["long_name"] = "Cloud liquid water mixing ratio"
        qci = xr.DataArray(qci[np.newaxis, :], dims=('time', 'height'))
        qci.attrs["units"] = "g kg-1"
        qci.attrs["long_name"] = "Cloud ice water mixing ratio"

        cldmccl = xr.DataArray(cldmccl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[np.newaxis, :], dims=('time', 'height'))
        cldssci.attrs["units"] = 'g kg-1'
        cldssci.attrs["long_name"] = "Stratiform cloud ice mixing ratio"
        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': q, 'ncl': N, 'qpl': q, 'qci': q, 'qpi': q,
                            'cldmccl': cldmccl, 'cldmcci': cldmcci,
                            'cldsscl': cldsscl, 'cldssci': cldssci,
                            'cldmcpl': cldmccl, 'cldmcpi': cldmcci,
                            'cldsspl': cldsscl, 'cldsspi': cldssci,
                            'time': times})
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.meter ** 3), 'ci': 500. * ureg.kg / (ureg.meter ** 3),
                        'pl': 1000. * ureg.kg / (ureg.meter ** 3), 'pi': 250. * ureg.kg / (ureg.meter ** 3)}
        self.fluffy = {'ci': 0.5 * ureg.dimensionless, 'pi': 0.5 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m ** 3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m ** 3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names_convective = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.conv_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.strat_re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"
