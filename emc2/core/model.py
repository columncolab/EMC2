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
from ..scattering import brandes

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
    asp_ratio_func: dict
       A dictionary that returns hydrometeor aspect ratios as a function of maximum dimension in mm.
    hyd_types: list
       list of hydrometeor classes to include in calcuations. By default set to be consistent
       with the model represented by the Model subclass.
    time_dim: str
       The name of the time dimension in the model.
    height_dim: str
       The name of the height dimension in the model.
    lat_dim: str
        Name of the latitude dimension in the model (relevant for regional output)
    lon_dim: str
        Name of the longitude dimension in the model (relevant for regional output)
    stacked_time_dim: str or None
        This attribute becomes a string of the original time dimension name only if
        stacking was required to enable EMC2 to processes a domain output (time x lat x lon).
    process_conv: bool
        If True, then processing convective model output (can typically be set to False for
        some models).
    model_name: str
       The name of the model.
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
        self.mu_field = None
        self.lambda_field = None
        self.hyd_types = []
        self.ds = None
        self.time_dim = "time"
        self.height_dim = "height"
        self.lat_dim = "lat"
        self.lon_dim = "lon"
        self.stacked_time_dim = None
        self.process_conv = True
        self.model_name = ""
        self.consts = {"c": 299792458.0,  # m/s
                       "R_d": 287.058,  # J K^-1 Kg^-1
                       "g": 9.80665,  # m/s^2
                       "Avogadro_c": 6.022140857e23,
                       "R": 8.3144598}  # J K^-1 mol^-1
        self.asp_ratio_func = {}

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
        return self.hyd_types

    @property
    def num_hydrometeor_classes(self):
        """
        The number of hydrometeor classes used
        """
        return len(self.hyd_types)

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

    def remove_subcol_fields(self, cloud_class="conv"):
        """
        Remove all subcolumn output fields for the given cloud class to save memory (mainly releveant
        for CESM and E3SM).
        """
        vars_to_drop = []
        for key in self.ds.keys():
            if np.logical_and(cloud_class in key,
                              np.any([fstr in key for fstr in ["sub_col", "subcol", "phase_mask"]])):
                vars_to_drop.append(key)
        self.ds = self.ds.drop_vars(vars_to_drop)

    def finalize_subcol_fields(self, more_fieldnames=[]):
        """
        Remove all zero values from subcolumn output fields enabling better visualization. Can be applied
        over additional fields using the more_fieldnames input parameter.
        """
        for key in self.ds.keys():
            if np.any([fstr in key for fstr in ["sub_col"] + more_fieldnames]):
                self.ds[key].values = np.where(self.ds[key].values == 0., np.nan, self.ds[key].values)

    def remove_appended_str(self, all_appended_in_lat=False):
        """
        Remove appended strings from xr.Dataset coords and fieldnames based on lat/lon coord names
        (typically required when using post-processed output data files).

        Parameters
        ----------
        all_appended_in_lat: bool
            If True using only the appended str portion to the lat_dim. Otherwise, combining
            the appended str from both the lat and lon dims (relevant if appended_str is True).
        """
        if all_appended_in_lat:
            coordinates = [self.lat_dim]
            out_coords = [x for x in self.ds.keys()]  # assuming lat is not in coords since using ncol
        else:
            coordinates = [self.lat_dim, self.lon_dim]
            out_coords = [x for x in self.ds.coords] 
        for coordinate in coordinates:
            coord_loc = np.argwhere([coordinate in x for x in out_coords]).item()
            if len(out_coords[coord_loc]) > len(coordinate):
                appended_str = out_coords[coord_loc][len(coordinate):]
                appended_coords = np.char.find(out_coords, appended_str)
                to_rename = {}
                for ind in range(len(out_coords)):
                    if appended_coords[ind] > -1:
                        to_rename[out_coords[ind]] = out_coords[ind][:appended_coords[ind]]
                out_fields = [x for x in self.ds.data_vars]
                appended_fields = np.char.find(out_fields, appended_str)
                for ind in range(len(out_fields)):
                    if appended_fields[ind] > -1:
                        to_rename[out_fields[ind]] = out_fields[ind][:appended_fields[ind]]
                self.ds = self.ds.rename(to_rename)

    def check_and_stack_time_lat_lon(self, out_coord_name="time_lat_lon", file_path=None, order_dim=True):
        """
        Stack the time dim together with the lat and lon dims (if the lat and/or lon dims are longer than 1)
        to enable EMC^2 processing of regional model output. Otherwise, squeezing the lat and lon dims (if they
        exist in dataset). Finally, the method reorder dimensions to time x height for proper processing by
        calling the "permute_dims_for_processing" class method.
        NOTE: tmp variables for lat, lon, and time are produced as xr.Datasets still have many unresolved bugs
        associated with pandas multi-indexing implemented in xarray for stacking (e.g., new GitHub issue #5692).
        Practically, after the subcolumn processing the stacking information is lost so an alternative dedicated
        method is used for unstacking

        Parameters
        ----------
        out_coord_name: str
            Name of output stacked coordinate.
        file_path: str
            Path and filename of ModelE simulation output.
        order_dim: bool
            When True, reorder dimensions to time x height for proper processing.
        """
        do_process = 0  # 0 - do nothing, 1 - stack lat+lon, 2 - stack lat dim only
        # Check to make sure we are loading a single column
        if self.lat_dim in [x for x in self.ds.dims.keys()]:
            if self.ds.dims[self.lat_dim] != 1:
                do_process = 1
            if self.lon_dim in [x for x in self.ds.dims.keys()]:
                if self.ds.dims[self.lon_dim] != 1:
                    do_process = 1
            elif do_process == 1:
                do_process = 2
            if do_process > 0:
                if file_path is None:
                    file_path = "The input filename"
                print("%s is a regional output dataset; Stacking the time, lat, "
                      "and lon dims for processing with EMC^2." % file_path)
                self.ds[self.lat_dim + "_tmp"] = \
                    xr.DataArray(self.ds[self.lat_dim].values,
                                 coords={self.lat_dim + "_tmp": self.ds[self.lat_dim].values})
                if do_process == 1:
                    self.ds[self.lon_dim + "_tmp"] = \
                        xr.DataArray(self.ds[self.lon_dim].values,
                                     coords={self.lon_dim + "_tmp": self.ds[self.lon_dim].values})
                self.ds[self.time_dim + "_tmp"] = \
                    xr.DataArray(self.ds[self.time_dim].values,
                                 coords={self.time_dim + "_tmp": self.ds[self.time_dim].values})
                if do_process == 1:
                    self.ds = self.ds.stack({out_coord_name: (self.lat_dim, self.lon_dim, self.time_dim)})
                else:
                    self.ds = self.ds.stack({out_coord_name: (self.lat_dim, self.time_dim)})
                self.stacked_time_dim, self.time_dim = self.time_dim, out_coord_name
            else:
                if self.lon_dim in [x for x in self.ds.dims.keys()]:
                    # No need for lat and lon dimensions
                    self.ds = self.ds.squeeze(dim=(self.lat_dim, self.lon_dim))
                else:
                    self.ds = self.ds.squeeze(dim=(self.lat_dim))
        if order_dim:
            self.permute_dims_for_processing()  # Consistent dim order (time x height).

    def unstack_time_lat_lon(self, order_dim=True):
        """
        Unstack the time, lat, and lon dims if they were previously stacked together
        (self.stacked_time_dim is not None). Finally, the method reorder dimensions to time x height for proper
        processing by calling the "permute_dims_for_processing" class method.
        NOTE: This is a dedicated method written because xr.Datasets still have many unresolved bugs
        associated with pandas multi-indexing implemented in xarray for stacking (e.g., new GitHub issue #5692).
        Practically, after the subcolumn processing the stacking information is lost so this is an alternative
        dedicated method.

        Parameters
        ----------
        order_dim: bool
            When True, reorder dimensions to subcolumn x time x height for proper processing.
        """
        if self.stacked_time_dim is None:
            raise TypeError("stacked_time_dim is None so dataset is apparently already unstacked!")
        out_fields = [x for x in self.ds.keys()]
        self.permute_dims_for_processing(base_order=[self.height_dim, self.time_dim], base_dim_first=False)
        if self.lon_dim + "_tmp" in self.ds.coords:
            more_dims = (self.ds[self.lon_dim + "_tmp"].dims[0],
                         self.ds[self.stacked_time_dim + "_tmp"].dims[0])
            more_shapes = (self.ds[self.lon_dim + "_tmp"].size,
                           self.ds[self.stacked_time_dim + "_tmp"].size)
        else:
            more_dims = (self.ds[self.stacked_time_dim + "_tmp"].dims[0],)
            more_shapes = (self.ds[self.stacked_time_dim + "_tmp"].size,)
        for key in out_fields:
            Attrs = self.ds[key].attrs
            Dims = self.ds[key].dims
            Shape = self.ds[key].shape
            if len(Shape) == 0:
                continue
            if self.time_dim in Dims:
                self.ds[key] = xr.DataArray(np.reshape(self.ds[key].values,
                                                       (*Shape[:-1], self.ds[self.lat_dim + "_tmp"].size,
                                                        *more_shapes)),
                                            dims=(*Dims[:-1], self.ds[self.lat_dim + "_tmp"].dims[0],
                                                  *more_dims),
                                            attrs=Attrs)
        self.ds = self.ds.drop_dims(self.time_dim)
        self.time_dim, self.stacked_time_dim = self.stacked_time_dim, None
        self.ds = self.ds.rename({self.lat_dim + "_tmp": self.lat_dim,
                                  self.time_dim + "_tmp": self.time_dim})
        if self.lon_dim + "_tmp" in self.ds.coords:
            self.ds = self.ds.rename({self.lon_dim + "_tmp": self.lon_dim})
        if order_dim:
            self.permute_dims_for_processing()  # Consistent dim order (subcolumn x time x height).

    def permute_dims_for_processing(self, base_order=None, base_dim_first=True):
        """
        Reorder dims for consistent processing such that the order is:
        subcolumn x time x height.
        Note: lat/lon dims are assumed to already be stacked with the time dim.

        Parameters
        ----------
        base_order: list or None
            List of preffered dimension order. Use default if None
        base_dim_first: bool
            Make the base dims (height and time) the first ones in the permutation if True.
        """
        if base_order is None:
            base_order = [self.time_dim, self.height_dim]
        Dims_new_order = [x for x in self.ds.dims
                          if x not in ["subcolumn"] + base_order]
        if "subcolumn" in self.ds.dims:
            base_order = ["subcolumn"] + base_order
        if base_dim_first:
            Dims_new_order = tuple(base_order + Dims_new_order)
        else:
            Dims_new_order = tuple(Dims_new_order + base_order)
        self.ds = self.ds.transpose(*Dims_new_order)

    def set_hyd_types(self, hyd_types):
        if hyd_types is None:
            if self.num_hydrometeor_classes == 0:
                raise ValueError("The '%s' Model subclass has 0 specified hydrometeor classes and "
                                 "no other 'hyd_types' variable was specified. Please check the "
                                 "Model class attributes setup." % self.model_name)
            return self.hyd_types
        else:
            return hyd_types

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
        time_range: tuple, list, or array, typically in datetime64 format
            Two-element array with starting and ending of time range.
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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.hyd_types = ["cl", "ci", "pl", "pi"]
        self.ds = read_netcdf(file_path)
        if np.logical_and("level" in self.ds.coords, not "p" in self.ds.coords):
            self.height_dim = "level"

        # crop specific model output time range (if requested)
        if time_range is not None:
            if np.issubdtype(time_range.dtype, np.datetime64):
                super()._crop_time_range(time_range)
            else:
                raise RuntimeError("input time range is not in the required datetime64 data type")

        # stack dimensions in the case of a regional output or squeeze lat/lon dims if exist and len==1
        super().check_and_stack_time_lat_lon(file_path=file_path)

        # ModelE has pressure units in mb, but pint only supports hPa
        self.ds["p_3d"].attrs["units"] = "hPa"
        self.model_name = "ModelE3"


class E3SM(Model):
    def __init__(self, file_path, time_range=None, time_dim="time", appended_str=False,
                 all_appended_in_lat=False):
        """
        This loads an E3SM simulation output with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to an E3SM simulation.
        time_range: tuple, list, or array, typically in datetime64 format
            Two-element array with starting and ending of time range.
        time_dim: str
            Name of the time dimension. Typically "time" or "ncol".
        appended_str: bool
            If True, removing appended strings added to fieldnames and coordinates during
            post-processing (e.g., in cropped regions from global simualtions).
        all_appended_in_lat: bool
            If True using only the appended str portion to the lat_dim. Otherwise, combining
            the appended str from both the lat and lon dims (relevant if appended_str is True).
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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.time_dim = time_dim
        self.conv_frac_names = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.strat_frac_names = {'cl': 'FREQL', 'ci': 'FREQI', 'pl': 'FREQR', 'pi': 'FREQS'}
        self.conv_frac_names_for_rad = {'cl': 'zeros_cf', 'ci': 'zeros_cf',
                                        'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.strat_frac_names_for_rad = {'cl': 'CLOUD', 'ci': 'CLOUD',
                                         'pl': 'FREQR', 'pi': 'FREQS'}
        self.conv_re_fields = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pi': 'zeros_cf', 'pl': 'zeros_cf'}
        self.strat_re_fields = {'cl': 'AREL', 'ci': 'AREI', 'pi': 'ADSNOW', 'pl': 'ADRAIN'}
        self.q_names_convective = {'cl': 'zeros_cf', 'ci': 'zeros_cf', 'pl': 'zeros_cf', 'pi': 'zeros_cf'}
        self.q_names_stratiform = {'cl': 'CLDLIQ', 'ci': 'CLDICE', 'pl': 'RAINQM', 'pi': 'SNOWQM'}
        self.mu_field = {'cl': 'mu_cloud', 'ci': None, 'pl': None, 'pi': None}
        self.lambda_field = {'cl': 'lambda_cloud', 'ci': None, 'pl': None, 'pi': None}
        self.hyd_types = ["cl", "ci", "pi"]
        self.process_conv = False
        self.ds = read_netcdf(file_path)
        if appended_str:
            if np.logical_and(not np.any(['ncol' in x for x in self.ds.coords]), all_appended_in_lat):
                for x in self.ds.dims:
                    if 'ncol' in x:  # ncol in dims but for some reason not in the coords
                        self.ds = self.ds.assign_coords({'ncol': self.ds[x]})
                        self.ds = self.ds.swap_dims({x: "ncol"})
                        break
            super().remove_appended_str(all_appended_in_lat)
            if all_appended_in_lat:
                self.lat_dim = "ncol"  # here 'ncol' is the spatial dim (acknowledging cube-sphere coords)

        if time_dim == "ncol":
            time_datetime64 = np.array([x.strftime('%Y-%m-%dT%H:%M') for x in self.ds["time"].values],
                                       dtype='datetime64')
            self.ds = self.ds.assign_coords(time=('ncol', time_datetime64))  # add additional time coords

        # crop specific model output time range (if requested)
        if time_range is not None:
            if np.issubdtype(time_range.dtype, np.datetime64):
                if time_dim == "ncol":
                    super()._crop_time_range(time_range, alter_coord="time")
                else:
                    super()._crop_time_range(time_range)
            else:
                raise RuntimeError("input time range is not in the required datetime64 data type")

        # Flip height coordinates in data arrays (to descending pressure levels / ascending height)
        self.ds = self.ds.assign_coords({self.height_dim: np.flip(self.ds[self.height_dim].values)})
        for key in self.ds.keys():
            if self.height_dim in self.ds[key].dims:
                rel_dim = np.argwhere([self.height_dim == x for x in self.ds[key].dims]).item()
                self.ds[key].values = np.flip(self.ds[key].values, axis=rel_dim)

        # stack dimensions in the case of a regional output or squeeze lat/lon dims if exist and len==1
        super().check_and_stack_time_lat_lon(file_path=file_path, order_dim=False)

        self.ds[self.p_field] = \
            ((self.ds["P0"] * self.ds["hyam"] + self.ds["PS"] * self.ds["hybm"]).T / 1e2).transpose(  # hPa
            *self.ds[self.T_field].dims)
        self.ds[self.p_field].attrs["units"] = "hPa"
        self.ds["zeros_cf"] = xr.DataArray(np.zeros_like(self.ds[self.p_field].values),
                                           dims=self.ds[self.p_field].dims)
        self.ds["zeros_cf"].attrs["long_name"] = "An array of zeros as only strat output is used for this model"
        for hyd in ["pl", "pi"]:
            self.ds[self.strat_re_fields[hyd]].values *= 0.5 * 1e6  # Assuming effective diameter in m was provided
        self.ds["rho_a"] = self.ds[self.p_field] * 1e2 / (self.consts["R_d"] * self.ds[self.T_field])
        self.ds["rho_a"].attrs["units"] = "kg / m ** 3"
        for hyd in ["cl", "ci", "pl", "pi"]:
            self.ds[self.N_field[hyd]].values *= self.ds["rho_a"].values / 1e6  # mass number to number [cm^-3]
            self.ds[self.strat_re_fields[hyd]].values = \
                np.where(self.ds[self.strat_re_fields[hyd]].values == 0.,
                         np.nan, self.ds[self.strat_re_fields[hyd]].values)

        self.permute_dims_for_processing()  # Consistent dim order (time x height).

        self.model_name = "E3SM"


class CESM2(E3SM):
    def __init__(self, file_path, time_range=None, time_dim="time", appended_str=False):
        """
        This loads a CESM2 simulation output with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to an E3SM simulation.
        time_range: tuple, list, or array, typically in datetime64 format
            Two-element array with starting and ending of time range.
        appended_str: bool
            If True, removing appended strings added to fieldnames and coordinates during
            post-processing (e.g., in cropped regions from global simualtions).
        """
        super().__init__(file_path, time_range, time_dim, appended_str)
        self.model_name = "CESM2"


class WRF(Model):
    def __init__(self, file_path, column_extent,
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
        column_extent: 4-tuple ints
            The horizontal start and end boundaries (x1, y1, x2, y2) of
            the column to consider.
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
            raise ModuleNotFoundError("wrf-python must be installed.")

        super().__init__()
        self.hyd_types = ["cl", "ci", "pl", "pi"]
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3),
                        'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3),
                        'pi': 250. * ureg.kg / (ureg.m**3)}

        self.fluffy = {'ci': 1.0 * ureg.dimensionless,
                       'pi': 1.0 * ureg.dimensionless}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.strat_frac_names = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                                 'pl': 'strat_pl_frac', 'pi': 'strat_pi_frac'}
        self.asp_ratio_func = {'cl': lambda x: 1, 'ci': lambda x: 1, 'pi': lambda x: 0.6, 'pl': brandes}
        self.conv_frac_names_for_rad = self.conv_frac_names
        self.strat_frac_names_for_rad = self.strat_frac_names
        self.re_fields = {'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
                          'pi': 'strat_pi_frac', 'pl': 'strat_pl_frac'}
        self.conv_frac_names_for_rad = {
            'cl': 'conv_frac', 'ci': 'conv_frac',
            'pl': 'conv_frac', 'pi': 'conv_frac'}
        self.strat_frac_names_for_rad = {
            'cl': 'strat_cl_frac', 'ci': 'strat_ci_frac',
            'pl': 'strat_pl_frac', 'pi': 'strat_pi_frac'}
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

        self.conv_re_fields = {'cl': 're_clc', 'ci': 're_cis',
                               'pl': 're_plc', 'pi': 're_pis'}
        self.strat_re_fields = {'cl': 're_cls', 'ci': 're_cis',
                                'pl': 're_pls', 'pi': 're_pis'}
        ds = xr.open_dataset(file_path)
        wrfin = Dataset(file_path)
        self.ds = {}
        self.ds["pressure"] = ds["P"] + ds["PB"]
        self.ds["Z"] = getvar(wrfin, "z", units="m", timeidx=ALL_TIMES)
        self.ds["T"] = getvar(wrfin, "tk", timeidx=ALL_TIMES)
        self.ds["T"] = self.ds["T"]
        self.ds["pressure"] = self.ds["pressure"][
            :, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2)
        self.ds["pressure"].attrs["units"] = "hPa"
        self.ds["Z"] = self.ds["Z"][
            :, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2)
        self.ds["T"] = self.ds["T"][
            :, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2)
        self.ds["T"].attrs["units"] = "K"
        self.ds["Z"].attrs["units"] = "m"
        rho = self.ds["pressure"] / (287.058 * self.ds["T"])
        W = getvar(wrfin, "wa", units="m s-1", timeidx=ALL_TIMES)
        cldfrac = getvar(wrfin, "cloudfrac", timeidx=ALL_TIMES)
        cldfrac2 = np.zeros_like(W)
        for i in range(int(W.shape[1] / 3)):
            cldfrac2[:, i, :, :] = cldfrac[0, :, :, :]
        for i in range(int(W.shape[1] / 3), 2 * int(W.shape[1] / 3)):
            cldfrac2[:, i, :, :] = cldfrac[1, :, :, :]
        for i in range(2 * int(W.shape[1] / 3), int(W.shape[1])):
            cldfrac2[:, i, :, :] = cldfrac[2, :, :, :]
        
        where_conv = np.zeros_like(cldfrac2)
        where_strat = cldfrac2
        
        num_points = (column_extent[3] - column_extent[2]) * \
            (column_extent[1] - column_extent[0])
        where_conv_column = where_conv[
            :, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].sum(axis=3).sum(axis=2)
        where_conv_strat = where_strat[
            :, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].sum(axis=3).sum(axis=2)
        cldfrac2 = cldfrac2[:, :, column_extent[0]:column_extent[1],
            column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2)
        self.ds["conv_frac"] = xr.DataArray(
            where_conv_column / num_points * cldfrac2,
             dims=('Time', 'bottom_top')).astype('float64')
        conversion_factor_qn = rho * 1e-6
        self.ds["qclc"] = xr.DataArray(
            self.ds["conv_frac"].values, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qcic"] = xr.DataArray(
            self.ds["conv_frac"].values, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qplc"] = xr.DataArray(
            self.ds["conv_frac"].values, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qpic"] = xr.DataArray(self.ds["conv_frac"].values,
                dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qcls"] = xr.DataArray(
            ds["QCLOUD"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2),
                dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qcis"] = xr.DataArray(
            ds["QICE"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2),
                dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qpls"] = xr.DataArray(
            ds["QRAIN"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2),
                dims=('Time', 'bottom_top')).astype('float64')
        self.ds["qpis"] = xr.DataArray(
            ds["QSNOW"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2),
                dims=('Time', 'bottom_top')).astype('float64')
        cldfrac2_pl = np.where(self.ds["qpls"].values > 0, cldfrac2, 0)
        self.ds["strat_pl_frac"] = xr.DataArray(
            cldfrac2_pl,
            dims=('Time', 'bottom_top')).astype('float64')
        cldfrac2_cl = np.where(self.ds["qcls"].values > 0, cldfrac2, 0)
        self.ds["strat_cl_frac"] = xr.DataArray(
            cldfrac2_cl,
            dims=('Time', 'bottom_top')).astype('float64')
        cldfrac2_ci = np.where(self.ds["qcis"].values > 0, cldfrac2, 0)
        self.ds["strat_ci_frac"] = xr.DataArray(
            cldfrac2_ci,
            dims=('Time', 'bottom_top')).astype('float64')
        cldfrac2_pi = np.where(self.ds["qpis"].values > 0, cldfrac2, 0)
        self.ds["strat_pi_frac"] = xr.DataArray(
            cldfrac2_pi,
            dims=('Time', 'bottom_top')).astype('float64')

        self.ds["QNCLOUD"] = xr.DataArray(
            ds["QNCLOUD"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2) *
            conversion_factor_qn, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["QNRAIN"] = xr.DataArray(
            ds["QNRAIN"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2) *
            conversion_factor_qn, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["QNSNOW"] = xr.DataArray(
            ds["QNSNOW"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2) *
            conversion_factor_qn, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["QNICE"] = xr.DataArray(
            ds["QNICE"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2) *
            conversion_factor_qn, dims=('Time', 'bottom_top')).astype('float64')
        self.ds["QVAPOR"] = xr.DataArray(
            ds["QVAPOR"].values[
                :, :, column_extent[0]:column_extent[1],
                column_extent[2]:column_extent[3]].mean(axis=3).mean(axis=2),
            dims=('Time', 'bottom_top')).astype('float64')
        self.time_dim = "Time"
        self.height_dim = "bottom_top"
        self.model_name = "WRF"
        self.lat_dim = "XLAT"
        self.lon_dim = "XLONG"
        self.process_conv = False
        wrfin.close()
        for keys in self.ds.keys():
            try:
                self.ds[keys] = self.ds[keys].drop("XTIME")
            except (KeyError, ValueError):
                continue
        self.ds = xr.Dataset(self.ds)

        # crop specific model output time range (if requested)
        if time_range is not None:
            super()._crop_time_range(time_range)

        # stack dimensions in the case of a regional output or squeeze lat/lon dims if exist and len==1
        super().check_and_stack_time_lat_lon(file_path=file_path)


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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.hyd_types = ["cl", "ci", "pl", "pi"]
        self.ds = xr.open_dataset(file_path)

        for variable in self.ds.variables.keys():
            my_attrs = self.ds[variable].attrs
            self.ds[variable] = self.ds[variable].astype('float64')
            self.ds[variable].attrs = my_attrs

        # crop specific model output time range (if requested)
        if time_range is not None:
            super()._crop_time_range(time_range)

        # stack dimensions in the case of a regional output or squeeze lat/lon dims if exist and len==1
        super().check_and_stack_time_lat_lon(file_path=file_path)
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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci',
                                'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci',
                                 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.conv_frac_names_for_rad = {'cl': 'cldmccl', 'ci': 'cldmcci',
                                        'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names_for_rad = {'cl': 'cldsscl', 'ci': 'cldssci',
                                         'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"
        self.hyd_types = ["cl", "ci", "pl", "pi"]


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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.hyd_types = ["cl", "ci", "pl", "pi"]
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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.hyd_types = ["cl", "ci", "pl", "pi"]
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
        self.vel_param_a = {'cl': 3e7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
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
        self.hyd_types = ["cl", "ci", "pl", "pi"]
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"
