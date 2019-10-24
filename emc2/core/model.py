"""
===============
emc2.core.Model
===============

This module contains the Model class and example Models for your use.

"""
import xarray as xr
import numpy as np


class Model():
    """
    This class stores the model specific parameters for the radar simulator.

    Attributes
    ----------
    Rho_hyd: dict
       A dictionary whose keys are the names of the model's hydrometeor classes and
       whose values are the density of said hydrometeors in :math:`kg\ m^{-3}`
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
    """

    def __init__(self):
        self.Rho_hyd = {}
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
        self.ds = None

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


class ModelE(Model):
    def __init__(self, file_path):
        """
        This loads a ModelE simulation with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to a ModelE simulation.
        """
        self.Rho_hyd = {'cl': 1000., 'ci': 5000., 'pl': 1000., 'pi': 250.}
        self.lidar_ratio = {'cl': 18., 'ci': 24., 'pl': 5.5, 'pi': 24.0}
        self.LDR_per_hyd = {'cl': 0.03, 'ci': 0.35, 'pl': 0.1, 'pi': 0.40}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2., 'ci': 1., 'pl': 0.8, 'pi': 0.41}
        self.q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = xr.open_dataset(file_path, decode_times=False)

        # Check to make sure we are loading a single column
        if self.ds.dims['lat'] != 1 or self.ds.dims['lon'] != 1:
            self.ds.close()
            raise RuntimeError("%s is not an SCM run. EMC^2 will only work with SCM runs." % file_path)

        # No need for lat and lon dimensions
        self.ds = self.ds.squeeze(dim=('lat', 'lon'))


class TestModel(Model):
    """
    This is a test Model structure used only for unit testing. It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 1, 100.)
        N = 100 * np.ones_like(q)
        heights = xr.DataArray(np.linspace(0, 11000., 1000))
        temp = 15.04 - 0.00649 * heights + 273.15
        p = 101.29 * (temp / 288.08) ** 5.256
        es = 0.6112 * np.exp(17.67 * temp / (temp + 243.5))
        qv = 0.622 * es / (p - es)
        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': q, 'ncl': N, 'qpl': q, 'qci': q, 'qpi': q})
        self.Rho_hyd = {'cl': 1000., 'ci': 5000., 'pl': 1000., 'pi': 250.}
        self.lidar_ratio = {'cl': 18., 'ci': 24., 'pl': 5.5, 'pi': 24.0}
        self.LDR_per_hyd = {'cl': 0.03, 'ci': 0.35, 'pl': 0.1, 'pi': 0.40}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2., 'ci': 1., 'pl': 0.8, 'pi': 0.41}
        self.q_names_convective = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
