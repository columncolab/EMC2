"""
===============
emc2.core.Model
===============

This module contains the Model class and example Models for your use.

"""
import xarray as xr
import numpy as np

from .instrument import ureg, quantity

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
    time_dim: str
       The name of the time dimension in the model.
    height_dim: str
       The name of the height dimension in the model.
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
        self.time_dim = "time"
        self.height_dim = "height"

    def _add_vel_units(self):
        for my_keys in self.vel_param_a.keys():
            self.vel_param_a[my_keys] = self.vel_param_a[my_keys] * (
                ureg.meter ** (1 - self.vel_param_b[my_keys].magnitude) / ureg.second)

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


class ModelE(Model):
    def __init__(self, file_path):
        """
        This loads a ModelE simulation with all of the necessary parameters for EMC^2 to run.

        Parameters
        ----------
        file_path: str
            Path to a ModelE simulation.
        """
        self.Rho_hyd = {'cl': 1000. * ureg.kg/(ureg.m**3), 'ci': 5000.* ureg.kg/(ureg.m**3),
                        'pl': 1000. * ureg.kg/(ureg.m**3), 'pi': 250. * ureg.kg/(ureg.m**3)}
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24.0 * ureg.dimensionless}
        self.LDR_per_hyd = {'cl': 0.03 * 1/(ureg.kg/(ureg.m**3)),
                            'ci': 0.35 * 1/(ureg.kg/(ureg.m**3)),
                            'pl': 0.1 * 1/(ureg.kg/(ureg.m**3)),
                            'pi': 0.40 * 1/(ureg.kg/(ureg.m**3))}
        self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless}
        super()._add_vel_units()
        self.q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.height_dim = "plm"
        self.time_dim = "time"
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
        q = np.linspace(0, 1, 1000.) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) / (ureg.centimeter ** 3)
        heights = np.linspace(0, 11000., 1000) * ureg.meter
        temp = 15.04 * ureg.kelvin - quantity(0.00649, 'kelvin/meter') * heights + 273.15 * ureg.kelvin
        temp_c = temp.to('degC')
        p = 1012.9 * ureg.hPa * (temp / (288.08 * ureg.kelvin)) ** 5.256
        es = 0.6112 * ureg.hPa * np.exp(17.67 * temp_c.magnitude / (temp_c.magnitude + 243.5))
        qv = 0.622 * es * 1e3 / (p * 1e2 - es * 1e3)
        times = xr.DataArray(np.array([0]), dims=('time'))
        times.attrs["units"] = "seconds"
        heights = xr.DataArray(heights.magnitude, dims=('height'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[:, np.newaxis], dims=('height', 'time'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[:, np.newaxis], dims=('height', 'time'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = temp_c.units
        temp = xr.DataArray(temp_c.magnitude[:, np.newaxis], dims=('height', 'time'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[:, np.newaxis], dims=('height', 'time'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % qv_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[:, np.newaxis], dims=('height', 'time'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % qv_units

        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'height': heights,
                            'qcl': q, 'ncl': N, 'qpl': q, 'qci': q, 'qpi': q,
                            'time': times})
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 5000. * ureg.kg / (ureg.m ** 3),
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
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "height"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds


class TestConvection(Model):
    """
    This is a test Model structure used only for unit testing.
    This model has a 100% convective column from 1 km to 11 km.
    It is not recommended for end users.
    """
    def __init__(self):
        q = np.linspace(0, 1, 1000.) * ureg.gram / ureg.kilogram
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
        heights = xr.DataArray(heights[:, np.newaxis], dims=('height', 'time'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[:, np.newaxis], dims=('height', 'time'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[:, np.newaxis], dims=('height', 'time'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = temp_c.units
        temp = xr.DataArray(temp_c.magnitude[:, np.newaxis], dims=('height', 'time'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[:, np.newaxis], dims=('height', 'time'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[:, np.newaxis], dims=('height', 'time'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units

        re_cl = xr.DataArray(re_cl.magnitude[:, np.newaxis], dims=('height', 'time'))
        re_cl.attrs["units"] = "micrometer"
        re_cl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        re_pl = xr.DataArray(re_pl.magnitude[:, np.newaxis], dims=('height', 'time'))
        re_pl.attrs["units"] = "micrometer"
        re_pl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        cldmccl = xr.DataArray(cldmccl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldssci.attrs["units"] = 'g kg-1'
        cldssci.attrs["long_name"] = "Stratiform cloud ice mixing ratio"

        Nci = xr.DataArray(Nci[:, np.newaxis], dims=('height', 'time'))
        Nci.attrs["units"] = "cm-3"
        Nci.attrs["long_name"] = "cloud ice particle number concentration"

        Npl = xr.DataArray(Npl[:, np.newaxis], dims=('height', 'time'))
        Npl.attrs["units"] = "cm-3"
        Npl.attrs["long_name"] = "liquid precipitation particle number concentration"

        Npi = xr.DataArray(Npi[:, np.newaxis], dims=('height', 'time'))
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
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 5000. * ureg.kg / (ureg.m ** 3),
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
        self.re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
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
        q = np.linspace(0, 1, 1000.) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) * (ureg.centimeter ** -3)
        Npl = 0.001 * np.ones_like(1) * (ureg.centimeter ** -3)
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
        heights = xr.DataArray(heights[:, np.newaxis], dims=('height', 'time'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[:, np.newaxis], dims=('height', 'time'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[:, np.newaxis], dims=('height', 'time'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = "degC"
        temp = xr.DataArray(temp_c[:, np.newaxis], dims=('height', 'time'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[:, np.newaxis], dims=('height', 'time'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[:, np.newaxis], dims=('height', 'time'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units
        qcl = xr.DataArray(qcl[:, np.newaxis], dims=('height', 'time'))
        qcl.attrs["units"] = "g kg-1"
        qcl.attrs["long_name"] = "Cloud liquid water mixing ratio"
        qci = xr.DataArray(qci[:, np.newaxis], dims=('height', 'time'))
        qci.attrs["units"] = "g kg-1"
        qci.attrs["long_name"] = "Cloud ice water mixing ratio"

        re_cl = xr.DataArray(re_cl[:, np.newaxis], dims=('height', 'time'))
        re_cl.attrs["units"] = "micrometer"
        re_cl.attrs["long_name"] = "Effective radius of cloud liquid particles"

        re_pl = xr.DataArray(re_pl[:, np.newaxis], dims=('height', 'time'))
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
        cldmccl = xr.DataArray(cldmccl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[:, np.newaxis], dims=('height', 'time'))
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
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m ** 3), 'ci': 5000. * ureg.kg / (ureg.m ** 3),
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
        self.re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
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
        q = np.linspace(0, 1, 1000.) * ureg.gram / ureg.kilogram
        N = 100 * np.ones_like(q) * (ureg.centimeter ** -3)
        Npl = 0.001 * np.ones_like(1) * (ureg.centimeter ** -3)
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
        heights = xr.DataArray(heights.magnitude[:, np.newaxis], dims=('height', 'time'))
        heights.attrs['units'] = "meter"
        heights.attrs["long_name"] = "Height above MSL"

        p_units = p.units
        p = xr.DataArray(p.magnitude[:, np.newaxis], dims=('height', 'time'))
        p.attrs["long_name"] = "Air pressure"
        p.attrs["units"] = '%s' % p_units

        qv_units = qv.units
        qv = xr.DataArray(qv.magnitude[:, np.newaxis], dims=('height', 'time'))
        qv.attrs["long_name"] = "Water vapor mixing ratio"
        qv.attrs["units"] = '%s' % qv_units

        t_units = "degC"
        temp = xr.DataArray(temp_c[:, np.newaxis], dims=('height', 'time'))
        temp.attrs["long_name"] = "Air temperature"
        temp.attrs["units"] = '%s' % t_units

        q_units = q.units
        q = xr.DataArray(q.magnitude[:, np.newaxis], dims=('height', 'time'))
        q.attrs["long_name"] = "Liquid cloud water mixing ratio"
        q.attrs["units"] = '%s' % q_units

        N_units = N.units
        N = xr.DataArray(N.magnitude[:, np.newaxis], dims=('height', 'time'))
        N.attrs["long_name"] = "Cloud particle number concentration"
        N.attrs["units"] = '%s' % N_units

        qcl = xr.DataArray(qcl[:, np.newaxis], dims=('height', 'time'))
        qcl.attrs["units"] = "g kg-1"
        qcl.attrs["long_name"] = "Cloud liquid water mixing ratio"
        qci = xr.DataArray(qci[:, np.newaxis], dims=('height', 'time'))
        qci.attrs["units"] = "g kg-1"
        qci.attrs["long_name"] = "Cloud ice water mixing ratio"

        cldmccl = xr.DataArray(cldmccl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmccl.attrs["units"] = 'g kg-1'
        cldmccl.attrs["long_name"] = "Convective cloud liquid mixing ratio"
        cldmcci = xr.DataArray(cldmcci.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldmcci.attrs["units"] = 'g kg-1'
        cldmcci.attrs["long_name"] = "Convective cloud ice mixing ratio"
        cldsscl = xr.DataArray(cldsscl.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldsscl.attrs["units"] = 'g kg-1'
        cldsscl.attrs["long_name"] = "Stratiform cloud liquid mixing ratio"
        cldssci = xr.DataArray(cldssci.magnitude[:, np.newaxis], dims=('height', 'time'))
        cldssci.attrs["units"] = 'g kg-1'
        cldssci.attrs["long_name"] = "Stratiform cloud ice mixing ratio"
        my_ds = xr.Dataset({'p_3d': p, 'q': qv, 't': temp, 'z': heights,
                            'qcl': q, 'ncl': N, 'qpl': q, 'qci': q, 'qpi': q,
                            'cldmccl': cldmccl, 'cldmcci': cldmcci,
                            'cldsscl': cldsscl, 'cldssci': cldssci,
                            'cldmcpl': cldmccl, 'cldmcpi': cldmcci,
                            'cldsspl': cldsscl, 'cldsspi': cldssci,
                            'time': times})
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.meter ** 3), 'ci': 5000. * ureg.kg / (ureg.meter ** 3),
                        'pl': 1000. * ureg.kg / (ureg.meter ** 3), 'pi': 250. * ureg.kg / (ureg.meter ** 3)}
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
        self.re_fields = {'cl': 're_cl', 'ci': 're_cl', 'pl': 're_pl', 'pi': 're_pl'}
        self.q_field = "q"
        self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
        self.p_field = "p_3d"
        self.z_field = "z"
        self.T_field = "t"
        self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
        self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
        self.ds = my_ds
        self.height_dim = "height"
        self.time_dim = "time"