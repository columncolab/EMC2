"""
==========================
emc2.core.hydrometeor_info
==========================

This stores the hydrometeor fit information that is used for the radar simulator.
These dictionaries provide global constants for the following four hydrometeor
types:
    'cl': Cloud liquid water

    'ci': Cloud ice

    'pl': Liquid precipitation

    'pi': Ice precipitation

+--------------+-----------------------------+
|Variable name | Description                 |
+==============+=============================+
|Rho_hyd       | density in :math:`kg m^{-3}`|
+--------------+-----------------------------+
|lidar_ratio   |                             |
+--------------+-----------------------------+
|LDR_per_hyd   |                             |
+--------------+-----------------------------+
| vel_param_a  | The :math:`a` coefficient in|
|              | the assumed terminal        |
|              | velocity relationship       |
|              | :math:`V = aD^b`.           |
+--------------+-----------------------------+
| vel_param_b  | The :math:`b` coefficient in|
|              | the assumed terminal        |
|              | velocity relationship       |
|              | :math:`V = aD^b`.           |
+--------------+-----------------------------+

"""
Rho_hyd = {'cl': 1000., 'ci': 5000., 'pl': 1000., 'pi': 250.}
lidar_ratio = {'cl': 18., 'ci': 24., 'pl': 5.5, 'pi': 24.0}
LDR_per_hyd = {'cl': 0.03, 'ci': 0.35, 'pl': 0.1, 'pi': 0.40}
vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
vel_param_b = {'cl': 2., 'ci': 1., 'pl': 0.8, 'pi': 0.41}
q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
#frac_names = {'cl': }
