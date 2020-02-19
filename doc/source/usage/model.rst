================================
Construction of the Model object
================================

EMC^2 uses the Model object to know what fields to load from a given
file in order to obtain the required information. In order to define your
Model object, it is highly recommended to use class inheritance. For example::

$ class ModelE(Model):
$    def __init__(self, file_path):
$        """
$        This loads a ModelE simulation with all of the necessary
$        parameters for EMC^2 to run.
$
$        Parameters
$        ----------
$        file_path: str
$            Path to a ModelE simulation.
$        """


In particular, EMC^2 will require information about the mixing ratio and
the number concentration of four
different species: cloud liquid (*cl*), cloud ice (*cl*), precipitating liquid (*pl*),
and precipitating ice (*pi*). These precipitation classes are commonly used
in many models in order to represent the cloud microphysical properties. EMC^2
derives the radar and lidar parameters for all 4 of these species. First,
in order to specify the fields that EMC^2 needs to look for, certain entries
whose keys correspond to the names of these four precipitation species must
be specified::

$   # Names of mixing ratios of species
$   self.q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
$
$   # Number concentration of each species
$   self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
$
$   # Convective fraction
$   self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
$   # Stratiform fraction
$   self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
$   # Effective radius
$   self.re_fields = {'cl': 're_mccl', 'ci': 're_mcci', 'pi': 're_mcpi', 'pl': 're_mcpl'}
$   # Convective mixing ratio
$   self.q_names_convective = {'cl': 'QCLmc', 'ci': 'QCImc', 'pl': 'QPLmc', 'pi': 'QPImc'}
$   # Stratiform mixing ratio
$   self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
$

In addition, other fields must also be specified::

$   # Water vapor mixing ratio
$   self.q_field = "q"
$   # Pressure
$   self.p_field = "p_3d"
$   # Height
$   self.z_field = "z"
$   # Temperature
$   self.T_field = "t"
$   # Name of height dimension
$   self.height_dim = "plm"
$   # Name of time dimension
$   self.time_dim = "time"

What if your model does not produce output for all 4 species, as is common
in many GCMs? Simply place in zero arrays that are the same shape as your
model fields!