import warnings
warnings.filterwarnings("ignore")

import netCDF4
import glob
import copy 
import pandas as pd
import scipy
from scipy import interpolate
import numpy as np
import xarray as xr


def save_to_nc(my_e3sm,kazr_DBZE_BINS_ground,N_sub,CF_3D,SR_4D,cfaddbz35_cal_alltime,cfadSR_cal_alltime,outdir,file_name_array):

    """
    save processed CFADs into nc file

    Parameters
    ----------
    my_e3sm: func:`emc2.core.Model` class
        The model 
    
    Returns
    -------
    kazr_DBZE_BINS_ground: float 
        radar CFAD bins, unit: dBZ  
        
    N_sub: int
        number of subcolumns
        
    CF_3D: float 
        cloud fraction, unit: none
        
        
    SR_4D : float
        calculated lidar scattering ratio, unit: none
        
        
    cfaddbz35_cal_alltime: float
        radar cfad, unit: none
        
        
    cfadSR_cal_alltime: float
        lidar SR cfad, unit: none        
        
 
    outdir: string
        output file directory
        
        
    file_name_array: string
        output file name    
        
 
    """        
    
    
    
    my_e3sm_ds_add=my_e3sm.ds
    #my_e3sm_ds_add = my_e3sm_ds_add.expand_dims(cosp_ze=kazr_DBZE_BINS_ground)
    #my_e3sm_ds_add = my_e3sm_ds_add.expand_dims(cosp_scol_emc2=np.arange(N_sub))
    my_e3sm_ds_add.assign_coords({"cosp_dbz": ("cosp_dbz", kazr_DBZE_BINS_ground)})
    my_e3sm_ds_add["CFAD_Radar"]=(['cosp_ht', 'cosp_ze'], cfaddbz35_cal_alltime)
    my_e3sm_ds_add["CFAD_SR"]=(['cosp_ht', 'cosp_sr'], cfadSR_cal_alltime)
    print('saved CFADs')
    my_e3sm_ds_add["cloud_fraction"]=(['time', 'cosp_ht', 'ncol'],  CF_3D)
    print('saved CF')
    #my_e3sm_ds_add["SR"]=(['cosp_scol_emc2', 'time', 'cosp_ht', 'ncol'],  SR_4D)
    
    output_file=f'{outdir}{file_name_array}'
    my_e3sm_ds_add.to_netcdf(path=output_file, mode='w', format='NETCDF4')
    print('Output saved: ', output_file)
   