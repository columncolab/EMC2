import warnings
warnings.filterwarnings("ignore")


def save_cfads_to_nc(model, kazr_DBZE_BINS_ground, CF_3D,
                     cfaddbz35_cal_alltime, cfadSR_cal_alltime,
                     outdir, file_name_array):

    """
    Save processed CFADs into nc file

    Parameters
    ----------
    model: func:`emc2.core.Model` class
        The model

    Returns
    -------
    kazr_DBZE_BINS_ground: float
        radar CFAD bins, unit: dBZ
    CF_3D: float
        cloud fraction, unit: none
    cfaddbz35_cal_alltime: float
        radar cfad, unit: none
    cfadSR_cal_alltime: float
        lidar SR cfad, unit: none
    outdir: string
        output file directory
    file_name_array: string
        output file name
    """

    model_ds_add = model.ds
    model_ds_add.assign_coords({"cosp_dbz": ("cosp_dbz", kazr_DBZE_BINS_ground)})
    model_ds_add["CFAD_Radar"] = (['cosp_ht', 'cosp_ze'], cfaddbz35_cal_alltime)
    model_ds_add["CFAD_SR"] = (['cosp_ht', 'cosp_sr'], cfadSR_cal_alltime)
    print('saved CFADs')
    model_ds_add["cloud_fraction"] = (['time', 'cosp_ht', 'ncol'], CF_3D)
    print('saved CF')
    output_file = f'{outdir}{file_name_array}'
    model_ds_add.to_netcdf(path=output_file, mode='w', format='NETCDF4')
    print('Output saved: ', output_file)
