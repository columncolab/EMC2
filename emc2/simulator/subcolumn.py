import numpy as np


def set_convective_sub_col_frac(model, N_columns, model_ds):
    """

    Parameters
    ----------

    :param N_columns:
    :param model_ds:
    :return:
    """

    num_hydrometeor_classes = len(model.conv_frac_names.keys())
    data_frac = np.zeros(model_ds[model.conv_frac_names].shape)
    data_frac = np.tile(data_frac, (N_columns, num_hydrometeor_classes, 1, 1)).T

    i = 0
    for hyd_type in model.conv_frac_names.keys():
        data_frac[i] = np.round(model_ds[model.conv_frac_names[hyd_type]] * N_columns)
        i = i + 1
