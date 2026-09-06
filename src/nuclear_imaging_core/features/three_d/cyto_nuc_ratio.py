import numpy as np


def cyto_nuc_ratio_cellular(c_raw_crop, c_labels_crop, nuc_labels_crop):
    """
    Calculate the cytoplasmic to nuclear ratio for each cell in the image.
    Parameters
    ----------
    c_raw_crop : 3D numpy array
        The raw image of the cytoplasmic channel.
    c_labels_crop : 3D numpy array
        The labeled image of the cytoplasmic channel.
    nuc_labels_crop : 3D numpy array
        The labeled image of the nuclear channel.

    Returns
    -------
    cyto_nuc_ratios : list
        A list of the cytoplasmic to nuclear ratio for each cell in the image.
    """


    cell_mask = c_labels_crop > 0
    nuc_mask = nuc_labels_crop > 0
    cyto_intensity_sum = np.sum(c_raw_crop[cell_mask])
    nuc_intensity_sum = np.sum(c_raw_crop[nuc_mask])
    cyto_intensity_mean = cyto_intensity_sum / np.sum(cell_mask)
    nuc_intensity_mean = nuc_intensity_sum / np.sum(nuc_mask)
    if nuc_intensity_mean == 0:
        cyto_nuc_sum_ratio = np.nan
        cyto_nuc_mean_ratio = np.nan
    else:
        cyto_nuc_sum_ratio = cyto_intensity_sum / nuc_intensity_sum
        cyto_nuc_mean_ratio = cyto_intensity_mean / nuc_intensity_mean

    return {"sum_ratio": cyto_nuc_sum_ratio, "mean_ratio": cyto_nuc_mean_ratio}


def cyto_nuc_ratio_image(chan_raw,chan_labels,nuc_labels):

    cell_mask = chan_labels > 0
    nuc_mask = nuc_labels > 0
    cyto_intensity_sum = np.sum(chan_raw[cell_mask])
    nuc_intensity_sum = np.sum(chan_raw[nuc_mask])
    cyto_intensity_mean = cyto_intensity_sum / np.sum(cell_mask)
    nuc_intensity_mean = nuc_intensity_sum / np.sum(nuc_mask)
    if nuc_intensity_mean == 0:
        cyto_nuc_sum_ratio = np.nan
        cyto_nuc_mean_ratio = np.nan
    else:
        cyto_nuc_sum_ratio = cyto_intensity_sum / nuc_intensity_sum
        cyto_nuc_mean_ratio = cyto_intensity_mean / nuc_intensity_mean

    return {"img_sum_ratio": cyto_nuc_sum_ratio, "img_mean_ratio": cyto_nuc_mean_ratio}
