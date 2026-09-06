import numpy as np

from .utils import gen_mask_erosion





def nuclear_radialIntensityFeatures(crop,mask_crop,
                                    nbins=10,maxdistxy=100,maxdistz=None):


    ## get masks from gen_mask_erosion
    # masks,bins_dr = gen_mask_erosion(mask_crop,nbins,maxdistxy,maxdistz)

    ## get the radial intensity features
    feat = {}
    radial_intensity = np.zeros([nbins],np.float64)
    for i in range(nbins):
        mask,bins_dr = gen_mask_erosion(mask_crop,i,nbins,maxdistxy,maxdistz)
        if mask.sum() == 0:
            radial_intensity[i] = 0
        else:
            radial_intensity[i] = np.mean(crop[mask > 0])
        feat["radial_{}".format(i)] = radial_intensity[i]


    # Calculate the average over the first 0.5*nbins entries of feat
    outer_avg = np.nanmean([feat["radial_{}".format(i)] for i in range(int(0.25 * nbins))])
    feat["boundary_avg"] = outer_avg

    # Calculate the average over the last 0.5*nbins entries of feat
    inner_avg = np.nanmean([feat["radial_{}".format(i)] for i in range(int(0.25 * nbins), nbins)])
    feat["bulk_avg"] = inner_avg

    del mask, bins_dr


    return feat