# -*- coding: utf-8 -*-
"""
Library for computing features that describe the texture of a given image
This module provides functions that one can use to obtain and describe the texture of a given image
Available Functions:
-gclm_textures:Compute GLCM features at different length sclates
-Peripherial_Distribution_Index: Todo
"""

# Import modules
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from skimage import measure


def gclm_textures(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20],angles= [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    """ Compute GLCM features at given lengths

    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales
     """
    # Contruct GCL matrix at given pixels lengths
    # print(regionmask,intensity)

    print("bad vals glcm",np.where(intensity < -1),np.where(intensity > 1))
    glcm = graycomatrix(
        img_as_ubyte((intensity * regionmask)),
        distances=lengths,
        angles=angles,
    )

    feat = {}

    contrast_vals = np.mean(graycoprops(glcm, "contrast"), axis=1).tolist()
    contrast_keys = ["contrast_" + str(col) for col in lengths]
    contrast_dict = dict(zip(contrast_keys,contrast_vals))

    feat.update(contrast_dict)
##    print(contrast)
    # contrast = pd.DataFrame(contrast_dict,index=[l])
##    print(contrast_df)

    dissimilarity_vals = np.mean(graycoprops(glcm, "dissimilarity"), axis=1).tolist()
    dissimilarity_keys = ["dissimilarity_" + str(col) for col in lengths]
    dissimilarity_dict = dict(zip(dissimilarity_keys,dissimilarity_vals))
    # dissimilarity = pd.DataFrame(dissimilarity_dict,index=[l])
    feat.update(dissimilarity_dict)

    homogeneity_vals = np.mean(graycoprops(glcm, "homogeneity"), axis=1).tolist()
    homogeneity_keys = ["homogeneity_" + str(col) for col in lengths]
    homogeneity_dict = dict(zip(homogeneity_keys,homogeneity_vals))
    # homogeneity = pd.DataFrame(homogeneity_dict,index=[l])
    feat.update(homogeneity_dict)

    ASM_vals = np.mean(graycoprops(glcm, "ASM"), axis=1).tolist()
    ASM_keys = ["ASM_" + str(col) for col in lengths]
    ASM_dict = dict(zip(ASM_keys,ASM_vals))
    # ASM = pd.DataFrame(ASM_dict,index=[l])
    feat.update(ASM_dict)

    energy_vals = np.mean(graycoprops(glcm, "energy"), axis=1).tolist()
    energy_keys = ["energy_" + str(col) for col in lengths]
    energy_dict = dict(zip(energy_keys,energy_vals))
    # energy = pd.DataFrame(energy_dict,index=[l])
    feat.update(energy_dict)

    correlation_vals = np.mean(graycoprops(glcm, "correlation"), axis=1).tolist()
    correlation_keys = ["correlation_" + str(col) for col in lengths]
    correlation_dict = dict(zip(correlation_keys,correlation_vals))
    # correlation = pd.DataFrame(correlation_dict,index=[l])
    feat.update(correlation_dict)

    del glcm

    return feat


def peripherial_distribution_index(regionmask: np.ndarray, intensity: np.ndarray):
    """Computes peripherial distribution index of a grayscale image
    Ref PMID: 3116470
    """
    pass


def image_moments(regionmask: np.ndarray, intensity: np.ndarray):
    """ Compute image moments
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """

    moments_features = ['weighted_centroid','weighted_moments','weighted_moments_normalized',
                        'weighted_moments_central','weighted_moments_hu',
                        'moments','moments_normalized','moments_central','moments_hu']
    regionmask=regionmask.astype('uint8')

    # feat = pd.DataFrame(measure.regionprops_table(regionmask,intensity,
    #                            properties=moments_features))
    test = regionmask > 0
    print(test.shape, intensity.shape)
    feat = measure.regionprops_table(regionmask,intensity,
                               properties=moments_features)

    # print(feat)
    feat_first_elements = {key: value[0] for key, value in feat.items()}
    # print(feat_first_elements)

    return feat_first_elements

def center_mismatch(regionmask: np.ndarray, intensity: np.ndarray):
    """ Compute distance between centroid and center of mass

    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    regionmask=regionmask.astype('uint8')
    measures = measure.regionprops_table(regionmask,intensity,
                         properties=['centroid','weighted_centroid'])

    print(measures,regionmask.size)
    try:
        dist = np.sqrt(np.square(measures['centroid-0']-measures['weighted_centroid-0'])+
                    np.square(measures['centroid-1']-measures['weighted_centroid-1']))[0]
    except:
        dist = 0

    # feat = {"center_mismatch": np.percentile(intensity[regionmask], 0)}
    feat = {"center_mismatch": dist}
    return feat





def textureFeatures(intensity: np.ndarray, regionmask: np.ndarray, lengths=[1, 5, 20],
                             measure_gclm: bool = True, measure_moments: bool = True):
    """Compute all texture features
    This function computes all features that describe the image texture
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales
    Returns: A pandas dataframe with all the features for the given image
    """
    # compute features
    # all_features = pd.DataFrame()

    # if(measure_gclm):
    #     all_features = pd.concat([all_features, gclm_textures(regionmask, intensity, lengths).reset_index(drop=True)], axis = 1)
    # if(measure_moments):
    #     all_features = pd.concat([all_features, image_moments(regionmask, intensity).reset_index(drop=True)], axis = 1)
    # return all_features

    all_features = {}
    scaled_intensity = intensity * regionmask
    scaled_intensity = (scaled_intensity - np.min(scaled_intensity)) / (np.max(scaled_intensity) - np.min(scaled_intensity) + 1e-10)
    # all_features.update(image_moments(regionmask, scaled_intensity))
    all_features.update(gclm_textures(regionmask, scaled_intensity,
                                      lengths= lengths,angles=np.arange(4)))
    print("gclm done")
    all_features.update(center_mismatch(regionmask, scaled_intensity))
##    print(all_features)
    return all_features