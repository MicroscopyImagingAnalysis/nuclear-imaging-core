"""Three-dimensional gray-level co-occurrence matrix measurements."""

# Import modules
import numpy as np
import pandas as pd
from skimage.feature import graycoprops
from skimage import measure



import warnings
warnings.filterwarnings("ignore")

def graycomatrix_3d(img:np.ndarray, distances:float, angles_1, angles_2, resolution:float = [1.0, 0.31 ,0.31]):

    def calc_offset(angle_pair, distance:float):
        z:float = np.multiply(distance, np.sin(angle_pair[1]), dtype=float)*resolution[0]
        x2:float = np.multiply(np.multiply(distance, np.cos(angle_pair[1]),dtype=float), np.sin(angle_pair[0]),dtype=float)*resolution[1]
        y2:float = np.multiply(np.multiply(distance, np.cos(angle_pair[1]),dtype=float), np.cos(angle_pair[0]),dtype=float)*resolution[2]
        return(np.array([int(z),int(y2),int(x2)], dtype=int))

    img:int = (np.rint(img)).astype(np.uint8)
    levels:int = img.max() + 1

    grid1, grid2 = np.meshgrid(angles_1, angles_2)
    angles:int = np.vstack([grid1.ravel(), grid2.ravel()]).T

    # Get the shape of the image
    z_max:int  = img.shape[0]
    x_max:int = img.shape[2]
    y_max:int = img.shape[1]
    print(z_max,y_max,x_max,levels)

    glcm:int = np.zeros((levels, levels, len(distances), len(angles)), dtype=int)
    print("glcm shape",glcm.shape)
    for i,distance in enumerate(distances):
        print("doing grayco for dist",i,len(distances))
        for j,angle_pair in enumerate(angles):
##            print("doing grayco for angle",i,j,len(angles))
            offset = calc_offset(angle_pair, distance)
            for z in np.arange(z_max, dtype=int):
                for y in np.arange(y_max, dtype=int):
                    for x in np.arange(x_max, dtype=int):
                        curr = np.array([z,y,x], dtype=int)
                        neighbor = np.add(curr,offset, dtype=int)
                        if neighbor[1] >= 0 and neighbor[1] < y_max and neighbor[2] >= 0 and neighbor[2] < x_max and neighbor[0] >= 0 and neighbor[0] < z_max:
                            glcm[img[z,y,x], img[neighbor[0],neighbor[1],neighbor[2]], i, j] += 1
                        else:
                            continue

    return(glcm)



def gclm_textures(regionmask: np.ndarray, intensity: np.ndarray, lengths=[2, 5, 20],
                  angles=np.arange(4), resolution:float = [1.0,0.31,0.31]):
    """ Compute GLCM features at given lengths

    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales
     """
    # Contruct GCL matrix at given pixels lengths

    #### make a subset of our array
##    print(regionmask.shape)
    ranges = np.where(regionmask > 0)
    zmin = max(0,np.min(ranges[0])-1)
    zmax = min(np.max(ranges[0])+2,regionmask.shape[0])

    ymin = max(0,np.min(ranges[1])-1)
    ymax = min(np.max(ranges[1])+2,regionmask.shape[1])

    xmin = max(0,np.min(ranges[2])-1)
    xmax = min(np.max(ranges[2])+2,regionmask.shape[2])

    regionmask_subset = regionmask[zmin:zmax,ymin:ymax,xmin:xmax]
    intensity_subset = intensity[zmin:zmax,ymin:ymax,xmin:xmax]

    glcm = graycomatrix_3d(intensity_subset * regionmask_subset,
        distances=lengths,
        angles_1 = angles, angles_2 = angles
    )
    # print("done gray co",l)

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

    del glcm,regionmask_subset,intensity_subset

    return feat

def image_moments(regionmask: np.ndarray, intensity: np.ndarray):
    """ Compute image moments
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """

    moments_features = ['weighted_centroid','weighted_moments','weighted_moments_normalized',
                        'weighted_moments_central','weighted_moments_hu',
                        'moments','moments_normalized','moments_central','moments_hu']
    regionmask=regionmask.astype(np.uint16)

    # feat = pd.DataFrame(measure.regionprops_table(regionmask,intensity,
    #                            properties=moments_features))

    feat = measure.regionprops_table(regionmask,intensity,
                               properties=moments_features)
    feat_first_elements = {key: value[0] for key, value in feat.items()}
    # print(feat_first_elements)

    return feat_first_elements


def center_mismatch(regionmask: np.ndarray, intensity: np.ndarray, resolution):
    """ Compute distance between centroid and center of mass

    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    regionmask=regionmask.astype(np.uint16)
    measures = measure.regionprops_table(regionmask,intensity,
                         properties=['centroid','weighted_centroid'])
    dist = np.sqrt(
        np.multiply(np.square(measures['centroid-0']-measures['weighted_centroid-0']), resolution[0])
        + np.multiply(np.square(measures['centroid-1']-measures['weighted_centroid-1']), resolution[1])
        + np.multiply(np.square(measures['centroid-2']-measures['weighted_centroid-2']), resolution[2])
        [0])

    feat = {"center_mismatch": dist[0]}
    return feat


def textureFeatures(intensity: np.ndarray,regionmask: np.ndarray,
                    lengths=[4, 8, 16, 20], resolution:float = [1.0, 0.31 ,0.31]):
    """Compute all texture features
    This function computes all features that describe the image texture
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales
    Returns: A pandas dataframe with all the features for the given nucleus -- masked with
    """
    # compute features

    all_features = {}

    # all_features.update(image_moments(regionmask, intensity))
    all_features.update(gclm_textures(regionmask, intensity,
                                      lengths= lengths,angles=np.arange(4),
                                       resolution = resolution))

    all_features.update(center_mismatch(regionmask, intensity, resolution = resolution))
##    print(all_features)
    return all_features
