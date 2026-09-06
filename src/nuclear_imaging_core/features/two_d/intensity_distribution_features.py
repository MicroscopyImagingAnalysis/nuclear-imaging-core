import numpy as np
import itertools

from scipy.stats.mstats import kurtosis, skew
from scipy import stats
from skimage import measure, morphology

def basicIntensity(full_raw: np.ndarray,binary_mask: np.ndarray, hc_thresh: float = 1.0):

    ### calculate basic features of the intensity distribution

    print("intensity shapes",full_raw.shape,binary_mask.shape)
    try:
        subreg = full_raw[binary_mask > 0]
    except:
        print("binary mask",binary_mask.shape)
        print("full raw",full_raw.shape)
        print("binary mask",binary_mask)
        print("full raw",full_raw)
        exit(0)
    # subreg = full_raw[binary_mask>0]



    high, low = np.nanpercentile(subreg, q=(80, 20))
    if low ==0:
        low = 1
    hc = np.mean(subreg) + (hc_thresh * np.std(subreg))
##    hc = np.nanpercentile(subreg,q=67)
    if np.sum(subreg >= hc) == 0:
        print("WHOA WHOA WHOA",np.min(subreg),np.mean(subreg),np.std(subreg),hc,np.max(subreg))
##        exit(0)
    feat = {
        "i80_i20": high / low,
        "nhigh_nlow": np.sum(subreg >= high)/ np.sum(subreg <= low),
        "area_ratio_hilo": np.sum(subreg >= hc) / np.sum(subreg < hc), #### subreg >= hc returns an array of size subreg which is true false -- sum gives you how many trues
        "area_ratio_hitotal": np.sum(subreg >= hc) / np.sum(subreg > 0),
        "content_ratio_hilo": np.sum(np.where(subreg >= hc, subreg, 0)) #### find all indices where subreg >= hc, get out values, reset the rest to zero and sum over the intensity values themselves
            / np.sum(np.where(subreg < hc, subreg, 0)),
        "content_ratio_hitotal": np.sum(np.where(subreg >= hc, subreg, 0))
            / np.sum(np.where(subreg > 0, subreg, 0)),
        "int_min": np.nanpercentile(subreg, 0),
        "int_d25": np.nanpercentile(subreg, 25),
        "int_median": np.nanpercentile(subreg, 50),
        "int_d75": np.nanpercentile(subreg, 75),
        "int_max": np.nanpercentile(subreg, 100),
        "int_mean": np.mean(subreg),
        "int_mode": stats.mode(subreg[subreg>0], axis=None).mode,
        "int_sd": np.std(subreg),
        "kurtosis": float(kurtosis(subreg.ravel())),
        "skewness": float(skew(subreg.ravel())),
        "entropy": stats.entropy(subreg.ravel()),


    }

    del subreg

    return feat





def spatialIntensity(full_raw: np.ndarray,binary_mask: np.ndarray,
                      peak_labels: np.ndarray, peak_points:np.ndarray,
                      nbins: int=10, pixel_size:float = 1 ):




    if full_raw[binary_mask > 0].size == 0 or peak_points.shape[0]==0:
        feat = {"peak_count":0,
            "peak_mean_size":np.nan,
            "peak_std_size":np.nan,
            "peak_tot_vol":np.nan,
            "peak_COM_mismatch":np.nan, ## from centres of foci
            "peak_NND":np.nan, ## distance between peaks -- find min for each column and then take average
            "peak_boundary_dist":np.nan,
            "peak_centroid_dist":np.nan,
            }

        #### bin distance from centroid and get intensity as a function of radius
        print("doing radial intensity -- spatial -- no peaks")

        nuc_foo = (measure.regionprops_table(binary_mask.astype(int),
                                             properties=["centroid"]))
##    print(foo)

        # obtain the edge pixels
        nuc_bw = binary_mask > 0
        # nuc_cenz, nuc_ceny, nuc_cenx = (float(nuc_foo['centroid-0'][0]),
        #                                 float(nuc_foo['centroid-1'][0]),
        #                                 float(nuc_foo['centroid-2'][0]))

        nuc_ceny, nuc_cenx = (float(nuc_foo['centroid-0'][0]),
                             float(nuc_foo['centroid-1'][0]))

        return feat



    valid_size = peak_points.shape[0]
    uniq_labels = np.unique(peak_labels[peak_labels>0])[:valid_size]
    print("doing spatial intensity with -- labels",len(uniq_labels),valid_size)
    foci_sizes = []

    for l in uniq_labels:
        foci_sizes.append(peak_labels[peak_labels==l].size)



    peak_interdistances = np.zeros([len(uniq_labels),len(uniq_labels)],np.float64)
    print("matrix shape",peak_interdistances.shape)
    peak_boundarydistances = np.zeros([len(uniq_labels)],np.float64)
    peak_centroiddistances = np.zeros([len(uniq_labels)],np.float64)


    print("generating point pairs",peak_interdistances.shape)
    peak_pairs = itertools.combinations(uniq_labels,2)
    peak_point_pairs = itertools.combinations(peak_points,2)
    ppplist = list(peak_point_pairs)
    pplist = list(peak_pairs)

    print("calculating distances",len(ppplist))
    distances = np.array([np.linalg.norm(point_pair[0]-point_pair[1]) for point_pair in ppplist])

    print("done distances")

    peak_min_interdistances = np.zeros([len(uniq_labels)],np.float64)

    for i in range(len(uniq_labels)):
        l = uniq_labels[i]
        pair_subset = np.where(np.array(pplist)==l)
        all_subset = pair_subset[0] ### rows where l appears
        try:
            peak_min_interdistances[i] = np.min(distances[all_subset])
        except:
            peak_min_interdistances[i] = -1


    print("done min interdistances",peak_min_interdistances.shape)


    nuc_foo = (measure.regionprops_table(binary_mask.astype(int),properties=["centroid"]))

    nuc_bw = binary_mask > 0
    # nuc_cenz, nuc_ceny, nuc_cenx = (float(nuc_foo['centroid-0'][0]),float(nuc_foo['centroid-1'][0]),float(nuc_foo['centroid-2'][0]))

    nuc_ceny, nuc_cenx = (float(nuc_foo['centroid-0'][0]),float(nuc_foo['centroid-1'][0]))


    edge = np.subtract(nuc_bw * 1, morphology.erosion(nuc_bw) * 1)
    # (boundary_z, boundary_y,boundary_x) = [np.where(edge > 0)[0], np.where(edge > 0)[1],np.where(edge>0)[2]]

    (boundary_y,boundary_x) = [np.where(edge > 0)[0], np.where(edge > 0)[1]]

    boundary_points = np.zeros([boundary_x.shape[0],2],np.float64)
    for i in range(boundary_x.shape[0]):
        boundary_points[i,0] = pixel_size*(boundary_x[i]+0.5)
        boundary_points[i,1] = pixel_size*(boundary_y[i]+0.5)


    for i in range(len(uniq_labels)):
        dist_f_b = np.sqrt(
            np.square(boundary_points[:,0]-peak_points[i,0]) +
            np.square(boundary_points[:,1]-peak_points[i,1])
        )

        peak_boundarydistances[i] = np.min(dist_f_b)

        # peak_centroiddistances[i] = np.sqrt(np.square(nuc_cenx*pixel_size - peak_points[i,0]) +
        #                                     np.square(nuc_ceny*pixel_size - peak_points[i,1]) +
        #                                     np.square(nuc_cenz*z_step_size - peak_points[i,2]))

        peak_centroiddistances[i] = np.sqrt(np.square(nuc_cenx*pixel_size - peak_points[i,0]) +
                                            np.square(nuc_ceny*pixel_size - peak_points[i,1]))





    print(peak_points.shape,np.mean(peak_points,axis=0).shape)
    if peak_points.shape[0] > 0:
        # peak_com_mismatch = np.linalg.norm(np.mean(peak_points,axis=0)-np.array([nuc_cenx*pixel_size,nuc_ceny*pixel_size,nuc_cenz*z_step_size]))
        peak_com_mismatch = np.linalg.norm(np.mean(peak_points,axis=0)-np.array([nuc_cenx*pixel_size,nuc_ceny*pixel_size]))
    else:
        peak_com_mismatch = np.nan


    print("masked feature dict")
    try:
        spam_nnd = np.mean(peak_min_interdistances)
    except:
        spam_nnd = np.nan

    feat = {"peak_count": len(uniq_labels),
            "peak_mean_size": np.mean(np.array(foci_sizes)),
            "peak_std_size": np.std(np.array(foci_sizes)),
            "peak_tot_vol": peak_labels[peak_labels>0].size,
            "peak_COM_mismatch":peak_com_mismatch, ## from centres of foci
            "peak_NND":spam_nnd, ## distance between peaks -- find min for each column and then take average
            "peak_boundary_dist":np.mean(peak_boundarydistances),
            "peak_centroid_dist":np.mean(peak_centroiddistances),
            }

##    for i in range(nbins):
##        feat["radial_DAPI_{}".format(i)] = radial_intensity[i]
    del peak_interdistances, peak_boundarydistances, peak_centroiddistances, peak_min_interdistances, boundary_points, distances, ppplist, pplist
    print("done spatial intensity")
    return feat



def intensityFeatures(raw_image, region_mask, peak_labels, peak_points,
                      nbins=10, pixel_size=0.385,z_step_size=1.0):
    """
    Calculate intensity features for a given image and region mask
    """
    feat_dict = {}

    feat_dict.update(basicIntensity(raw_image, region_mask))
    feat_dict.update(spatialIntensity(raw_image, region_mask,
                                      peak_labels,peak_points,
                                      nbins=nbins,
                                      pixel_size=pixel_size))


    return feat_dict
