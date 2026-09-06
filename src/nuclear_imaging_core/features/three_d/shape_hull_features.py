from skimage.morphology import erosion
from scipy import stats
from scipy.spatial import distance, distance_matrix
import numpy as np
from skimage.transform import rotate
import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats.mstats import kurtosis, skew
from skimage import morphology, feature, measure
from skimage.feature import peak_local_max
from skimage.measure import (
    regionprops_table,
    marching_cubes,
    mesh_surface_area,
    regionprops,
)
from skimage.segmentation import watershed
from scipy.spatial import ConvexHull, convex_hull_plot_2d

#### reconsider 3D morphological features until appropriate size transform applied on pixel xyz geometry
#### maybe make a separate ConvexHull based routine for these geometrical features
def sk3dmorpho(full_labelled: np.ndarray):
    morphological_properties = [
         "convex_image",
         "equivalent_diameter",
         "extent",
         "major_axis_length",
         "minor_axis_length",
         "solidity"]
##    morphological_properties = ["major_axis_length","minor_axis_length","centroid"]

    morphological_features = regionprops_table(
            np.uint8(full_labelled),
            properties=morphological_properties,
            separator="_",
        )

    morphological_features["nuclear_volume"] = np.sum(full_labelled)
    # morphological_features["convex_hull_vol"] = np.sum(
    #     morphological_features["convex_image"]
    # )
    # morphological_features["concavity_3d"] = (
    #     morphological_features["convex_hull_vol"]
    #     - morphological_features["nuclear_volume"]
    # ) / morphological_features["convex_hull_vol"]
    #
    # del morphological_features["convex_hull_vol"]
    # del morphological_features["concavity_3d"]
    del morphological_features["convex_image"]
    # print(morphological_features)
##    morphological_features.update({'label':np.array([lblist])})
    print(morphological_features)

    return morphological_features



def hullFeatures2D(binary_mask: np.ndarray):
    ### subroutine to construct the 2d hull of a cell from the 3d points projected on 2d
    pixel_size = 0.35 # um/px
    z_step_size = 1.5 # um/px

    foo = (measure.regionprops_table(binary_mask.astype(int),properties=["centroid"]))
##    print(foo)

    # obtain the edge pixels
    bw = binary_mask > 0
    cenz, ceny, cenx = (float(foo['centroid-0'][0]),float(foo['centroid-1'][0]),float(foo['centroid-2'][0]))
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_z, boundary_y,boundary_x) = [np.where(edge > 0)[0], np.where(edge > 0)[1],np.where(edge>0)[2]]

    surfpoints = np.zeros([boundary_x.shape[0],2],np.float64)

    ### populate boundary and bulk

    for i in range(boundary_x.shape[0]):
        surfpoints[i,0] = pixel_size*(boundary_y[i]+0.5)
        surfpoints[i,1] = pixel_size*(boundary_x[i]+0.5)
##        surfpoints[i,2] = 0.0 #z_step_size*(boundary_z[i]+0.5)

    ### now compute 2D hull
    bulk_y,bulk_x = [np.where(binary_mask>0)[1],np.where(binary_mask>0)[2]]
    bulkpoints = np.zeros([bulk_x.shape[0],2],np.float64)

    for i in range(bulk_x.shape[0]):
        bulkpoints[i,0] = pixel_size*(bulk_y[i]+0.5)
        bulkpoints[i,1] = pixel_size*(bulk_x[i]+0.5)
    try:
        surfhull = ConvexHull(surfpoints)
        feat = {"2dhull_calc": 1,
                "proj_area": surfhull.volume,
                "proj_perim": surfhull.area,
                "proj_concavity": (surfhull.volume - np.unique(bulkpoints).shape[0]*pixel_size*pixel_size) / surfhull.volume,
                "proj_solidity": np.unique(bulkpoints).shape[0]*pixel_size*pixel_size/surfhull.volume,
                "proj_shape_factor": surfhull.area**2 / ( 4*np.pi *surfhull.volume),
                }
    except:
        spamarea = np.unique(bulkpoints).shape[0]*pixel_size*pixel_size
        spamperim = np.unique(surfpoints).shape[0]*pixel_size
        feat = {"2dhull_calc": 0,
                "proj_area":spamarea,
                "proj_perim":spamperim,
                "proj_concavity":0,
                "proj_solidity":1,
                "proj_shape_factor":spamarea**2/(4*np.pi*spamperim)
                }

    del surfpoints, bulkpoints, surfhull, edge, boundary_x, boundary_y, boundary_z, bulk_x, bulk_y
    return feat




def hullFeatures(binary_mask: np.ndarray,
                 pixel_size: np.float64 = 0.35,z_step_size: np.float64 = 1.5):

    #### subroutine to convert to points, get hull and compute geometric properties thereof
    #### along the way compare the surface hull to the full hull as a debug step


    foo = (measure.regionprops_table(binary_mask.astype(int),properties=["centroid"]))

    # obtain the edge pixels
    bw = binary_mask > 0
    cenz, ceny, cenx = (float(foo['centroid-0'][0]),float(foo['centroid-1'][0]),float(foo['centroid-2'][0]))
    print("hull cen",cenx,ceny,cenz)



    (bulk_z,bulk_y,bulk_x) = [np.where(binary_mask>0)[0],np.where(binary_mask>0)[1],np.where(binary_mask>0)[2]]
    if len(np.unique(bulk_z)) <=1:
        feat = {"hull_calc": -1,
            "hull_vol": np.nan,
            "hull_surf_area":np.nan,
            "concavity": np.nan,
            "hull_vertex_count": np.nan,
            "major_axis":np.nan,
            "mid_axis":np.nan,
            "minor_axis":np.nan,
            "shape_anisotropy":np.nan,
            "min_max_axis_ratio":np.nan,
            "orientation":np.nan,
            }


        return feat


    bulkpoints = np.zeros([bulk_x.shape[0],3],np.float64)
    hull_calc = 1
    for i in range(bulk_x.shape[0]):
        bulkpoints[i,0] = bulk_x[i] # pixel_size*(bulk_x[i]+0.5)
        bulkpoints[i,1] = bulk_y[i] # pixel_size*(bulk_y[i]+0.5)
        bulkpoints[i,2] = bulk_z[i] #z_step_size*(bulk_z[i]+0.5)

    try:
        bulkhull = ConvexHull(bulkpoints)
        surface_area = bulkhull.area
        volume = bulkhull.volume
        verts = bulkhull.vertices
        hull_calc = 1

        print("bulkhull size",bulk_x.shape,verts.shape,volume,surface_area)
    except:
        print("couldnt hull the bulk",bulkpoints.shape)
        verts, faces, _, _ = marching_cubes(bulkpoints, 0.0)
        surface_area = mesh_surface_area(verts, faces)
        volume = bulk_x.shape[0]*pixel_size*pixel_size*z_step_size
        hull_calc = 0


    print(bulkpoints.shape,verts.shape,hull_calc)
    try:
        surfxc = bulkpoints[verts]*np.array([pixel_size,pixel_size,z_step_size])
    except:
        print("WHOA WHOA WHOA",bulkpoints.shape,verts.shape,hull_calc)
    print(surfxc.shape,bulkpoints.shape)
    cent = np.array([pixel_size*cenx,pixel_size*ceny,z_step_size*cenz]) #np.mean(surfxc,axis=0)
    denom = surfxc.shape[0]
    gyr_tensor = np.zeros([3,3],np.float64)
    for i in range(3):
##            print(i,chrid,cent[i])
        for j in range(3):
##                print(i,j,chrid,cent[j])
            for pt in surfxc:
                gyr_tensor[i,j] += (1.0/denom)*(pt[i] - cent[i])*(pt[j]-cent[j]) ## treated as assembly
    eigval,eigvec = np.linalg.eigh(gyr_tensor)
    D = np.diag(eigval)
    P_inv = np.linalg.inv(eigvec)
    gyr_tensor_diag = np.dot(P_inv,np.dot(gyr_tensor,eigvec))


    lambsq = eigval
    Rg = np.sqrt(np.sum(lambsq)/denom)

    kappasq = 1.5*(lambsq[0]**2 + lambsq[1]**2 + lambsq[2]**2)/(np.sum(lambsq)**2) - 0.5
    asphere = 1.5*lambsq[2] - 0.5*Rg**2
    orient = (180.0/np.pi)*abs(np.arccos(np.dot(eigvec[2],np.array([1,0,0]))/np.linalg.norm(eigvec[2])))

    feat = {"hull_calc": hull_calc,
            "hull_vol": volume,
            "hull_surf_area": surface_area,
            "concavity": (volume - bulk_x.shape[0]*pixel_size*pixel_size*z_step_size) / volume,
            "hull_vertex_count": verts.shape[0],
            "major_axis":np.sqrt(lambsq[2]),
            "mid_axis":np.sqrt(lambsq[1]),
            "minor_axis":np.sqrt(lambsq[0]),
            "shape_anisotropy":kappasq,
            "asphericity":asphere,
            "min_max_axis_ratio":np.sqrt(lambsq[0])/np.sqrt(lambsq[2]),
            "orientation":orient,
            }

    del bulkpoints, surfxc, bulkhull, verts, gyr_tensor, eigval, eigvec, D, P_inv, gyr_tensor_diag
    return feat

def basicBoundary(binary_mask: np.ndarray):


    pixel_size = 0.35 # um/px
    z_step_size = 1.5 # um/px

    foo = (measure.regionprops_table(binary_mask.astype(int),properties=["centroid"]))
    # print(foo)

    # obtain the edge pixels
    bw = binary_mask > 0
    cenz, ceny, cenx = (float(foo['centroid-0'][0]),float(foo['centroid-1'][0]),float(foo['centroid-2'][0]))
    # print(cenx,ceny,cenz)
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_z, boundary_y,boundary_x) = [np.where(edge > 0)[0], np.where(edge > 0)[1],np.where(edge>0)[2]]
    # print(min(boundary_z),max(boundary_z),cenz,min(boundary_y),max(boundary_y),ceny,min(boundary_x),max(boundary_x),cenx)
    # calculate radii
    dist_b_c = np.sqrt(np.square((boundary_x - cenx)*pixel_size) +
                       np.square((boundary_y - ceny)*pixel_size) +
                       np.square((boundary_z - cenz)*z_step_size))



    feat ={ "min_radius": np.min(dist_b_c),
            "max_radius": np.max(dist_b_c),
            "med_radius": np.median(dist_b_c),
            "avg_radius": np.mean(dist_b_c),
            "mode_radius": stats.mode(dist_b_c.astype(int), axis=None).mode,
            "d25_radius": np.percentile(dist_b_c, 25),
            "d75_radius": np.percentile(dist_b_c, 75),
            "std_radius": np.std(dist_b_c),
          }

    return feat




def measure_calliper(binary_mask: np.ndarray,angular_resolution:int = 10):

    def max_z_projection(binary_mask):
        return np.max(binary_mask, axis=0)

    def measure_2d_calliper(binary_mask_2d, angles):
        max_dist = 0
        min_dist = float('inf')
        coords = np.column_stack(np.where(binary_mask_2d > 0))
        centroid = np.mean(coords, axis=0)

        for angle in range(0, 180, angles):
            theta = np.deg2rad(angle)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_coords = np.dot(coords - centroid, rotation_matrix) + centroid
            min_x, min_y = np.min(rotated_coords, axis=0)
            max_x, max_y = np.max(rotated_coords, axis=0)
            dist = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
            max_dist = max(max_dist, dist)
            min_dist = min(min_dist, dist)

        return max_dist, min_dist

    binary_mask_2d = max_z_projection(binary_mask)
    max_calliper, min_calliper = measure_2d_calliper(binary_mask_2d, angular_resolution)
    feat = {
        "2d_max_calliper": max_calliper,
        "2d_min_calliper": min_calliper
    }

    del binary_mask_2d, max_calliper, min_calliper

    return feat







def shapeFeatures(binary_image:np.ndarray, angular_resolution:int = 10,
                  calc_calliper:bool =True):
    """Compute all boundary features
        This function computes all features that describe the boundary features
        Args:
            binary_image:(image_array) Binary image
            angular_resolution:(integer) value between 1-359 to determine the number of rotations
        Returns: A pandas dataframe with all the features for the given image
        """
    feat ={}


    feat.update(basicBoundary(binary_image))

    feat.update(hullFeatures(binary_image))

    feat.update(hullFeatures2D(binary_image))

    # feat.update(sk3dmorpho(binary_image))

    if(calc_calliper):
        feat.update(measure_calliper(binary_image, angular_resolution))

    return feat