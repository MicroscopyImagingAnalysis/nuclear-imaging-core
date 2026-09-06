import numpy as np
import pandas as pd
from .utils import gen_mask_fast,gen_mask_erosion,gen_mask_expand
from skimage import morphology
import scipy.ndimage as ndi

def get_nuc_posn_in_spheroid(inp_nucdata_df,sphdata_df,nuc_labels,spheroid_labels):

    nucdata_df = inp_nucdata_df.copy()  # Ensure no call-by-reference issues

    # unique_sphs = np.unique(spheroid_labels[spheroid_labels > 0])
    # unique_nucs = np.unique(nuc_labels[nuc_labels > 0])
    unique_sphs = np.unique(sphdata_df['label-id'].values)
    unique_nucs = np.unique(nucdata_df['label-id'].values)

    for si in range(len(unique_sphs)):
        sl = unique_sphs[si]

        spheroid_mask = spheroid_labels == sl

        nuc_in_spheroid = nucdata_df[(nucdata_df['of-spheroid'] == sl) &
                                      (nucdata_df['in-spheroid'] == 1)]
        for index, row in nuc_in_spheroid.iterrows():
            cenz, ceny, cenx = int(row['cenz']), int(row['ceny']), int(row['cenx'])

            # Find distance from nucleus to spheroid mask boundary in the z-plane
            z_plane_mask = spheroid_mask[cenz, :, :]
            # Find the edge of the z-plane mask
            z_plane_edges = morphology.binary_erosion(z_plane_mask) ^ z_plane_mask
            distance_to_boundary = ndi.morphology.distance_transform_edt(~z_plane_edges)[ceny, cenx]

            # Find distance of cenz in the range (minz, maxz)
            minz, maxz = np.min(np.argwhere(spheroid_mask)[:, 0]), np.max(np.argwhere(spheroid_mask)[:, 0])
            distance_in_z_range = (cenz - minz) / (maxz - minz)

            nucdata_df.at[index, 'dist_to_boundary'] = distance_to_boundary
            nucdata_df.at[index, 'dist_in_z_range'] = distance_in_z_range

    del inp_nucdata_df, spheroid_mask, nuc_in_spheroid

    return nucdata_df

def assign_shell_nucleii(inp_nucdata_df,sphdata_df,nuc_labels,spheroid_labels,
                   nbins:int = 20, maxdist:float = 400):

    nucdata_df = inp_nucdata_df.copy()  # Ensure no call-by-reference issues
    # unique_sphs = np.unique(spheroid_labels[spheroid_labels > 0])
    # unique_nucs = np.unique(nuc_labels[nuc_labels > 0])
    unique_sphs = np.unique(sphdata_df['label-id'].values)
    unique_nucs = np.unique(nucdata_df['label-id'].values)

    ### radial_df is a df with col-0 = bin distance

    spam_bin_arr = np.zeros([nucdata_df.index.values.shape[0]],np.int64)

    for si in range(len(unique_sphs)):
        sl = unique_sphs[si]

        #### generate an array of masks [nbins x imagey x image x] -- starting from the boundary inwards

        ##
        spheroid_mask = spheroid_labels == sl
        # masks,bins_dr = gen_mask_erosion(spheroid_mask,nbins,maxdist)
        # print("masks shape",masks.shape)
        #### calc properties in the masked area

        for i in range(nbins):
            ### filter nucs in our bin of interest

            # print(nucdata_df.index.values)
            # for i in nucdata_df.index.values:
            #     print(i,nucdata_df.iloc[i-1]["cenz"],
            #           nucdata_df.iloc[i-1]["ceny"],
            #           nucdata_df.iloc[i-1]["cenx"])

            coords = np.array([np.array([nucdata_df.iloc[n]["cenz"].astype(int),
                                         nucdata_df.iloc[n]["ceny"].astype(int),
                                         nucdata_df.iloc[n]["cenx"].astype(int)])
                                           for n in range(len(nucdata_df.index.values))])
            mask,bins_dr = gen_mask_erosion(spheroid_mask,i,nbins,maxdist)
            # print("mask shape",mask.shape)
            mask_coords = np.argwhere(mask > 0)

            rows = np.array([np.any((mask_coords == coords[j]).all(axis=1))
                              for j in range(coords.shape[0])])

            spam_bin_arr[np.where(rows==True)[0]] = i

    nucdata_df["shell_bin_ref_sph"] = spam_bin_arr

    del inp_nucdata_df, mask, mask_coords, coords, rows
    return nucdata_df


def assign_outward_nucleii(inp_nucdata_df,sphdata_df,nuc_labels,spheroid_labels,
                   nbins:int = 20, maxdist:float = 400):

    nucdata_df = inp_nucdata_df.copy()  # Ensure no call-by-reference issues
    # unique_sphs = np.unique(spheroid_labels[spheroid_labels > 0])
    # unique_nucs = np.unique(nuc_labels[nuc_labels > 0])
    unique_sphs = np.unique(sphdata_df['label-id'].values)
    unique_nucs = np.unique(nucdata_df['label-id'].values)

    ### radial_df is a df with col-0 = bin distance

    dr = maxdist/nbins

    spam_bin_arr = np.zeros([nucdata_df.index.values.shape[0]],np.int64)
    spam_core_shell_arr = np.zeros([nucdata_df.index.values.shape[0]],np.int64)

    for si in range(len(unique_sphs)):
        sl = unique_sphs[si]

        #### generate an array of masks [nbins x imagey x image x] -- centred at each spheroid centre
        spheroid_mask = spheroid_labels == sl
        # masks,bins_dr = gen_mask_expand(spheroid_mask,nbins,maxdist)

        #### calc properties in the masked area

        for i in range(nbins):
            ### filter nucs in our bin of interest

            coords = np.array([np.array([nucdata_df.iloc[n]["cenz"].astype(int),
                                         nucdata_df.iloc[n]["ceny"].astype(int),
                                         nucdata_df.iloc[n]["cenx"].astype(int)])
                                           for n in range(len(nucdata_df.index.values))])

            mask,bins_dr = gen_mask_expand(spheroid_mask,i,nbins,maxdist)
            mask_coords = np.argwhere(mask > 0)

            rows = np.array([np.any((mask_coords == coords[j]).all(axis=1))
                             for j in range(coords.shape[0])])
            # preset = spam_bin_arr[np.where(rows==True)[0]] > 0
            spam_bin_arr[np.where(rows==True)[0]] = i

    del inp_nucdata_df, mask, mask_coords, coords, rows

    nucdata_df["outward_bin_ref_sph"] = spam_bin_arr

    return nucdata_df



def process_radial(nucdata_df,sphdata_df,nuc_labels,spheroid_labels,
                   analysis_params,row_dict):

    feat_dict = {}

    unique_sphs = np.unique(spheroid_labels[spheroid_labels > 0])
    unique_nucs = np.unique(nuc_labels[nuc_labels > 0])

    maxdist_shell = analysis_params["maxdist-sph-shell"]


    maxdist_outward = analysis_params["maxdist-sph-outward"]
    nbins = analysis_params["nbins-sph"]

    dr_shell = maxdist_shell/nbins
    dr_outward = maxdist_outward/nbins

    bins_dr = np.array([(i+0.5)*dr_shell for i in range(nbins)])

    radial_df = pd.DataFrame({"bin-centre-shell":bins_dr.T})

    bins_dr = np.array([(i+0.5)*dr_outward for i in range(nbins)])

    radial_df["bin-centre-outward"] = bins_dr.T


    nucdata_df = assign_shell_nucleii(nucdata_df,sphdata_df,nuc_labels,spheroid_labels,
                                      nbins,maxdist_shell)

    print("done shell nucleii",nucdata_df.values.shape)

    nucdata_df = assign_outward_nucleii(nucdata_df,sphdata_df,nuc_labels,spheroid_labels,
                                      nbins,maxdist_outward)

    print("done outward nucleii",nucdata_df.values.shape)

    nucdata_df = get_nuc_posn_in_spheroid(nucdata_df,sphdata_df,nuc_labels,spheroid_labels)




    #### get nuclear features in radial bin

    print("processing radial features")

    nuc_col_list = [col for col in nucdata_df.columns if
                     col.startswith('DAPI') or col.startswith('chan')]
    # Drop specified columns from nuc_col_list
    cols_to_drop = ['chan1', 'chan1_type', 'chan2', 'chan2_type', 'chan3', 'chan3_type']
    nuc_col_list = [col for col in nuc_col_list if col not in cols_to_drop]

    # print(nuc_col_list)

    for col in nuc_col_list:
        # print(col)
        col_arr_shell = np.zeros([nbins],np.float64)
        col_arr_outward = np.zeros([nbins],np.float64)
        for i in range(nbins):
        ### get nucs

            if nucdata_df[nucdata_df["shell_bin_ref_sph"]==i][col].values.size == 0:
                col_arr_shell[i] = 0
            else:
                col_arr_shell[i] = np.mean(nucdata_df[nucdata_df["shell_bin_ref_sph"]==i][col].values)

        # for i in range(nbins_outward):
        ### get nucs

            if nucdata_df[nucdata_df["outward_bin_ref_sph"]==i][col].values.size == 0:
                col_arr_outward[i] = 0
            else:
                col_arr_outward[i] = np.mean(nucdata_df[nucdata_df["outward_bin_ref_sph"]==i][col].values)


        radial_df[f"shell-{col}"] = col_arr_shell
        radial_df[f"outward-{col}"] = col_arr_outward


    col_arr_shell = np.zeros([nbins],np.float64)
    col_arr_outward = np.zeros([nbins],np.float64)
    for i in range(nbins):

        if nucdata_df[nucdata_df["shell_bin_ref_sph"]==i][col].values.size == 0:
            col_arr_shell[i] = 0
        else:
            col_arr_shell[i] = nucdata_df[nucdata_df["shell_bin_ref_sph"]==i][col].values.shape[0]

    # for i in range(nbins_outward):

        if nucdata_df[nucdata_df["outward_bin_ref_sph"]==i][col].values.size == 0:
            col_arr_outward[i] = 0
        else:
            col_arr_outward[i] = nucdata_df[nucdata_df["outward_bin_ref_sph"]==i][col].values.shape[0]

    radial_df["shell_nucs_in"] = col_arr_shell
    radial_df["outward_nucs_in"] = col_arr_outward

    #### get spheroid features in radial bin
    # sph_col_list = [col for col in sphdata_df.columns if col.startswith('sph')]

    # print(sph_col_list)

    # # Drop specified columns from sph_col_list
    # cols_to_drop_sph = ['spheroid_count', 'spheroid_area', 'spheroid_label_filename']
    # sph_col_list = [col for col in sph_col_list if col not in cols_to_drop_sph]

    # for col in sph_col_list:
    #     col_arr_shell = np.zeros([nbins],np.float64)
    #     print(col)
    #     for si in range(len(unique_sphs)):
    #         sl = unique_sphs[si]
    #         ##
    #         spheroid_mask = spheroid_labels == sl

    #         for i in range(nbins):
    #             ### get nucs
    #             mask,bins_dr = gen_mask_erosion(spheroid_mask,i,nbins,maxdist)
    #             col_arr_shell[i] += np.mean(spheroid_mask[mask > 0])

    #     col_arr_shell /= len(unique_sphs)

    #     radial_df["shell-"+col] = col_arr_shell

    print(radial_df)

    return radial_df,nucdata_df,sphdata_df