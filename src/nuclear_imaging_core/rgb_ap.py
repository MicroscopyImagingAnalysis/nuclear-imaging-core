import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, morphology, exposure, feature, measure
from skimage.color import rgb2hsv
import scipy.ndimage as ndi
from napari_simpleitk_image_processing import gaussian_blur
from matplotlib.colors import LinearSegmentedColormap




def get_colors(cmap_name, n=256):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n))
#     print(colors)
    spam = []
    spam.extend(colors)
#     print(spam)
    np.random.shuffle(spam)
    spam[0]=[0,0,0,1]

#     print("hey",spam)
    new_cmap = LinearSegmentedColormap.from_list('new_cmap', spam)
    return new_cmap


def get_nuclear_masks_standard(dapi_image, file_prefix,
                      blur_radius: int = 2,
                      mo_classes: int = 3,
                      min_dist: int = 2,
                      min_size: int = 9,
                      plotflag: bool=False):

    dapi_image = (dapi_image - np.min(dapi_image))/(np.max(dapi_image) - np.min(dapi_image))

    blurred = gaussian_blur(dapi_image, variance_x=blur_radius, variance_y=blur_radius)
    res = filters.threshold_multiotsu(blurred,mo_classes)
    binary = blurred > res[-1]
##    print(binary[binary>0].shape)
    #### detect spots/maxima

    coords = feature.peak_local_max(blurred, min_distance=min_dist)
    #     print("len coords",len(coords))
    peak_mask = np.zeros(blurred.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    #### binary and to eliminate noise
    markers = np.logical_and(binary,markers)
    markers = measure.label(markers)


    res = filters.threshold_multiotsu(dapi_image,classes=mo_classes)
##    print(res,dapi_image.shape)
    binary2 = dapi_image > res[-1]
    distance = ndi.distance_transform_edt(binary2)

    # labels = segmentation.watershed(-distance, markers, mask=binary)
    labels = segmentation.watershed(-dapi_image, markers, mask=binary2)
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = measure.label(labels)

    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
##        print(l,xtent,ytent)
        if xtent > 50 or xtent < 10 or ytent > 50 or ytent < 10:
##            print(l,xtent,ytent,labels[labels==l].size)
            labels[labels==l] = 0
        if labels[labels==l].size > 500 or labels[labels==l].size < 16:
##            print(l,labels[labels==l].size)
            labels[labels==l] = 0
    labels = measure.label(labels)

    print("nuclear segmentation std -- markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))
##    print(maxlabel)
    if plotflag==True:
        fig = plt.figure(figsize=(12, 4))
        ax0 = fig.add_subplot(141)
        ax1 = fig.add_subplot(142)
        ax2 = fig.add_subplot(143)
        ax3 = fig.add_subplot(144)
        #show raw image
        ax0.imshow(dapi_image,aspect='auto',cmap='viridis')
        ax0.axis('off')
        ax0.title.set_text('raw image')
        #show segmented image
        ax1.imshow(binary2,aspect='auto',cmap='viridis')
        ax1.axis('off')
        ax1.title.set_text('gaussian blur + threshold')

        maxlabel = len(np.unique(markers))
        blurred_markers = gaussian_blur(maxlabel*markers, variance_x=4, variance_y=4)
        offset = np.zeros_like(blurred_markers)
        offset[blurred_markers>0] = maxlabel
        ax2.imshow(offset+blurred_markers,aspect='auto',cmap='viridis',vmin=0,vmax=2*maxlabel)
        ax2.axis('off')
        ax2.title.set_text('peak mask intersection')

        cbticklabels = list(np.linspace(0,2*maxlabel,100))
    ##    print(cmin,cmax)
        offset = np.zeros_like(labels)
        offset[labels>0] = maxlabel
        im1 = ax3.imshow(offset+labels,aspect='auto',cmap='viridis',vmin=0,vmax=2*maxlabel)
        ax3.axis('off')
        ax3.title.set_text('labelled image')

    spamlabels =10000*labels
    io.imsave(file_prefix+'_nuc_labels'+'.tif',spamlabels.astype('uint8'))

    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_nuc_labels.png')

    return labels,markers


def get_nuclear_masks_dense(dapi_image,file_prefix,
                      blur_radius: int = 5 ,
                      mo_classes: int = 3,
                      min_dist: int = 4,
                      min_size: int = 25,
                      plotflag: bool=False):

    dapi_image = (dapi_image - np.min(dapi_image))/(np.max(dapi_image) - np.min(dapi_image))

    more_blurred = gaussian_blur(dapi_image, variance_x=15, variance_y=15)
    res = filters.threshold_multiotsu(more_blurred,mo_classes+1)
    binary = more_blurred > res[0]
##    print(binary[binary>0].shape)



    test = dapi_image - more_blurred
    test = (test - np.min(test))/(np.max(test) - np.min(test))

    test = exposure.adjust_gamma(test - np.min(test), 0.7)
    # # test = exposure.adjust_log(test, 0.7,inv=True)
    test = exposure.equalize_adapthist(test,kernel_size=16,clip_limit=0.2)
    res = filters.threshold_multiotsu(test,classes=mo_classes)
    binary2 = test > res[-1]
    binary2 = binary2 * binary



    res = filters.threshold_multiotsu(dapi_image,classes=mo_classes)
##    print(res,dapi_image.shape)
    binary3 = dapi_image > res[-1]
    print(dapi_image.shape,blur_radius)

    #### detect spots/maxima
    blurred = gaussian_blur(dapi_image, variance_x=blur_radius, variance_y=blur_radius)
    coords = feature.peak_local_max(blurred * binary2, min_distance=min_dist)
    #     print("len coords",len(coords))
    peak_mask = np.zeros(blurred.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    binary2 = morphology.remove_small_objects(binary2, min_size=16)
    binary2 = morphology.binary_opening(binary2)
    binary2 = morphology.binary_closing(binary2)
    binary2 = morphology.remove_small_holes(binary2, area_threshold=16)

    #### binary and to eliminate noise
    markers = np.logical_and(binary2,markers)
    markers = measure.label(markers)

    # labels = morphology.remove_small_objects(binary2, min_size=16)
    # labels = morphology.binary_opening(labels)
    # labels = morphology.binary_closing(labels)
    # labels = morphology.remove_small_holes(labels, area_threshold=16)
    distance = ndi.distance_transform_edt(binary2)

    # labels = segmentation.watershed(-distance, markers, mask=binary2)
    labels = segmentation.watershed(-dapi_image+more_blurred, markers, mask=binary2)
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = measure.label(labels)
##    print("old",np.unique(labels),markers[markers>0].shape)

    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
##        print(l,xtent,ytent)
        if xtent > 50 or xtent < 10 or ytent > 50 or ytent < 10:
##            print(l,xtent,ytent,labels[labels==l].size)
            labels[labels==l] = 0
        if labels[labels==l].size > 500 or labels[labels==l].size < 16:
##            print(l,labels[labels==l].size)
            labels[labels==l] = 0

    labels = measure.label(labels)
    print("nuc segmentation dense : markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))
##    print(maxlabel)



    if plotflag==True:
        fig = plt.figure(figsize=(12, 3))
        ax0 = fig.add_subplot(141)
        ax1 = fig.add_subplot(142)
        ax2 = fig.add_subplot(143)
        ax3 = fig.add_subplot(144)
        #show raw image
        ax0.imshow(dapi_image,aspect='auto',cmap='viridis')
        ax0.axis('off')
        ax0.title.set_text('raw image')
        #show segmented image
        ax1.imshow(measure.label(binary3),aspect='auto',cmap='viridis')
        ax1.axis('off')
        ax1.title.set_text('gaussian blur + threshold')

        maxlabel = len(np.unique(markers))
        blurred_markers = gaussian_blur(maxlabel*markers, variance_x=4, variance_y=4)
        offset = np.zeros_like(blurred_markers)
        offset[blurred_markers>0] = maxlabel
        # ax2.imshow(offset+blurred_markers,aspect='auto',cmap='viridis',vmin=0,vmax=2*maxlabel)
        ax2.imshow(test,aspect='auto',cmap='viridis') #,vmin=0,vmax=2*maxlabel)
        ax2.axis('off')
        ax2.title.set_text('peak mask intersection')
        # ax3.imshow(labels,aspect='auto',cmap='viridis')
        # ax3.axis('off')
        # ax3.title.set_text('labelled image')

        ### generate a random labellist
        unique_labels = np.arange(len(np.unique(labels[labels>0])))

        np.random.shuffle(unique_labels)
##        print(len(unique_labels),np.arange(len(unique_labels)))
        for l in range(1,len(unique_labels)):
##            print(l,labels[labels==l].size,unique_labels[l])
            labels[labels==l] = unique_labels[l]

        cbticklabels = list(np.linspace(0,2*maxlabel,100))
    ##    print(cmin,cmax)
        offset = np.zeros_like(labels)
        offset[labels>0] = maxlabel
        im1 = ax3.imshow(offset+labels,aspect='auto',cmap='jet',vmin=0,vmax=2*maxlabel)
        # ax3.axis('off')
        ax3.title.set_text('labelled image')
        # cb = fig.colorbar(im1)
        # cb.ax.set_yticklabels([f'{x:.2f}' for x in cbticklabels],size=2)
        # cb.set_label("Contact frequency", labelpad=-1, size=25)





    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_nuc_labels_dense.png')



    spamlabels =10000*labels
    io.imsave(file_prefix+'_nuc_labels_dense'+'.tif',spamlabels.astype('uint8'))





    return labels, markers

def get_spheroid_masks_rgb_no_dapi(rgb_image,file_prefix,
                            min_size=36,
                           ):

    hsv_image = rgb2hsv(rgb_image)


    filt = ((hsv_image[:,:,0] < 0.4 ) | (hsv_image[:,:,0] > 0.5)) & (hsv_image[:,:,1] > np.quantile(hsv_image[:,:,1],0.5))
    print(filt[filt==True].size)

    # new_spam = new_spam[new_spam > 0.7])
    useuse = -1*np.copy(hsv_image[:,:,2])
    usemin = np.min(useuse)
    usemax = np.max(useuse)
    useuse = (useuse - usemin)/(usemax - usemin)

    hsv_filt =  useuse * filt

    # Apply Gaussian blur to smooth the image# Apply Gaussian blur to smooth the image# Apply Gaussian blur to smooth the image
    blurred_hsv_filt = ndi.gaussian_filter(hsv_filt, sigma=2)

    # Normalize the blurred image
    blurred_hsv_filt = (blurred_hsv_filt - np.min(blurred_hsv_filt)) / (np.max(blurred_hsv_filt) - np.min(blurred_hsv_filt))

    # Update hsv_filt with the blurred version
    hsv_filt = blurred_hsv_filt


    # Apply multi-Otsu thresholding
    res = filters.threshold_multiotsu(hsv_filt, classes=3)
    binary = hsv_filt > res[0]

    # Check if the first threshold covers more than 50% of the image
    if np.sum(binary) > 0.25 * hsv_filt.size:
        binary = hsv_filt > res[1]


    # Calculate product of the two masks
    final_mask = binary.copy()
    print("simple mask",np.unique(measure.label(final_mask)))
    labels = measure.label(final_mask)



    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = morphology.binary_closing(labels)
    # labels = morphology.binary_fill_holes(labels)
    labels = morphology.remove_small_holes(labels, area_threshold=512)
    labels = measure.label(labels)
    labels = segmentation.clear_border(labels, buffer_size=2) #, bgval=0, mask=None, *, out=None)[source]

    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
        if xtent > 500 or xtent < 16 or ytent > 500 or ytent < 16:
            labels[labels==l] = 0
        if labels[labels==l].size > 250000 or labels[labels==l].size < 256:
            labels[labels==l] = 0

    # Calculate the centroids of each labeled region
    region_props = measure.regionprops(labels)
    markers = np.zeros_like(labels, dtype=np.int32)
    for i, region in enumerate(region_props):
        centroid = region.centroid
        markers[int(centroid[0]), int(centroid[1])] = i + 1

    print("sph segmentation dense : markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))

    spamlabels =1000*labels
    io.imsave(file_prefix+'_spheroid_labels.tif',spamlabels.astype('uint8'))
    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_labels.png')


    spamlabels =10000*hsv_filt
    print(file_prefix+'_spheroid_chan_intensity.tif')
    io.imsave(file_prefix+'_spheroid_chan_intensity.tif',spamlabels.astype('uint8'))
    plt.clf()

    plt.imshow(hsv_filt,cmap='viridis',vmin=0,vmax=np.max(hsv_filt))
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_chan_intensity.png')

    return labels,markers,hsv_filt

def get_spheroid_masks_rgb_expanded(rgb_image,nuc_labels,file_prefix,
                            blur_radius=30,
                            mo_classes=3,
                            min_dist=400,
                            min_size=36,
                           ):

    hsv_image = rgb2hsv(rgb_image)

    spam = np.zeros_like(hsv_image[:,:,2])
    # spam =
##    hue_spam = np.copy(hsv_image[:,:,0]) ### hue
##    sat_spam = np.copy(hsv_image[:,:,1]) ### saturation
##    val_spam = np.copy(hsv_image[:,:,2]) ### value
    # filt = ((hue_spam < 0.25 ) | (hue_spam > 0.8)) & (sat_spam > np.mean(sat_spam))
    filt = ((hsv_image[:,:,0] < 0.4 ) | (hsv_image[:,:,0] > 0.5)) & (hsv_image[:,:,1] > np.quantile(hsv_image[:,:,1],0.5))
    print(filt[filt==True].size)

    # new_spam = new_spam[new_spam > 0.7])
    useuse = -1*np.copy(hsv_image[:,:,2])
    usemin = np.min(useuse)
    usemax = np.max(useuse)
    useuse = (useuse - usemin)/(usemax - usemin)

    hsv_filt =  useuse * filt

    # Apply multi-Otsu thresholding
    res = filters.threshold_multiotsu(hsv_filt, classes=3)
    binary = hsv_filt > res[0]

    # Check if the first threshold covers more than 50% of the image
    if np.sum(binary) > 0.25 * hsv_filt.size:
        binary = hsv_filt > res[1]

    # Expand nuclear labels
    expanded_nuc_labels = segmentation.expand_labels(nuc_labels, distance=30)
    expanded_nuc_labels = morphology.remove_small_holes(expanded_nuc_labels, area_threshold=128)

    # Generate mask for regions covered by expanded nuclear labels
    expanded_nuc_mask = expanded_nuc_labels > 0

    # Calculate product of the two masks
    final_mask = np.logical_and(binary, expanded_nuc_mask)

    hsv_filt = hsv_filt * final_mask
    print("simple mask",np.unique(measure.label(final_mask)))
    labels = measure.label(final_mask)
    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
        if xtent > 500 or xtent < 16 or ytent > 500 or ytent < 16:
            labels[labels==l] = 0
        if labels[labels==l].size > 250000 or labels[labels==l].size < 256:
            labels[labels==l] = 0


    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = morphology.binary_closing(labels)
    # labels = morphology.binary_fill_holes(labels)
    labels = morphology.dilation(labels, morphology.disk(20))
    labels = morphology.remove_small_holes(labels, area_threshold=128)
    labels = measure.label(labels)
    labels = segmentation.clear_border(labels, buffer_size=2) #, bgval=0, mask=None, *, out=None)[source]

    # Calculate the centroids of each labeled region
    region_props = measure.regionprops(labels)
    markers = np.zeros_like(labels, dtype=np.int32)
    for i, region in enumerate(region_props):
        centroid = region.centroid
        markers[int(centroid[0]), int(centroid[1])] = i + 1

    print("sph segmentation dense : markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))

    spamlabels =1000*labels
    io.imsave(file_prefix+'_spheroid_labels.tif',spamlabels.astype('uint8'))
    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_labels.png')


    spamlabels =10000*hsv_filt
    print(file_prefix+'_spheroid_chan_intensity.tif')
    io.imsave(file_prefix+'_spheroid_chan_intensity.tif',spamlabels.astype('uint8'))
    plt.clf()

    plt.imshow(hsv_filt,cmap='viridis',vmin=0,vmax=np.max(hsv_filt))
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_chan_intensity.png')

    return labels,markers,hsv_filt


def get_spheroid_masks_rgb_model(rgb_image,nuc_labels,file_prefix,model,
                            blur_radius=30,
                            mo_classes=3,
                            min_dist=400,
                            min_size=36,
                           ):

    hsv_image = rgb2hsv(rgb_image)

    spam = np.zeros_like(hsv_image[:,:,2])
    # spam =
##    hue_spam = np.copy(hsv_image[:,:,0]) ### hue
##    sat_spam = np.copy(hsv_image[:,:,1]) ### saturation
##    val_spam = np.copy(hsv_image[:,:,2]) ### value
    # filt = ((hue_spam < 0.25 ) | (hue_spam > 0.8)) & (sat_spam > np.mean(sat_spam))
    filt = ((hsv_image[:,:,0] < 0.4 ) | (hsv_image[:,:,0] > 0.5)) & (hsv_image[:,:,1] > np.quantile(hsv_image[:,:,1],0.5))
    print(filt[filt==True].size)

    # new_spam = new_spam[new_spam > 0.7])
    useuse = -1*np.copy(hsv_image[:,:,2])
    usemin = np.min(useuse)
    usemax = np.max(useuse)
    useuse = (useuse - usemin)/(usemax - usemin)

    hsv_filt =  useuse * filt

    blurred = gaussian_blur(255*hsv_filt, variance_x=blur_radius, variance_y=blur_radius)

    bigblur = gaussian_blur(255*hsv_filt, variance_x=500, variance_y=500)

    blurred2 = blurred - bigblur

    #### binary and to eliminate noise
    res = filters.threshold_multiotsu(blurred,classes=3)
    binary = blurred > res[0]

    if res[1] >= 0.2:
        binary2 = blurred > res[0]
    else:
        binary2 = blurred > res[1]

    print(binary.size,binary2.size)

    coords = feature.peak_local_max(blurred2, min_distance=min_dist)

    peak_mask = np.zeros(blurred2.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    markers = np.logical_and(binary2,markers)
    markers = measure.label(markers)

    ##    markers = markers * nuc_mask
##    print("n spheroid markers",np.unique(markers[markers > 0]))


##    nuc_labels, details = model.predict_instances(dapi_image)
    nuc_labels = morphology.dilation(nuc_labels,morphology.square(10))

    last_mask = ((binary2 > 0) | (nuc_labels > 0))

    hsv_filt = hsv_filt * last_mask
    distance = ndi.distance_transform_edt(binary2)
    print("simple mask",np.unique(measure.label(last_mask)))
    labels = measure.label(last_mask)
    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
##        print("simple sizes",l,xtent,ytent,labels[labels==l].size)


    # labels = segmentation.watershed(-hsv_filt, markers, mask=last_mask)
    ##    labels = segmentation.watershed(-hsv_filt, markers, mask=nuc_mask)
    # labels = segmentation.watershed(-blurred, markers, mask=binary)
    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = morphology.remove_small_holes(labels, area_threshold=36)
    labels = measure.label(labels)
    labels = segmentation.clear_border(labels, buffer_size=2) #, bgval=0, mask=None, *, out=None)[source]

    # for l in np.unique(labels[labels>0]):
    #     print(l,labels[labels==l].size)


    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
##        print(l,xtent,ytent,labels[labels==l].size)
        if xtent > 500 or xtent < 10 or ytent > 500 or ytent < 10:
    ##            print(l,xtent,ytent,labels[labels==l].size)
            labels[labels==l] = 0
        if labels[labels==l].size > 250000 or labels[labels==l].size < 450:
    ##            print(l,labels[labels==l].size)
            labels[labels==l] = 0


    labels = measure.label(labels)
    print("sph segmentation dense : markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))

    spamlabels =1000*labels
    io.imsave(file_prefix+'_spheroid_labels.tif',spamlabels.astype('uint8'))
    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_labels.png')


    spamlabels =10000*hsv_filt
    print(file_prefix+'_spheroid_chan_intensity.tif')
    io.imsave(file_prefix+'_spheroid_chan_intensity.tif',spamlabels.astype('uint8'))
    plt.clf()
##    cm = get_colors('jet',n=256)
##    maxlabel = len(np.unique(labels))
##    offset = np.zeros_like(labels)
##    offset[labels>0] = maxlabel
    plt.imshow(hsv_filt,cmap='viridis',vmin=0,vmax=np.max(hsv_filt))
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_chan_intensity.png')

    return labels,markers,hsv_filt

def get_spheroid_masks_rgb_conservative(rgb_image,nuc_labels,file_prefix,model,
                            blur_radius=30,
                            mo_classes=3,
                            min_dist=400,
                            min_size=36,
                           ):

    hsv_image = rgb2hsv(rgb_image)

    spam = np.zeros_like(hsv_image[:,:,2])
    # spam =
##    hue_spam = np.copy(hsv_image[:,:,0]) ### hue
##    sat_spam = np.copy(hsv_image[:,:,1]) ### saturation
##    val_spam = np.copy(hsv_image[:,:,2]) ### value
    # filt = ((hue_spam < 0.25 ) | (hue_spam > 0.8)) & (sat_spam > np.mean(sat_spam))
    filt = ((hsv_image[:,:,0] < 0.4 ) | (hsv_image[:,:,0] > 0.5)) & (hsv_image[:,:,1] > np.quantile(hsv_image[:,:,1],0.5))
    print(filt[filt==True].size)

    # new_spam = new_spam[new_spam > 0.7])
    useuse = -1*np.copy(hsv_image[:,:,2])
    usemin = np.min(useuse)
    usemax = np.max(useuse)
    useuse = (useuse - usemin)/(usemax - usemin)

    hsv_filt =  useuse * filt

    # Apply multi-Otsu thresholding
    res = filters.threshold_multiotsu(hsv_filt, classes=3)
    binary = hsv_filt > res[0]

    # Check if the first threshold covers more than 50% of the image
    if np.sum(binary) > 0.25 * hsv_filt.size:
        binary = hsv_filt > res[1]

    # Expand nuclear labels
    expanded_nuc_labels = segmentation.expand_labels(nuc_labels, distance=10)
    expanded_nuc_labels = morphology.remove_small_holes(expanded_nuc_labels, area_threshold=64)

    # Generate mask for regions covered by expanded nuclear labels
    expanded_nuc_mask = expanded_nuc_labels > 0

    # Calculate product of the two masks
    final_mask = np.logical_and(binary, expanded_nuc_mask)

    hsv_filt = hsv_filt * final_mask
    print("simple mask",np.unique(measure.label(final_mask)))
    labels = measure.label(final_mask)
    for l in np.unique(labels[labels>0]):
        # print(print(np.where(labels==l)))
        xtent = max(np.where(labels==l)[0]) - min(np.where(labels==l)[0])
        ytent = max(np.where(labels==l)[1]) - min(np.where(labels==l)[1])
        if xtent > 500 or xtent < 16 or ytent > 500 or ytent < 16:
            labels[labels==l] = 0
        if labels[labels==l].size > 250000 or labels[labels==l].size < 256:
            labels[labels==l] = 0


    labels = morphology.remove_small_objects(labels, min_size=min_size)
    labels = morphology.binary_closing(labels)
    # labels = morphology.binary_fill_holes(labels)
    labels = morphology.remove_small_holes(labels, area_threshold=128)
    labels = measure.label(labels)
    labels = segmentation.clear_border(labels, buffer_size=2) #, bgval=0, mask=None, *, out=None)[source]

    # Calculate the centroids of each labeled region
    region_props = measure.regionprops(labels)
    markers = np.zeros_like(labels, dtype=np.int32)
    for i, region in enumerate(region_props):
        centroid = region.centroid
        markers[int(centroid[0]), int(centroid[1])] = i + 1

    print("sph segmentation dense : markers ",np.unique(labels),markers[markers>0].shape)
    maxlabel = np.max(np.array(np.unique(labels)))

    spamlabels =1000*labels
    io.imsave(file_prefix+'_spheroid_labels.tif',spamlabels.astype('uint8'))
    plt.clf()
    cm = get_colors('jet',n=256)
    maxlabel = len(np.unique(labels))
    offset = np.zeros_like(labels)
    offset[labels>0] = maxlabel
    plt.imshow(offset + labels,cmap=cm,vmin=0,vmax=2*maxlabel)
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_labels.png')


    spamlabels =10000*hsv_filt
    print(file_prefix+'_spheroid_chan_intensity.tif')
    io.imsave(file_prefix+'_spheroid_chan_intensity.tif',spamlabels.astype('uint8'))
    plt.clf()

    plt.imshow(hsv_filt,cmap='viridis',vmin=0,vmax=np.max(hsv_filt))
    plt.tight_layout()
    plt.savefig(file_prefix+'_spheroid_chan_intensity.png')

    return labels,markers,hsv_filt


RGBAPProfile = str


def get_spheroid_masks_rgb(*args, profile: RGBAPProfile = "expanded", **kwargs):
    """Segment RGB/AP spheroids with an explicit morphology profile."""
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized == "expanded":
        return get_spheroid_masks_rgb_expanded(*args, **kwargs)
    if normalized == "conservative":
        return get_spheroid_masks_rgb_conservative(*args, **kwargs)
    raise ValueError(f"Unknown RGB/AP segmentation profile: {profile!r}")


__all__ = [
    "get_colors",
    "get_nuclear_masks_dense",
    "get_nuclear_masks_standard",
    "get_spheroid_masks_rgb",
    "get_spheroid_masks_rgb_conservative",
    "get_spheroid_masks_rgb_no_dapi",
    "get_spheroid_masks_rgb_model",
    "get_spheroid_masks_rgb_expanded",
]
