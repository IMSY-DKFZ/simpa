# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from queue import Empty
from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import closing, erosion, dilation

from skimage.filters import threshold_otsu, threshold_li
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.segmentation import slic, join_segmentations, watershed
from skimage.filters import try_all_threshold


def find_label_regions(image):
    '''
    TODO
    :param image: the image to be thresholded
    :return: binary_mask
    '''
    thresh = threshold_li(image)
    binary_mask = closing(image > thresh)
    #label_image = label(bw)
    return binary_mask

def initialize_mse_dict():
    '''
    Initialization of dictionary that stores MSE values.
    :return: mse
    '''
    mse = {}
    mse['centroid'] = 0
    mse['num_pixels'] = 0
    return mse

def quantitative_results(props_initial_pressure, props_reconstruction, label_, label_reco_, mse):
    '''
    Calculation of quantitative measures between segmentation properties of initial pressure and reconstruction
    :param props_initial_pressure: region properties of initial pressure segmentation
    :param props_reconstruction: region properties of reconstruction segmentation
    :param label_: label_id of props_initial_pressure
    :param label_reco_: label_id of props_reconstruction
    :param mse: mean squarred error value
    :return: mse
    '''
    print(f'centroid position for initial pressure at label {label_} : '
        f'{int(props_initial_pressure[label_].centroid[0])}, {int(props_initial_pressure[label_].centroid[1])}')
    print(f'centroid position for reconstruction at label {label_} : '
        f'{int(props_reconstruction[label_reco_].centroid[0])}, {int(props_reconstruction[label_reco_].centroid[1])}')
    mse['centroid'] += np.sqrt(
        (int(props_initial_pressure[label_].centroid[0]) - \
        int(props_reconstruction[label_reco_].centroid[0])) ** 2 + \
        (int(props_initial_pressure[label_].centroid[1]) - \
        int(props_reconstruction[label_reco_].centroid[1])) ** 2)
    print(f'pixels for initial pressure at label {label_} : '
        f'{int(props_initial_pressure[label_].area)}')
    print(f'pixels for reconstruction at label {label_} : '
        f'{int(props_reconstruction[label_reco_].area)}')
    mse['num_pixels'] += np.sqrt((
        int(props_initial_pressure[label_].area) - \
        int(props_reconstruction[label_reco_].area)) ** 2)
        
    return mse

def match_labels(joinseg_initial_pressure, joinseg_reconstruction):
    '''
    Matches the labels of the inital pressure and reconstruction segmentation, since the number of labels can vary. For every label in the initial pressure segmentation, the corresponding labeled region in the reconstructions is assigned the same label. 

    NOTE and TODO Not sure if this works also if the reco segmentation has more classes than the p0 segmentation. 

    :param joinseg_initial_pressure: segmentation of initial pressure
    :param joinseg_reconstruction: segmentation of reconstruction s
    :return: joinseg_reconstruction_new
    '''
    joinseg_reconstruction_new = np.zeros((joinseg_reconstruction.shape), dtype=np.int64)

    label_idx = 1
    while label_idx <= np.max(joinseg_initial_pressure):
        X, Y = np.where(joinseg_initial_pressure == label_idx)
        for x, y in zip(X,Y):
            # print(x, y)
            if joinseg_reconstruction[x,y] != 0:
                label_reco = joinseg_reconstruction[x,y]
                joinseg_reconstruction_new[joinseg_reconstruction==label_reco] = label_idx
                break
        label_idx += 1
    return joinseg_reconstruction_new

def region_properties_matching_and_quantification(props_initial_pressure, props_reconstruction, label_, label_reco_, mse):
    '''
    Matches the labels of the region properties of inital pressure and reconstruction segmentation, since the original label order is lost. (E.g., if the reconstruction did not have a label 3, the label 3 of region properties would correspond to original label 4) and then calculates quantitative comparison values.
    :param props_initial_pressure: region properties of initial pressure segmentation
    :param props_reconstruction: region properties of reconstruction segmentation
    :param label_: label_id of props_initial_pressure
    :param label_reco_: label_id of props_reconstruction
    :param mse: mean squarred error value
    :return: mse
    '''
    if props_initial_pressure[label_].label == props_reconstruction[label_reco_].label:
        mse = quantitative_results(props_initial_pressure, props_reconstruction, label_, label_reco_, mse)
        # print(label_+1, label_reco_+1)
        return mse, label_+1, label_reco_+1
    else: 
        # print(label_+1, label_reco_)
        return mse, label_+1, label_reco_




PATH = '/home/melanie/workplace/data/reco_test/CompletePipelineTestMSOTtest_4711.hdf5'

# load data
initial_pressure = sp.load_data_field(PATH, data_field=Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength=700)
reconstruction = sp.load_data_field(PATH, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=700)
segmentation = sp.load_data_field(PATH, data_field=Tags.DATA_FIELD_SEGMENTATION, wavelength=700)

# min max normalisation of reconstruction and initial_pressure images
reconstruction = (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
initial_pressure = (initial_pressure - np.min(initial_pressure)) / (np.max(initial_pressure) - np.min(initial_pressure))

# min connected pixels are 0.5 % of all pixels
minimum_connected_pixels = initial_pressure.shape[0] * initial_pressure.shape[1] * 0.005

# segmentation
initial_pressure_labels = find_label_regions(initial_pressure)
reconstruction_labels = find_label_regions(reconstruction)

edges_initial_pressure = closing(feature.canny(initial_pressure), square(5))
edges_reconstruction = closing(feature.canny(reconstruction), square(5))

joinseg_initial_pressure = np.zeros((initial_pressure_labels.shape)).astype(int)
for x in range(initial_pressure_labels.shape[0]):
    for y in range(initial_pressure_labels.shape[1]):
        if edges_initial_pressure[x, y] == 1 and initial_pressure_labels[x, y] == 1:
            joinseg_initial_pressure[x, y] = 1

joinseg_reconstruction = np.zeros((reconstruction_labels.shape)).astype(int)
for x in range(reconstruction_labels.shape[0]):
    for y in range(reconstruction_labels.shape[1]):
        if edges_reconstruction[x, y] == 1 and reconstruction_labels[x, y] == 1:
            joinseg_reconstruction[x, y] = 1

joinseg_initial_pressure = label(closing(joinseg_initial_pressure, square(5)), connectivity=2)
joinseg_initial_pressure = np.rot90(joinseg_initial_pressure, -1)
joinseg_reconstruction = label(closing(joinseg_reconstruction, square(5)), connectivity=2)
joinseg_reconstruction = np.rot90(joinseg_reconstruction, -1)


# accept only labels of minimum_connected_pixels 
for label_idx in range(np.max(joinseg_initial_pressure) + 1):
    mask = joinseg_initial_pressure[joinseg_initial_pressure == label_idx]
    label_count = np.sum(mask)
    if label_count < minimum_connected_pixels:
        joinseg_initial_pressure[joinseg_initial_pressure == label_idx] = 0.0
    else:
        joinseg_initial_pressure[joinseg_initial_pressure == label_idx] = 1.0

for label_idx in range(np.max(joinseg_reconstruction) + 1):
    mask = joinseg_reconstruction[joinseg_reconstruction == label_idx]
    label_count = np.sum(mask)
    if label_count < minimum_connected_pixels:
        joinseg_reconstruction[joinseg_reconstruction == label_idx] = 0.0
    else:
        joinseg_reconstruction[joinseg_reconstruction == label_idx] = 1.0

joinseg_initial_pressure = label(joinseg_initial_pressure, connectivity=2)
joinseg_reconstruction = label(joinseg_reconstruction, connectivity=2)

# match label classes
joinseg_reconstruction = match_labels(joinseg_initial_pressure, joinseg_reconstruction) 

plt.imshow(joinseg_initial_pressure)
plt.colorbar()
plt.savefig('/home/melanie/workplace/data/reco_test/p0.png')
plt.close()
plt.imshow(joinseg_reconstruction)
plt.colorbar()
plt.savefig('/home/melanie/workplace/data/reco_test/reco.png')
plt.close()

plt.subplot(231)
plt.imshow(np.rot90(segmentation, -1))
plt.subplot(232)
plt.imshow(np.rot90(initial_pressure, -1))
plt.colorbar(fraction=0.035)
plt.subplot(233)
plt.imshow(np.rot90(reconstruction, -1))
plt.colorbar(fraction=0.035)
plt.subplot(235)
plt.imshow(np.rot90(joinseg_initial_pressure, -1))
plt.colorbar(fraction=0.035)
plt.subplot(236)
plt.imshow(np.rot90(joinseg_reconstruction, -1))
plt.colorbar(fraction=0.035)
plt.show()

# DSC TODO add other matrices
DSC = metrics.f1_score(joinseg_initial_pressure.reshape(-1), joinseg_reconstruction.reshape(-1), average='macro', labels=[1,2,3,4,5])
print(f'DSC (non-weighted average) is: ', DSC)

# region properties
props_initial_pressure = regionprops(joinseg_initial_pressure)
props_reconstruction = regionprops(joinseg_reconstruction)

# centroid position comparison
label_ = 0
label_reco_ = 0
mse = initialize_mse_dict()
while label_ <= np.max(joinseg_initial_pressure)-1:
    mse, label_, label_reco_= region_properties_matching_and_quantification(props_initial_pressure, props_reconstruction, label_, label_reco_, mse)

# MSE = mse / np.max(joinseg_initial_pressure)
print(mse)
