"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

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

# FIXME temporary workaround for newest Intel architectures
import os


def find_label_regions(image):
    thresh = threshold_li(image)
    bw = closing(image > thresh)
    #label_image = label(bw)
    return bw

PATH = '/home/p253n/project_HN/RecoTests/results/CompletePipelineTestMSOT3_4711.hdf5'
#PATH = '/home/p253n/project_HN/depth_effect_simulations/results/Blood_differential_no_bandpass_depth_6_side_8.hdf5'

initial_pressure = sp.load_data_field(PATH, data_field=Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=700)
reconstruction = sp.load_data_field(PATH, data_field=Tags.RECONSTRUCTED_DATA, wavelength=700)
segmentation = sp.load_data_field(PATH, data_field=Tags.PROPERTY_SEGMENTATION, wavelength=700)
reconstruction = (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
initial_pressure = (initial_pressure - np.min(initial_pressure)) / (np.max(initial_pressure) - np.min(initial_pressure))

minimum_connected_pixels = initial_pressure.shape[0] * initial_pressure.shape[1] * 0.005

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
joinseg_reconstruction = label(closing(joinseg_reconstruction, square(5)), connectivity=2)

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

props_initial_pressure = regionprops(joinseg_initial_pressure)
props_reconstruction = regionprops(joinseg_reconstruction)

mse = 0
for label_idx in range(np.max(joinseg_initial_pressure)):
    print(f'centroid position for initial pressure at label {label_idx} : '
          f'{int(props_initial_pressure[label_idx].centroid[0])}, {int(props_initial_pressure[label_idx].centroid[1])}')
    print(f'centroid position for reconstruction at label {label_idx} : '
          f'{int(props_reconstruction[label_idx].centroid[0])}, {int(props_reconstruction[label_idx].centroid[1])}')
    mse += np.sqrt((int(props_initial_pressure[label_idx].centroid[0]) - int(props_reconstruction[label_idx].centroid[0])) ** 2 + \
          (int(props_initial_pressure[label_idx].centroid[1]) - int(props_reconstruction[label_idx].centroid[1])) ** 2)

MSE = mse / np.max(joinseg_initial_pressure)
print(MSE)
