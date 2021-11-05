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

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

# FIXME temporary workaround for newest Intel architectures
import os

PATH = '/home/p253n/project_HN/RecoTests/results/CompletePipelineTestMSOT_4711.hdf5'
initial_pressure = sp.load_data_field(PATH, data_field=Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=700)
reconstruction = sp.load_data_field(PATH, data_field=Tags.RECONSTRUCTED_DATA, wavelength=700)
segmentation = sp.load_data_field(PATH, data_field=Tags.PROPERTY_SEGMENTATION, wavelength=700)

reconstruction = (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
initial_pressure = (initial_pressure - np.min(initial_pressure)) / (np.max(initial_pressure) - np.min(initial_pressure))

edges_initial_pressure = feature.canny(initial_pressure)
edges_reconstruction = feature.canny(reconstruction)

edges_initial_pressure = erosion(dilation(edges_initial_pressure))
edges_reconstruction = erosion(dilation(edges_reconstruction))

# mi = metrics.normalized_mutual_info_score(edges_initial_pressure.flatten(), edges_reconstruction.flatten())
# print(mi)
# ssi = ssim(initial_pressure.flatten(), reconstruction.flatten())
# print(ssi)

# apply threshold
thresh = threshold_otsu(reconstruction)
bw = reconstruction.copy()
bw[bw < thresh] = 0
bw[bw >= thresh] = 1
#bw = closing(reconstruction > thresh)
# remove artifacts connected to image border
#cleared = clear_border(bw)
# label image regions
label_image = label(bw)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
edges_reconstruction = label2rgb(label_image, image=reconstruction, bg_label=0)

# apply threshold
thresh = threshold_otsu(initial_pressure)
bw = closing(initial_pressure > thresh)
# remove artifacts connected to image border
#cleared = clear_border(bw)
# label image regions
label_image = label(bw)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
edges_initial_pressure = label2rgb(label_image, image=initial_pressure, bg_label=0)

plt.subplot(231)
plt.imshow(np.rot90(segmentation, -1))
plt.subplot(232)
plt.imshow(np.rot90(initial_pressure, -1))
plt.colorbar(fraction=0.035)
plt.subplot(233)
plt.imshow(np.rot90(reconstruction, -1))
plt.colorbar(fraction=0.035)
plt.subplot(235)
plt.imshow(np.rot90(edges_initial_pressure, -1))
plt.colorbar(fraction=0.035)
plt.subplot(236)
plt.imshow(np.rot90(edges_reconstruction, -1))
plt.colorbar(fraction=0.035)
plt.show()