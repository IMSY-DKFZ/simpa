# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.io_handling import load_hdf5, save_hdf5
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from simpa.utils import SegmentationClasses, Tags
from simpa.utils.settings_generator import Settings

values = []
names = []

for string in SegmentationClasses.__dict__:
    if string[0:2] != "__":
        values.append(SegmentationClasses.__dict__[string])
        names.append(string)

values = np.asarray(values)
names = np.asarray(names)
sort_indexes = np.argsort(values)
values = values[sort_indexes]
names = names[sort_indexes]

colors = [list(np.random.random(3)) for _ in range(len(names))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', colors, len(names))

PATH = "D:/save/LNetOpticalForward_planar_SMALL.hdf5"
WAVELENGTH = 532

file = load_hdf5(PATH)
settings = Settings(file["settings"])

fluence = (file['simulations']['original_data']['optical_forward_model_output']
           [str(WAVELENGTH)]['fluence'])
initial_pressure = (file['simulations']['original_data']
                    ['optical_forward_model_output']
                    [str(WAVELENGTH)]['initial_pressure'])
absorption = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['mua'])

segmentation = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['seg'])

reconstruction = None
speed_of_sound = None
if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings and settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
    time_series = np.squeeze(
        file["simulations"]["original_data"]["time_series_data"][str(WAVELENGTH)]["time_series_data"])
    reconstruction = np.squeeze(
            file["simulations"]["original_data"]["reconstructed_data"][str(WAVELENGTH)]["reconstructed_data"])

    speed_of_sound = file['simulations']['original_data']['simulation_properties'][str(WAVELENGTH)]["sos"]

shape = np.shape(fluence)

x_pos = int(shape[0]/2)
y_pos = 0

if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings and settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
    if len(shape) > 2:
        plt.figure()
        plt.subplot(141)
        plt.imshow(np.rot90((time_series[:, :]), -1), aspect=0.2)
        plt.subplot(142)
        plt.imshow(np.rot90((reconstruction[:, y_pos, :]), -1))
        plt.subplot(143)
        plt.imshow(np.rot90(np.log10(initial_pressure[:, y_pos, :]), -1))
        plt.subplot(144)
        plt.imshow(np.rot90(segmentation[:, y_pos, :], -1), vmin=values[0], vmax=values[-1], cmap=cmap)
        plt.show()
    else:
        plt.figure()
        plt.subplot(141)
        plt.imshow(np.rot90((reconstruction[:, :]), -1))
        plt.subplot(142)
        plt.imshow(np.rot90((speed_of_sound), -1))
        plt.subplot(143)
        plt.imshow(np.rot90(np.log10(initial_pressure), -1))
        plt.subplot(144)
        plt.imshow(np.rot90(segmentation, -1), vmin=values[0], vmax=values[-1], cmap=cmap)
        plt.show()
else:
    if len(shape) > 2:
        plt.figure()
        plt.subplot(241)
        plt.title("Fluence")
        plt.imshow(np.rot90((fluence[x_pos, :, :]), -1))
        plt.subplot(242)
        plt.title("Absorption")
        plt.imshow(np.rot90(np.log10(absorption[x_pos, :, :]), -1))
        plt.subplot(243)
        plt.title("Initial Pressure")
        plt.imshow(np.rot90(np.log10(initial_pressure[x_pos, :, :]), -1))
        plt.subplot(244)
        plt.title("Segmentation")
        plt.imshow(np.rot90(segmentation[x_pos, :, :], -1), vmin=values[0], vmax=values[-1], cmap=cmap)
        cbar = plt.colorbar(ticks=values)
        cbar.ax.set_yticklabels(names)
        plt.subplot(245)
        plt.imshow(np.rot90(fluence[:, y_pos, :], -1))
        plt.subplot(246)
        plt.imshow(np.rot90(np.log10(absorption[:, y_pos, :]), -1))
        plt.subplot(247)
        plt.imshow(np.rot90(np.log10(initial_pressure[:, y_pos, :]), -1))
        plt.subplot(248)
        plt.imshow(np.rot90(segmentation[:, y_pos, :], -1), vmin=values[0], vmax=values[-1], cmap=cmap)
        cbar = plt.colorbar(ticks=values)
        cbar.ax.set_yticklabels(names)
        plt.show()
    else:
        plt.figure()
        plt.subplot(141)
        plt.imshow(np.rot90(np.log10(fluence), -1))
        plt.subplot(142)
        plt.imshow(np.rot90(np.log10(absorption), -1))
        plt.subplot(143)
        plt.imshow(np.rot90(np.log10(initial_pressure), -1))
        plt.subplot(144)
        plt.imshow(np.rot90(segmentation, -1))
        plt.show()
