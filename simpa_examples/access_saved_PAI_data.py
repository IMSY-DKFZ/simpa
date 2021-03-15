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

PATH = "D:/mcx-tmp-output/MyVolumeName_471.hdf5"
WAVELENGTH = 700

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

    reconstruction = reconstruction.T

shape = np.shape(initial_pressure)

x_pos = int(shape[0]/2)
y_pos = int(shape[1]/2)
z_pos = int(shape[2]/2)

plt.figure()
plt.imshow(np.rot90(fluence[:, y_pos, :], -1))
plt.show()

