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

from simpa.io_handling import load_hdf5
from simpa.utils.settings_generator import Settings
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils import Tags
from simpa.core.device_digital_twins.rsom_device import RSOMExplorerP50
from simpa.core.image_reconstruction.reconstruction_modelling import perform_reconstruction
import matplotlib.pyplot as plt
import numpy as np

PATH = "D:/save/LNetOpticalForward_planar.hdf5"
WAVELENGTH = 532

file = load_hdf5(PATH)
settings = Settings(file["settings"])
print(settings)
settings[Tags.WAVELENGTH] = WAVELENGTH
settings[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_BACKPROJECTION
acoustic_data_path = generate_dict_path(settings, Tags.TIME_SERIES_DATA, wavelength=settings[Tags.WAVELENGTH],
                                                upsampled_data=True)
device = RSOMExplorerP50()
settings = device.adjust_simulation_volume_and_settings(settings)
time_series_data = load_hdf5(PATH, acoustic_data_path)[Tags.TIME_SERIES_DATA]

print(Tags.ACOUSTIC_SIMULATION_3D in settings)
print(settings[Tags.ACOUSTIC_SIMULATION_3D])

perform_reconstruction(settings)

reconstructed_image_path = generate_dict_path(settings, Tags.RECONSTRUCTED_DATA,
                                              wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)

reconstructed_image = load_hdf5(PATH, reconstructed_image_path)[Tags.RECONSTRUCTED_DATA]

print(np.shape(reconstructed_image))

plt.subplot(121)
plt.imshow(np.rot90(time_series_data, -3), aspect=0.2)
plt.subplot(122)
plt.imshow(reconstructed_image[40, :, :])
plt.show()