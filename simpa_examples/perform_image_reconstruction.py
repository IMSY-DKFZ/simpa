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
import time

PATH = "D:/save/LNetOpticalForward_planar_LARGE.hdf5"
WAVELENGTH = 532

file = load_hdf5(PATH)
settings = Settings(file["settings"])
settings[Tags.WAVELENGTH] = WAVELENGTH
settings[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_BACKPROJECTION
settings[Tags.RECONSTRUCTION_MODE] = Tags.RECONSTRUCTION_MODE_FULL
settings[Tags.DIGITAL_DEVICE_POSITION] = [0, 0, -1.4]
acoustic_data_path = generate_dict_path(settings, Tags.TIME_SERIES_DATA, wavelength=settings[Tags.WAVELENGTH],
                                                upsampled_data=True)
optical_data_path = generate_dict_path(settings, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=settings[Tags.WAVELENGTH],
                                                upsampled_data=True)
device = RSOMExplorerP50()
device.check_settings_prerequisites(settings)
settings = device.adjust_simulation_volume_and_settings(settings)
time_series_data = load_hdf5(PATH, acoustic_data_path)[Tags.TIME_SERIES_DATA]
initial_pressure = load_hdf5(PATH, optical_data_path)[Tags.OPTICAL_MODEL_INITIAL_PRESSURE]

print(Tags.ACOUSTIC_SIMULATION_3D in settings)
print(settings[Tags.ACOUSTIC_SIMULATION_3D])
start = time.time()

perform_reconstruction(settings)

print("Took", time.time()-start, "seconds")

reconstructed_image_path = generate_dict_path(settings, Tags.RECONSTRUCTED_DATA,
                                              wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)

reconstructed_image = load_hdf5(PATH, reconstructed_image_path)[Tags.RECONSTRUCTED_DATA]
print(np.shape(reconstructed_image))

if len(np.shape(initial_pressure)) < 3:
    plt.subplot(121)
    plt.imshow(initial_pressure)
    plt.subplot(122)
    plt.imshow(np.abs(reconstructed_image[:, int(np.shape(reconstructed_image)[1]/2), :]))
    plt.show()
else:
    plt.subplot(161)
    plt.imshow(initial_pressure[int(np.shape(initial_pressure)[0]/2), :, :])
    plt.subplot(162)
    plt.imshow(np.abs(reconstructed_image[int(np.shape(reconstructed_image)[0]/2), :, :]))
    plt.subplot(163)
    plt.imshow(initial_pressure[:, int(np.shape(initial_pressure)[1]/2), :])
    plt.subplot(164)
    plt.imshow(np.abs(reconstructed_image[:, int(np.shape(reconstructed_image)[1]/2), :]))
    plt.subplot(165)
    plt.imshow(initial_pressure[:, :, int(np.shape(initial_pressure)[2]/2)])
    plt.subplot(166)
    plt.imshow(np.abs(reconstructed_image[:, :, int(np.shape(reconstructed_image)[2]/2)]))
    plt.show()