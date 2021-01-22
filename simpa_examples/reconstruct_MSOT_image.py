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

import nrrd
import numpy as np
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags
from simpa.core.image_reconstruction.reconstruction_modelling import perform_reconstruction
import matplotlib.pyplot as plt
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling import load_hdf5, save_hdf5


from simpa.core.device_digital_twins.msot_devices import MSOTAcuityEcho

raw, _ = nrrd.read("I:/reco/Scan_10.PA.rf.0000000.nrrd")
raw = raw[:, :, 0]
das, _ = nrrd.read("I:/reco/Scan_10.PA.bf.0000000.nrrd")
das = das[:, :, 0]

settings = Settings()
settings[Tags.DIM_VOLUME_X_MM] = 38.4
settings[Tags.DIM_VOLUME_Y_MM] = 0.15
settings[Tags.DIM_VOLUME_Z_MM] = 38.4
settings[Tags.SPACING_MM] = 0.15
settings[Tags.PERFORM_IMAGE_RECONSTRUCTION] = True
settings[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_BACKPROJECTION
settings[Tags.ACOUSTIC_MODEL_BINARY_PATH] = "C:/Program Files/MATLAB/R2020b/bin/matlab.exe"
settings[Tags.PROPERTY_ALPHA_POWER] = 1.05
settings[Tags.GPU] = True
settings[Tags.PMLInside] = False
settings[Tags.PMLSize] = [31, 32]
settings[Tags.PMLAlpha] = 1.5
settings[Tags.PlotPML] = False
settings[Tags.RECORDMOVIE] = False
settings[Tags.MOVIENAME] = "visualization_log"
settings[Tags.ACOUSTIC_LOG_SCALE] = True
settings[Tags.SIMULATION_PATH] = "D:/save/"
settings[Tags.VOLUME_CREATOR] = "None"
settings[Tags.WAVELENGTH] = 700
settings[Tags.SIMPA_OUTPUT_PATH] = "D:/save/TestFile.hdf5"
settings[Tags.DIGITAL_DEVICE] = Tags.DIGITAL_DEVICE_MSOT
settings[Tags.DIGITAL_DEVICE_POSITION] = [0, 0, 0]
device = MSOTAcuityEcho()
settings = device.adjust_simulation_volume_and_settings(settings)

print(settings)

save_hdf5({Tags.SETTINGS: settings}, settings[Tags.SIMPA_OUTPUT_PATH])
time_series_image_path = generate_dict_path(settings, Tags.TIME_SERIES_DATA,
                                            wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)
save_hdf5({Tags.TIME_SERIES_DATA: raw}, settings[Tags.SIMPA_OUTPUT_PATH], time_series_image_path)
acoustic_data_path = generate_dict_path(settings, Tags.PROPERTY_SPEED_OF_SOUND,
                                                wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)
save_hdf5({Tags.PROPERTY_SPEED_OF_SOUND: 1540}, settings[Tags.SIMPA_OUTPUT_PATH], acoustic_data_path)

perform_reconstruction(settings)

reconstructed_image_path = generate_dict_path(settings, Tags.RECONSTRUCTED_DATA,
                                              wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)
reconstructed_image = load_hdf5("D:/save/TestFile.hdf5", reconstructed_image_path)[Tags.RECONSTRUCTED_DATA]

shape = np.shape(reconstructed_image)

if len(np.shape(reconstructed_image)) > 2:
    plt.subplot(121)
    plt.imshow(np.rot90(das, -1))
    plt.subplot(122)
    plt.imshow(np.rot90(np.abs(reconstructed_image[:, 0, :]), -1))
    plt.show()
else:
    plt.subplot(121)
    plt.imshow(np.rot90(das, -1))
    plt.subplot(122)
    plt.imshow(np.rot90(np.abs(reconstructed_image), -1))
    plt.show()
