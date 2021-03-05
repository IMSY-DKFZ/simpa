# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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

from simpa.io_handling import load_hdf5, load_data_field
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags
from simpa.core.device_digital_twins.msot_devices import MSOTAcuityEcho
from simpa.core.image_reconstruction.reconstruction_modelling import perform_reconstruction
import numpy as np
import time
from simpa_examples.access_saved_PAI_data import visualise_data

PATH = "/path/to/time/series/file.hdf5"
WAVELENGTH = 700

file = load_hdf5(PATH)
settings = Settings(file["settings"])
settings[Tags.WAVELENGTH] = WAVELENGTH
settings[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_PYTORCH_DAS
settings[Tags.RECONSTRUCTION_MODE] = Tags.RECONSTRUCTION_MODE_PRESSURE

device = MSOTAcuityEcho()
device.check_settings_prerequisites(settings)
settings = device.adjust_simulation_volume_and_settings(settings)
time_series_data = load_data_field(PATH, Tags.TIME_SERIES_DATA, WAVELENGTH)
initial_pressure = load_data_field(PATH, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, WAVELENGTH)

print(Tags.ACOUSTIC_SIMULATION_3D in settings)
print(settings[Tags.ACOUSTIC_SIMULATION_3D])
start = time.time()

perform_reconstruction(settings)

print("Took", time.time()-start, "seconds")

reconstructed_image = load_data_field(PATH, Tags.RECONSTRUCTED_DATA, WAVELENGTH)
reconstructed_image = np.squeeze(reconstructed_image)

visualise_data(PATH, WAVELENGTH, show_absorption=False,
               show_initial_pressure=False,
               show_segmentation_map=False)
