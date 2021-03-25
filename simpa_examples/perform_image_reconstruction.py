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
import numpy as np
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa.core.pipeline_components import DelayAndSumReconstruction

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PATH = "D:/mcx-tmp-output/CompletePipelineTestMSOT_4711.hdf5"

file = load_hdf5(PATH)
settings = Settings(file["settings"])
settings[Tags.WAVELENGTH] = settings[Tags.WAVELENGTHS][0]
settings["reco_settings"] = {
    Tags.RECONSTRUCTION_ALGORITHM: Tags.RECONSTRUCTION_ALGORITHM_PYTORCH_DAS,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE
}

device = MSOTAcuityEcho()
device.check_settings_prerequisites(settings)
settings = device.adjust_simulation_volume_and_settings(settings)

DelayAndSumReconstruction(settings, "reco_settings").run()

reconstructed_image = load_data_field(PATH, Tags.RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])
reconstructed_image = np.squeeze(reconstructed_image)

visualise_data(PATH, settings[Tags.WAVELENGTH], show_absorption=False,
               show_initial_pressure=True,
               show_segmentation_map=True,
               log_scale=False)
