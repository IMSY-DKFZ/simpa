"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.io_handling import load_hdf5, load_data_field
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.path_manager import PathManager
from simpa.core.device_digital_twins.devices.pa_devices.ithera_msot_acuity import MSOTAcuityEcho
import numpy as np
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa.core import ImageReconstructionModuleDelayAndSumAdapter

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = PathManager()
PATH = path_manager.get_hdf5_file_save_path() + "/CompletePipelineTestMSOT_4711.hdf5"

file = load_hdf5(PATH)
settings = Settings(file["settings"])
settings[Tags.WAVELENGTH] = settings[Tags.WAVELENGTHS][0]

settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e6),
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
    Tags.SPACING_MM: settings[Tags.SPACING_MM]
})

# TODO use the correct device definition here
device = MSOTAcuityEcho()

ImageReconstructionModuleDelayAndSumAdapter(settings).run(device)

reconstructed_image = load_data_field(PATH, Tags.RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])
reconstructed_image = np.squeeze(reconstructed_image)

visualise_data(PATH, settings[Tags.WAVELENGTH], show_absorption=False,
               show_initial_pressure=False,
               show_segmentation_map=False,
               log_scale=False)
