# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = sp.PathManager()
PATH = path_manager.get_hdf5_file_save_path() + "/CompletePipelineExample_4711.hdf5"

file = sp.load_hdf5(PATH)
settings = sp.Settings(file["settings"])
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
device = file["digital_device"]

sp.DelayAndSumAdapter(settings).run(device)

reconstructed_image = sp.load_data_field(PATH, Tags.DATA_FIELD_RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])
reconstructed_image = np.squeeze(reconstructed_image)

sp.visualise_data(path_to_hdf5_file=PATH,
                  wavelength=settings[Tags.WAVELENGTH],
                  show_reconstructed_data=True,
                  show_xz_only=True)
