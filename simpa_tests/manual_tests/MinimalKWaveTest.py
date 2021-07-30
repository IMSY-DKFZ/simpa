"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.core.simulation import simulate
from simpa.utils import Tags, generate_dict_path, SaveFilePaths
from simpa.utils.settings import Settings
from simpa.utils.path_manager import PathManager
from simpa.core import AcousticForwardModelKWaveAdapter, ImageReconstructionModuleDelayAndSumAdapter
from simpa.core.device_digital_twins import *
from simpa.io_handling import save_hdf5, load_data_field
import numpy as np
import matplotlib.pyplot as plt
import os

path_manager = PathManager()

SPEED_OF_SOUND = 1470

# pa_device = PhotoacousticDevice(device_position_mm=np.asarray([10, 20, 50]))
# pa_device.set_detection_geometry(SingleDetectionElement(device_position_mm=np.asarray([10, 20, 50])))
pa_device = InVision256TF(device_position_mm=np.asarray([50, 15, 50]))

p0_path = path_manager.get_hdf5_file_save_path() + "/initial_pressure.npz"
if os.path.exists(p0_path):
    initial_pressure = np.load(p0_path)["initial_pressure"]
else:
    initial_pressure = np.zeros((100, 100, 100))
    initial_pressure[50, :, 50] = 1
speed_of_sound = np.ones((100, 30, 100)) * SPEED_OF_SOUND
density = np.ones((100, 30, 100)) * 1000
alpha = np.ones((100, 30, 100)) * 0.01


def get_settings():
    general_settings = {
        Tags.RANDOM_SEED: 4711,
        Tags.VOLUME_NAME: "SpeedOfSoundBug",
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: 1.0,
        Tags.DIM_VOLUME_Z_MM: 100,
        Tags.DIM_VOLUME_X_MM: 100,
        Tags.DIM_VOLUME_Y_MM: 30,
        Tags.GPU: True,
        Tags.WAVELENGTHS: [700]
    }

    acoustic_settings = {
        Tags.ACOUSTIC_SIMULATION_3D: True,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.PROPERTY_ALPHA_POWER: 0.00,
        Tags.SENSOR_RECORD: "p",
        Tags.PMLInside: False,
        Tags.PMLSize: [31, 32],
        Tags.PMLAlpha: 1.5,
        Tags.PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False
    }

    reconstruction_settings = {
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_HAMMING,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.PROPERTY_SPEED_OF_SOUND: SPEED_OF_SOUND,
        Tags.SPACING_MM: 0.1
    }

    _settings = Settings(general_settings)
    _settings.set_acoustic_settings(acoustic_settings)
    _settings.set_reconstruction_settings(reconstruction_settings)
    return _settings


settings = get_settings()
simulate([], settings, pa_device)
acoutsic_properties = {
    Tags.PROPERTY_SPEED_OF_SOUND: speed_of_sound,
    Tags.PROPERTY_DENSITY: density,
    Tags.PROPERTY_ALPHA_COEFF: alpha
}
save_hdf5(acoutsic_properties, settings[Tags.SIMPA_OUTPUT_PATH], SaveFilePaths.SIMULATION_PROPERTIES)
optical_output = {
    Tags.OPTICAL_MODEL_INITIAL_PRESSURE: {settings[Tags.WAVELENGTHS][0]: initial_pressure}
}
optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)
save_hdf5(optical_output, settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)
AcousticForwardModelKWaveAdapter(settings).run(pa_device)

time_series_data = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH],
                                   data_field=Tags.TIME_SERIES_DATA,
                                   wavelength=settings[Tags.WAVELENGTHS][0])

print(time_series_data)
print(np.shape(time_series_data))

plt.figure()
plt.plot(time_series_data.T)
plt.show()
plt.close()

ImageReconstructionModuleDelayAndSumAdapter(settings).run(pa_device)

reconstructed_image_1000 = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH],
                                          data_field=Tags.RECONSTRUCTED_DATA,
                                          wavelength=settings[Tags.WAVELENGTHS][0])

settings.get_reconstruction_settings()[Tags.PROPERTY_SPEED_OF_SOUND] = SPEED_OF_SOUND * 1.05
ImageReconstructionModuleDelayAndSumAdapter(settings).run(pa_device)

reconstructed_image_1050 = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH],
                                           data_field=Tags.RECONSTRUCTED_DATA,
                                           wavelength=settings[Tags.WAVELENGTHS][0])

settings.get_reconstruction_settings()[Tags.PROPERTY_SPEED_OF_SOUND] = SPEED_OF_SOUND * 0.95
ImageReconstructionModuleDelayAndSumAdapter(settings).run(pa_device)

reconstructed_image_950 = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH],
                                           data_field=Tags.RECONSTRUCTED_DATA,
                                           wavelength=settings[Tags.WAVELENGTHS][0])

plt.figure()
plt.subplot(1, 3, 1)
plt.title(f"{SPEED_OF_SOUND * 0.95} m/s")
plt.imshow(np.rot90(reconstructed_image_950, 3))
plt.subplot(1, 3, 2)
plt.title(f"{SPEED_OF_SOUND} m/s")
plt.imshow(np.rot90(reconstructed_image_1000, 3))
plt.subplot(1, 3, 3)
plt.title(f"{SPEED_OF_SOUND * 1.05} m/s")
plt.imshow(np.rot90(reconstructed_image_1050, 3))
plt.show()
plt.close()
