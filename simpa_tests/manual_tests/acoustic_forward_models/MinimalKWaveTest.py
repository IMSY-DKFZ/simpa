# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation import simulate
from simpa.utils import Tags, generate_dict_path
from simpa.utils.settings import Settings
from simpa.utils.path_manager import PathManager
from simpa import KWaveAdapter, DelayAndSumAdapter
from simpa.core.device_digital_twins import *
from simpa.io_handling import save_hdf5, load_data_field
import numpy as np
import matplotlib.pyplot as plt
import os
from simpa_tests.manual_tests import ManualIntegrationTestClass


class MinimalKWaveTest(ManualIntegrationTestClass):

    def setup(self):
        path_manager = PathManager()

        self.SPEED_OF_SOUND = 1470
        self.pa_device = InVision256TF(device_position_mm=np.asarray([50, 15, 50]))

        p0_path = path_manager.get_hdf5_file_save_path() + "/initial_pressure.npz"
        if os.path.exists(p0_path):
            self.initial_pressure = np.load(p0_path)["initial_pressure"]
        else:
            self.initial_pressure = np.zeros((100, 100, 100))
            self.initial_pressure[50, :, 50] = 1
        self.speed_of_sound = np.ones((100, 30, 100)) * self.SPEED_OF_SOUND
        self.density = np.ones((100, 30, 100)) * 1000
        self.alpha = np.ones((100, 30, 100)) * 0.01


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
                Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
                Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
                Tags.KWAVE_PROPERTY_PMLInside: False,
                Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
                Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
                Tags.KWAVE_PROPERTY_PlotPML: False,
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
                Tags.DATA_FIELD_SPEED_OF_SOUND: self.SPEED_OF_SOUND,
                Tags.SPACING_MM: 0.1
            }

            _settings = Settings(general_settings)
            _settings.set_acoustic_settings(acoustic_settings)
            _settings.set_reconstruction_settings(reconstruction_settings)
            _settings.set_volume_creation_settings({})
            return _settings

        self.settings = get_settings()

    def perform_test(self):
        simulate([], self.settings, self.pa_device)

        acoustic_properties = {
            Tags.DATA_FIELD_SPEED_OF_SOUND: self.speed_of_sound,
            Tags.DATA_FIELD_DENSITY: self.density,
            Tags.DATA_FIELD_ALPHA_COEFF: self.alpha
        }
        save_file_path = generate_dict_path(Tags.SIMULATION_PROPERTIES)
        save_hdf5(acoustic_properties, self.settings[Tags.SIMPA_OUTPUT_PATH], save_file_path)
        optical_output = {
            Tags.DATA_FIELD_INITIAL_PRESSURE: {self.settings[Tags.WAVELENGTHS][0]: self.initial_pressure}
        }
        optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)
        save_hdf5(optical_output, self.settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)
        KWaveAdapter(self.settings).run(self.pa_device)

        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_1000 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        wavelength=self.settings[Tags.WAVELENGTHS][0])

        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.SPEED_OF_SOUND * 1.025
        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_1050 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        wavelength=self.settings[Tags.WAVELENGTHS][0])

        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.SPEED_OF_SOUND * 0.975
        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_950 = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                       data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                       wavelength=self.settings[Tags.WAVELENGTHS][0])

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.title(f"{self.SPEED_OF_SOUND * 0.975} m/s")
        plt.imshow(np.rot90(self.reconstructed_image_950, 3))
        plt.subplot(1, 3, 2)
        plt.title(f"{self.SPEED_OF_SOUND} m/s")
        plt.imshow(np.rot90(self.reconstructed_image_1000, 3))
        plt.subplot(1, 3, 3)
        plt.title(f"{self.SPEED_OF_SOUND * 1.025} m/s")
        plt.imshow(np.rot90(self.reconstructed_image_1050, 3))
        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"minimal_kwave_test.png")
        plt.close()

if __name__ == "__main__":
    test = MinimalKWaveTest()
    test.run_test(show_figure_on_screen=False)