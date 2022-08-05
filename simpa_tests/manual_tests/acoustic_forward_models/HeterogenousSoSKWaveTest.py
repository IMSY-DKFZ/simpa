# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation import simulate
from simpa.utils import Tags, generate_dict_path
from simpa.utils.settings import Settings
from simpa.utils.path_manager import PathManager
from simpa import KWaveAdapter, DelayAndSumAdapter, visualise_device
from simpa.core.device_digital_twins import *
from simpa.io_handling import save_hdf5, load_data_field
import numpy as np
import matplotlib.pyplot as plt
import os
from simpa_tests.manual_tests import ManualIntegrationTestClass


class HeterogenousSoSKWaveTest(ManualIntegrationTestClass):

    def setup(self):
        path_manager = PathManager()

        self.SPEED_OF_SOUND = 1470
        self.pa_device = InVision256TF(device_position_mm=np.asarray([50, 15, 50]))

        """
        self.initial_pressure = np.zeros((20, 6, 30))
        self.initial_pressure[8:13, :, 14:17] = 1
        self.speed_of_sound_map = np.ones((20, 6, 30)) * self.SPEED_OF_SOUND
        # idea: forward model with hetero sos map, reconstruct with homogenous and hetero sos map and compare results
        self.speed_of_sound_map_hetero = np.ones((20, 6, 30)) * self.SPEED_OF_SOUND
        self.speed_of_sound_map_hetero[10:,:,:] = 4*self.SPEED_OF_SOUND
        self.density = np.ones((20, 6, 30)) * 1000
        self.alpha = np.ones((20, 6, 30)) * 0.01
        """

        # Hyperparam to change:
        self.SIMULATE_HETERO = True


        self.initial_pressure = np.zeros((100, 30, 100))
        self.initial_pressure[50, :, 50] = 1
        self.speed_of_sound_map = np.ones((100, 30, 100)) * self.SPEED_OF_SOUND
        self.speed_of_sound_map_hetero = self.speed_of_sound_map.copy()
        self.speed_of_sound_map_hetero = self.speed_of_sound_map_hetero * np.linspace(0.5,1.5,100)[None, None, :] 
        self.density = np.ones((100, 30, 100)) * 1000
        self.alpha = np.ones((100, 30, 100)) * 0.01

        

        def get_settings():
            general_settings = {
                Tags.RANDOM_SEED: 4711,
                Tags.VOLUME_NAME: "HeteroggenousSpeedOfSoundBug",
                Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
                Tags.SPACING_MM: 1.0,
                Tags.DIM_VOLUME_Z_MM: 100,
                Tags.DIM_VOLUME_X_MM: 100,
                Tags.DIM_VOLUME_Y_MM: 30,
                Tags.GPU: True,
                Tags.WAVELENGTHS: [700],
                Tags.SOS_HETEROGENOUS: True
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
                Tags.DATA_FIELD_SPEED_OF_SOUND:  # Heterogenous SoS if simulated heterogenously
                self.speed_of_sound_map_hetero if self.SIMULATE_HETERO else self.speed_of_sound_map,
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
            Tags.DATA_FIELD_SPEED_OF_SOUND: # Heterogenous map for foward modeling
            self.speed_of_sound_map_hetero if self.SIMULATE_HETERO else self.speed_of_sound_map, 
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

        # Heterogenous reconstruction
        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_hetero = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        wavelength=self.settings[Tags.WAVELENGTHS][0])


        # Homogenous reconstruction
        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.SPEED_OF_SOUND
        self.settings.get_reconstruction_settings()[Tags.SOS_HETEROGENOUS] = False
        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_homo = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        wavelength=self.settings[Tags.WAVELENGTHS][0])


    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        # visualize detector
        #visualise_device(self.pa_device)

        slice = self.speed_of_sound_map.shape[1]//2

    
        plt.figure(figsize=(18, 6))
        plt.subplot(3, 2, 1)
        plt.title(f"Hom. SoS map")
        plt.imshow(self.speed_of_sound_map[:,slice,:].T)
        plt.colorbar()
        plt.subplot(3, 2, 2)
        plt.title(f"Hetero SoS map\nmean = {self.speed_of_sound_map_hetero[:,slice,:].mean():.1f}")
        plt.imshow(self.speed_of_sound_map_hetero[:,slice,:].T)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.colorbar()
        plt.tight_layout()
        plt.subplot(3, 2, 3)
        plt.title(f"Set Initial pressure")
        plt.imshow(self.initial_pressure[:,self.initial_pressure.shape[1]//2,:].T)
        plt.colorbar()
        plt.subplot(3, 2, 5)
        plt.title(f"Reconstructed image\n with homogenous SoS {self.SPEED_OF_SOUND} m/s")
        try:
            plt.imshow(np.rot90(self.reconstructed_image_homo, 3))
        except:
            print("no image reconstructed")
        plt.subplot(3, 2, 6)
        plt.title(f"Reconstructed image\n with heterogenous SoS")
        try:
            plt.imshow(np.rot90(self.reconstructed_image_hetero, 3))
        except:
            print("no image reconstructed")
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"minimal_kwave_test_with_heterogenous_sos.png")
        plt.close()

if __name__ == "__main__":
    test = HeterogenousSoSKWaveTest()
    test.run_test(show_figure_on_screen=True)
