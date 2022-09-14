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


class hDASToySoS(ManualIntegrationTestClass):
    """
    Time series data are simulated using KWave and a heterogenous SoS-map and a initital pressure with just 
    1 non-zero entry (point source).
    DAS-Reconstruction assuming a fixed speed-of-sound-value (homogenous SoS-map) and hDAS-reconstruction considering 
    the correct heterogenous SoS-map are performed. 
    For this, one can choose between predefined "toy examples of SoS-maps" with gradients, steps or noise:
        - vertical gradient
        - horizontal gradient
        - 2 horizontal regions
        - 2 diagonal regions
        - gaussian noise
    The two outcoming reconstruction can be compared in one image. Whereby, in the hDAS reconstruction the signal
    maximum should remain in the center of the image, while in the DAS reconstruction the signal maximum off.
    """

    def setup(self):
        ###########  Hyperparam to change: ##############
        self.HETERO_OPTION = "vertical_gradient"
        #################################################

        path_manager = PathManager()

        self.SPEED_OF_SOUND = 1470
        self.SIMULATE_HETERO = True
        self.pa_device = InVision256TF(device_position_mm=np.asarray([50, 15, 50]))

        self.initial_pressure = np.zeros((100, 30, 100))
        self.initial_pressure[50, :, 50] = 1
        self.speed_of_sound_map = np.ones((100, 30, 100)) * self.SPEED_OF_SOUND
        np.random.seed(4711)
        heterogenous_map = {"vertical_gradient": self.speed_of_sound_map * np.linspace(0.5,1.5,100)[None, None, :],
                            "horizontal_gradient": self.speed_of_sound_map * np.linspace(0.5,1.5,100)[:, None, None],
                            "horizontal_2_regions": self.speed_of_sound_map * np.hstack((np.ones(50)*0.5, np.ones(50)*1.5))[:, None, None],
                            "diagonal_2_regions": self.speed_of_sound_map * 0.5 + np.triu(self.speed_of_sound_map[:,14,:])[:,None,:],
                            "gaussian_noise": self.speed_of_sound_map + np.random.randn(100, 30, 100) * 142}
        
        self.speed_of_sound_map_hetero = heterogenous_map[self.HETERO_OPTION]
        # scale to mean = self.SPEED_OF_SOUND
        self.speed_of_sound_map_hetero *= self.SPEED_OF_SOUND/self.speed_of_sound_map_hetero.mean()

        self.density = np.ones((100, 30, 100)) * 1000
        self.alpha = np.ones((100, 30, 100)) * 0.01
        

        def get_settings():
            general_settings = {
                Tags.RANDOM_SEED: 4711,
                Tags.VOLUME_NAME: "HeterogenousSpeedOfSoundToyTest",
                Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
                Tags.SPACING_MM: 1.0,
                Tags.DIM_VOLUME_Z_MM: 100,
                Tags.DIM_VOLUME_X_MM: 100,
                Tags.DIM_VOLUME_Y_MM: 30,
                Tags.GPU: True,
                Tags.WAVELENGTHS: [700],
                Tags.SOS_HETEROGENOUS: True # in order to reconstruct using the heterogen. map
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


        # Homogenous reconstruction (for this reset the settings for homogenous reconstruction)
        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.SPEED_OF_SOUND
        self.settings[Tags.SOS_HETEROGENOUS] = False
        DelayAndSumAdapter(self.settings).run(self.pa_device)

        self.reconstructed_image_homo = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        wavelength=self.settings[Tags.WAVELENGTHS][0])


    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        # visualize detector
        #visualise_device(self.pa_device)

        slice = self.speed_of_sound_map.shape[1]//2

        fig, axes = plt.subplots(2, 2)#, figsize=(12,12))
        axes[0,0].set_title(f"Initial pressure")
        im = axes[0,0].imshow(self.initial_pressure[:, slice, :].T)
        plt.colorbar(im, ax=axes[0,0])
   

        axes[0,1].set_title(f"""Hetero SoS map\nmean = {
            self.speed_of_sound_map_hetero[:,slice,:].mean():.1f}""")
        im = axes[0,1].imshow(self.speed_of_sound_map_hetero[:,slice,:].T)
        plt.colorbar(im, ax=axes[0,1])

        axes[1,0].set_title(f"DAS Reconstruction\nwith SoS={self.SPEED_OF_SOUND}")
        im = axes[1,0].imshow(self.reconstructed_image_homo.T)
        plt.colorbar(im, ax=axes[1,0])

        axes[1,1].set_title(f"hDAS Reconstruction")
        im = axes[1,1].imshow(self.reconstructed_image_hetero.T)
        plt.colorbar(im, ax=axes[1,1])

        plt.tight_layout()

        for ax in axes.flatten():
            ax.set_xlabel("x")
            ax.set_ylabel("z")

        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"minimal_kwave_test_with_heterogenous_sos_{self.HETERO_OPTION}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    test = hDASToySoS()
    test.run_test(show_figure_on_screen=True)
