# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import os
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.io_handling import load_data_field
from simpa.core.device_digital_twins import MSOTAcuityEcho
import matplotlib.pyplot as plt
from abc import abstractmethod


class ManualIntegrationTestClass(object):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def perform_test(self):
        pass

    @abstractmethod
    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        pass

    @abstractmethod
    def tear_down(self):
        pass

    def run_test(self, show_figure_on_screen=True, save_path=None):
        if save_path is None or not os.path.isdir(save_path):
            save_path = "figures/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.setup()
        self.perform_test()
        self.visualise_result(show_figure_on_screen, save_path)
        self.tear_down()


class ReconstructionAlgorithmTestBaseClass(ManualIntegrationTestClass):

    def __init__(self):
        self.reconstructed_image_pipeline = None
        self.reconstructed_image_convenience = None



    @abstractmethod
    def test_reconstruction_of_simulation(self):
        pass

    @abstractmethod
    def test_convenience_function(self):
        pass

    def setup(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        Please make sure that a valid path_config.env file is located in your home directory, or that you
        point to the correct file in the PathManager().
        :return:
        """

        self.path_manager = PathManager()
        self.VOLUME_TRANSDUCER_DIM_IN_MM = 75
        self.VOLUME_PLANAR_DIM_IN_MM = 20
        self.VOLUME_HEIGHT_IN_MM = 25
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711
        np.random.seed(self.RANDOM_SEED)
        self.device = MSOTAcuityEcho(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                  self.VOLUME_PLANAR_DIM_IN_MM/2, 0]))
        self.general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestImageReconstruction_" + str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,

            # Simulation Device
            Tags.DIGITAL_DEVICE: Tags.DIGITAL_DEVICE_MSOT_ACUITY,

            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [700]
        }
        self.settings = Settings(self.general_settings)

        self.settings.set_volume_creation_settings({
            Tags.STRUCTURES: self.create_example_tissue(),
            Tags.SIMULATE_DEFORMED_LAYERS: True
        })

        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        })

        self.settings.set_acoustic_settings({
            Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.GPU: True,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.05,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True
        })

        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
            Tags.TUKEY_WINDOW_ALPHA: 0.5,
            Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e6),
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
            Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
            Tags.SPACING_MM: self.settings[Tags.SPACING_MM],
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.ACOUSTIC_SIMULATION_3D: False,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True,
            Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
            Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
            Tags.DATA_FIELD_DENSITY: 1000,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
        })

        self.settings["noise_initial_pressure"] = {
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                        0, 10]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 10]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_2_dictionary = Settings()
        vessel_2_dictionary[Tags.PRIORITY] = 3
        vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 12,
                                                        0, 7]
        vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 10,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 5]
        vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
        vessel_2_dictionary[Tags.STRUCTURE_ECCENTRICITY] = 0.9
        vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.ELLIPTICAL_TUBULAR_STRUCTURE

        vessel_3_dictionary = Settings()
        vessel_3_dictionary[Tags.PRIORITY] = 3
        vessel_3_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 12, 0, 3]
        vessel_3_dictionary[Tags.STRUCTURE_X_EXTENT_MM] = 8
        vessel_3_dictionary[Tags.STRUCTURE_Y_EXTENT_MM] = 10
        vessel_3_dictionary[Tags.STRUCTURE_Z_EXTENT_MM] = 16
        vessel_3_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_3_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_3_dictionary[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        tissue_dict["vessel_3"] = vessel_3_dictionary
        return tissue_dict

    def perform_test(self):
        self.test_reconstruction_of_simulation()
        self.test_convenience_function()

    def tear_down(self):
        pass

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        initial_pressure = load_data_field(
            self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_INITIAL_PRESSURE,
            wavelength=self.settings[Tags.WAVELENGTH])

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.title("Initial pressure")
        plt.imshow(np.rot90(initial_pressure[:, 20, :], 3))
        plt.subplot(1, 3, 2)
        plt.title("Pipeline\nReconstruction")
        if self.reconstructed_image_pipeline is not None:
            plt.imshow(np.rot90(self.reconstructed_image_pipeline, 3))
        plt.subplot(1, 3, 3)
        plt.title("Convenience Method\nReconstruction")
        if self.reconstructed_image_convenience is not None:
            plt.imshow(np.rot90(self.reconstructed_image_convenience, 3))

        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"reconstrution_test_{self.__class__.__name__}.png")
        plt.close()
