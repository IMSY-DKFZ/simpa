# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT


from simpa.core.device_digital_twins import SlitIlluminationGeometry, LinearArrayDetectionGeometry, PhotoacousticDevice
from simpa import perform_k_wave_acoustic_forward_simulation
from simpa.core.simulation_modules.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    reconstruct_delay_and_sum_pytorch
from simpa import MCXAdapter, ModelBasedVolumeCreationAdapter, \
    GaussianNoise
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.io_handling import load_data_field
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa_tests.manual_tests import ManualIntegrationTestClass
import matplotlib.pyplot as plt

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class KWaveAcousticForwardConvenienceFunction(ManualIntegrationTestClass):
    """
    This class test the convenience function for acoustic forward simulation.
    It first creates a volume and runs an optical forward simulation. 
    Then the function is actually tested.
    Lastly the generated time series data is reconstructed to compare whether everything worked.
    """

    def setup(self):
        """
        Runs a pipeline consisting of volume creation and optical simulation. The resulting hdf5 file of the
        simple test volume is saved at SAVE_PATH location defined in the path_config.env file.
        """

        self.path_manager = PathManager()
        self.VOLUME_TRANSDUCER_DIM_IN_MM = 75
        self.VOLUME_PLANAR_DIM_IN_MM = 20
        self.VOLUME_HEIGHT_IN_MM = 25
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711
        self.VOLUME_NAME = "TestKWaveAcousticForwardConvenienceFunction_" + str(self.RANDOM_SEED)

        np.random.seed(self.RANDOM_SEED)

        # These parameters set the general properties of the simulated volume
        self.general_settings = {
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: self.VOLUME_NAME,
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.WAVELENGTHS: [700]
        }
        self.settings = Settings(self.general_settings)

        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: True,
            Tags.STRUCTURES: self.create_example_tissue()
        })
        self.settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: self.path_manager.get_mcx_binary_path(),
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            Tags.MCX_ASSUMED_ANISOTROPY: 0.9
        })
        self.settings["noise_model"] = {
            Tags.NOISE_MEAN: 0.0,
            Tags.NOISE_STD: 0.4,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
            Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }

        self.device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                       self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                       0]))
        self.device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=
                                                                        self.device.device_position_mm, pitch_mm=0.25,
                                                                        number_detector_elements=200))
        self.device.add_illumination_geometry(SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

        # run pipeline including volume creation and optical mcx simulation
        self.pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            GaussianNoise(self.settings, "noise_model")
        ]

    def teardown(self):
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def perform_test(self):
        simulate(self.pipeline, self.settings, self.device)
        self.test_convenience_function()

    def test_convenience_function(self):

        # load initial pressure
        initial_pressure = load_data_field(self.path_manager.get_hdf5_file_save_path() + "/" +
                                           self.VOLUME_NAME + ".hdf5",
                                           Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength=700)
        image_slice = np.s_[:, 40, :]
        self.initial_pressure = np.rot90(initial_pressure[image_slice], -1)

        # define acoustic settings and run simulation with convenience function
        acoustic_settings = {
            Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
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
        time_series_data = perform_k_wave_acoustic_forward_simulation(initial_pressure=self.initial_pressure,
                                                                      detection_geometry=self.device.
                                                                      get_detection_geometry(),
                                                                      speed_of_sound=1540, density=1000,
                                                                      alpha_coeff=0.0)

        # reconstruct the time series data to compare it with initial pressure
        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
            Tags.RECONSTRUCTION_BMODE_BEFORE_RECONSTRUCTION: True,
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
            Tags.SPACING_MM: 0.25,
            Tags.SENSOR_SAMPLING_RATE_MHZ: 40,
        })

        self.reconstructed = reconstruct_delay_and_sum_pytorch(
            time_series_data.copy(), self.device.get_detection_geometry(), self.settings)

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        '''plot initial pressure and reconstructed image volume to manually compare'''
        plt.subplot(2, 2, 1)
        plt.title("Initial Pressure Pipeline")
        plt.imshow(self.initial_pressure)
        plt.subplot(2, 2, 2)
        plt.title("Reconstructed Image Pipeline")
        plt.imshow(np.rot90(self.reconstructed, -1))
        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"TestKWaveConvenienceFunction.png")
        plt.close()

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.05, 100, 0.9)
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                        0, 10]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [
            self.VOLUME_TRANSDUCER_DIM_IN_MM/2, self.VOLUME_PLANAR_DIM_IN_MM, 10]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_2_dictionary = Settings()
        vessel_2_dictionary[Tags.PRIORITY] = 3
        vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10,
                                                        0, 5]
        vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [
            self.VOLUME_TRANSDUCER_DIM_IN_MM/2 - 10, self.VOLUME_PLANAR_DIM_IN_MM, 5]
        vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
        vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_2_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        epidermis_dictionary = Settings()
        epidermis_dictionary[Tags.PRIORITY] = 8
        epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 1]
        epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 1.1]
        epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
        epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["epidermis"] = epidermis_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        return tissue_dict


if __name__ == '__main__':
    test = KWaveAcousticForwardConvenienceFunction()
    test.run_test(show_figure_on_screen=False)
