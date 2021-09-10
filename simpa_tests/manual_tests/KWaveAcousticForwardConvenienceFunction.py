"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""


from simpa.core.device_digital_twins import SlitIlluminationGeometry, LinearArrayDetectionGeometry, PhotoacousticDevice
from simpa.core.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import perform_k_wave_acoustic_forward_simulation
from simpa.core.reconstruction_module.reconstruction_module_delay_and_sum_adapter import reconstruct_delay_and_sum_pytorch
from simpa.core import OpticalForwardModelMcxAdapter, VolumeCreationModelModelBasedAdapter, \
    GaussianNoiseProcessingComponent
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.io_handling import load_data_field
import numpy as np
from simpa.utils.path_manager import PathManager

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class KWaveAcousticForwardConvenienceFunction:
    """
    This class test the convenience function for acoustic forward simulation.
    It first creates a volume and runs an optical forward simulation. 
    Then the function is actually tested.
    Lastly the generated time series data is reconstructed to compare whether everything worked.
    """

    def create_volume_and_run_optical_simulation(self):
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
            Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }

        self.device = PhotoacousticDevice(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                       self.VOLUME_PLANAR_DIM_IN_MM/2,
                                                                       0]))
        self.device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=self.device.device_position_mm, pitch_mm=0.25,
                                                                        number_detector_elements=200))
        self.device.add_illumination_geometry(SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

        # run pipeline including volume creation and optical mcx simulation
        pipeline = [
            VolumeCreationModelModelBasedAdapter(self.settings),
            OpticalForwardModelMcxAdapter(self.settings),
            GaussianNoiseProcessingComponent(self.settings, "noise_model")
        ]
        simulate(pipeline, self.settings, self.device)

    def test_convenience_function(self):

        # load initial pressure
        initial_pressure = load_data_field(self.path_manager.get_hdf5_file_save_path() + "/" + self.VOLUME_NAME + ".hdf5",
                                           Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=700)
        image_slice = np.s_[:, 40, :]
        #initial_pressure = np.flip(np.rot90(initial_pressure[image_slice], axes=(0, 1)))
        initial_pressure = initial_pressure[image_slice].T

        # define acoustic settings and run simulation with convenience function
        acoustic_settings = {
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.PROPERTY_ALPHA_POWER: 0.0,
            Tags.SENSOR_RECORD: "p",
            Tags.PMLInside: False,
            Tags.PMLAlpha: 1.5,
            Tags.PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True,
            Tags.GPU: True,
            Tags.SPACING_MM: self.SPACING,
        }
        time_series_data = perform_k_wave_acoustic_forward_simulation(initial_pressure,
                                                                      self.device.get_detection_geometry(), speed_of_sound=1540, density=1000,
                                                                      alpha_coeff=0.0, acoustic_settings=acoustic_settings)

        # reconstruct the time series data to compare it with initial pressure
        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
            Tags.RECONSTRUCTION_BMODE_BEFORE_RECONSTRUCTION: True,
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.PROPERTY_SPEED_OF_SOUND: 1540,
            Tags.SPACING_MM: 0.25,
            Tags.SENSOR_SAMPLING_RATE_MHZ: 40,
        })

        reconstructed = reconstruct_delay_and_sum_pytorch(
            time_series_data.copy(), self.device.get_detection_geometry(), self.settings)

        self.visualize_results(initial_pressure, reconstructed)

    def visualize_results(self, initial_pressure, reconstructed):
        '''plot initial pressure and reconstructed image volume to manually compare'''
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.title("Initial Pressure")
        plt.imshow(initial_pressure)
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(np.rot90(reconstructed, -1))
        plt.show()

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
    test.create_volume_and_run_optical_simulation()
    test.test_convenience_function()
