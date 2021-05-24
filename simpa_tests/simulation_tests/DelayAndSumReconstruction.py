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
from simpa.utils import Tags, TISSUE_LIBRARY
import numpy as np
import matplotlib.pyplot as plt
from simpa.utils.settings import Settings
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils.path_manager import PathManager
from simpa.io_handling import load_data_field, load_hdf5
from simpa.core.simulation import simulate
from simpa.core.device_digital_twins.devices.pa_devices.ithera_msot_acuity import MSOTAcuityEcho
from simpa.core.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import AcousticForwardModelKWaveAdapter
from simpa.core.optical_simulation_module.optical_forward_model_mcx_adapter import OpticalForwardModelMcxAdapter
from simpa.core.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    ImageReconstructionModuleDelayAndSumAdapter
from simpa.core.volume_creation_module.volume_creation_module_model_based_adapter import \
    VolumeCreationModelModelBasedAdapter
from simpa.processing.noise_processing_components import GaussianNoiseProcessingComponent


class DelayAndSumReconstruction:
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm. To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """

    def setUp(self):
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
        self.SPACING = 0.3  # 15
        self.RANDOM_SEED = 4711
        np.random.seed(self.RANDOM_SEED)

        self.general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "CompletePipelineTestMSOT_" + str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,

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
            Tags.PROPERTY_ALPHA_POWER: 1.05,
            Tags.SENSOR_RECORD: "p",
            Tags.PMLInside: False,
            Tags.PMLSize: [31, 32],
            Tags.PMLAlpha: 1.5,
            Tags.PlotPML: False,
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
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
        })

        self.settings["noise_initial_pressure"] = {
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }


    def test_reconstruction_of_simulation(self):

        self.add_msot_specific_settings()

        SIMUATION_PIPELINE = [
            VolumeCreationModelModelBasedAdapter(self.settings),
            OpticalForwardModelMcxAdapter(self.settings),
            GaussianNoiseProcessingComponent(self.settings, "noise_initial_pressure"),
            AcousticForwardModelKWaveAdapter(self.settings),
            ImageReconstructionModuleDelayAndSumAdapter(self.settings)
        ]

        simulate(SIMUATION_PIPELINE, self.settings)


        reconstructed_image_path = generate_dict_path(
            Tags.RECONSTRUCTED_DATA,
            wavelength=self.settings[Tags.WAVELENGTH])

        reconstructed_image = load_hdf5(
            self.settings[Tags.SIMPA_OUTPUT_PATH],
            reconstructed_image_path)[Tags.RECONSTRUCTED_DATA]

        initial_pressure = load_data_field(
            self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.OPTICAL_MODEL_INITIAL_PRESSURE, wavelength=self.settings[Tags.WAVELENGTH])

        plt.subplot(1, 2, 1)
        plt.title("Initial pressure")
        plt.imshow(np.flipud(np.rot90(initial_pressure[:, 20, :])))
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed image")
        plt.imshow(np.rot90(reconstructed_image, -1))
        plt.show()

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

    def add_msot_specific_settings(self):
        volume_creator_settings = Settings(self.settings.get_volume_creation_settings())
        device = MSOTAcuityEcho()
        probe_size_mm = device.probe_height_mm
        mediprene_layer_height_mm = device.mediprene_membrane_height_mm
        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm

        new_volume_height_mm = self.settings[Tags.DIM_VOLUME_Z_MM] + mediprene_layer_height_mm + \
                               heavy_water_layer_height_mm

        # adjust the z-dim to msot probe height
        self.settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        # adjust the x-dim to msot probe width
        # 1 mm is added (0.5 mm on both sides) to make sure no rounding errors lead to a detector element being outside
        # of the simulated volume.

        if self.settings[Tags.DIM_VOLUME_X_MM] < round(device.probe_width_mm) + 1:
            width_shift_for_structures_mm = (round(device.probe_width_mm) + 1 - self.settings[Tags.DIM_VOLUME_X_MM]) / 2
            self.settings[Tags.DIM_VOLUME_X_MM] = round(device.probe_width_mm) + 1
            device.logger.debug(f"Changed Tags.DIM_VOLUME_X_MM to {self.settings[Tags.DIM_VOLUME_X_MM]}")
        else:
            width_shift_for_structures_mm = 0

        device.logger.debug(volume_creator_settings)

        for structure_key in volume_creator_settings[Tags.STRUCTURES]:
            device.logger.debug("Adjusting " + str(structure_key))
            structure_dict = volume_creator_settings[Tags.STRUCTURES][structure_key]
            if Tags.STRUCTURE_START_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_START_MM][0] = structure_dict[Tags.STRUCTURE_START_MM][
                                                                 0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_START_MM][2] = structure_dict[Tags.STRUCTURE_START_MM][
                                                                 2] + device.probe_height_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][0] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][
                                                               2] + device.probe_height_mm

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_thickness = np.random.normal(0.4, 0.1)
            us_gel_layer_settings = Settings({
                Tags.PRIORITY: 5,
                Tags.STRUCTURE_START_MM: [0, 0,
                                          heavy_water_layer_height_mm - us_gel_thickness + mediprene_layer_height_mm],
                Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
                Tags.CONSIDER_PARTIAL_VOLUME: True,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.ultrasound_gel(),
                Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
            })

            volume_creator_settings[Tags.STRUCTURES]["us_gel"] = us_gel_layer_settings
        else:
            us_gel_thickness = 0

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 5,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm - us_gel_thickness],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm - us_gel_thickness + mediprene_layer_height_mm],
            Tags.CONSIDER_PARTIAL_VOLUME: True,
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene(),
            Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
        })

        volume_creator_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer_settings

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.heavy_water(),
            Tags.STRUCTURE_TYPE: Tags.BACKGROUND
        })
        volume_creator_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background_settings


if __name__ == '__main__':
    test = DelayAndSumReconstruction()
    test.setUp()
    test.test_reconstruction_of_simulation()
