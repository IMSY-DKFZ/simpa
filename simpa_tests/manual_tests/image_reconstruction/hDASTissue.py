# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation import simulate
from simpa.utils import Tags, generate_dict_path
from simpa.utils.settings import Settings
from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.utils.path_manager import PathManager
from simpa import KWaveAdapter, DelayAndSumAdapter, visualise_device
from simpa.core.device_digital_twins import *
from simpa.io_handling import save_hdf5, load_data_field
import numpy as np
import matplotlib.pyplot as plt
import os
from simpa_tests.manual_tests import ManualIntegrationTestClass
from simpa.core.device_digital_twins import SlitIlluminationGeometry, LinearArrayDetectionGeometry, PhotoacousticDevice


# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import apply_b_mode
from simpa.utils import Tags

from simpa.io_handling import load_data_field
from simpa.core.simulation import simulate
from simpa import KWaveAdapter, MCXAdapter, \
    DelayAndSumAdapter, ModelBasedVolumeCreationAdapter, GaussianNoise
from simpa import reconstruct_delay_and_sum_pytorch
from simpa_tests.manual_tests import ManualIntegrationTestClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class hDASTissue(ManualIntegrationTestClass):
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm. To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """ 

    def setup(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        Please make sure that a valid path_config.env file is located in your home directory, or that you
        point to the correct file in the PathManager().
        :return:
        """
        DEVICE = "Acuity"

        self.path_manager = PathManager()
        self.VOLUME_TRANSDUCER_DIM_IN_MM = 75
        self.VOLUME_PLANAR_DIM_IN_MM = 20
        self.VOLUME_HEIGHT_IN_MM = 25
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711
        np.random.seed(self.RANDOM_SEED)
        if DEVICE == "Acuity":
            self.device = MSOTAcuityEcho(device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                  self.VOLUME_PLANAR_DIM_IN_MM/2, 0]))
        else:
            self.device = PhotoacousticDevice(
                device_position_mm=np.array([self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                             self.VOLUME_PLANAR_DIM_IN_MM/2, 0]),
                field_of_view_extent_mm=np.asarray([-self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     self.VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     0, 0, 0, 2*self.VOLUME_HEIGHT_IN_MM])
                )
            self.device.set_detection_geometry(LinearArrayDetectionGeometry(
                                                device_position_mm=self.device.device_position_mm,
                                                number_detector_elements=200,
                                                pitch_mm = 0.25,
                                                #field_of_view_extent_mm=self.device.field_of_view_extent_mm
                                                )
                                                )
            self.device.add_illumination_geometry(SlitIlluminationGeometry(slit_vector_mm=[50,0,0]))
        
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
            Tags.SIMULATE_DEFORMED_LAYERS: False,
            Tags.US_GEL: False
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

        """
        self.settings["noise_initial_pressure"] = {
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }"""

    # first volume i tried out
    def create_example_tissue_big_vessels(self):
        """
        1 circular 2 rectancular vessels
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

        """
        # add fat layer below skin
        fat_dictionary = Settings()
        fat_dictionary[Tags.PRIORITY] = 2
        fat_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        fat_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 2.75]
        # see https://www.metabolismjournal.com/article/0026-0495(89)90256-4/fulltext, 
        # https://www.sciencedirect.com/science/article/pii/0026049589902564 for thickness
        fat_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.subcutaneous_fat()
        fat_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        fat_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        fat_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE"""

        """# add epidermis
        epidermis_dictionary = Settings()
        epidermis_dictionary[Tags.PRIORITY] = 8
        epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 0.1]
        epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis()
        epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE"""

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
        #tissue_dict["fat"] = fat_dictionary
        #tissue_dict["epidermis"] = epidermis_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        tissue_dict["vessel_3"] = vessel_3_dictionary
        return tissue_dict

    # better for visualization
    def create_example_tissue(self):
        """
        4 circular shaped vessesl with increasing radius, no deformation
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.subcutaneous_fat()
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 20]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 50]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.SIMULATE_DEFORMED_LAYERS] = False
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        fat_dictionary = Settings()
        fat_dictionary[Tags.PRIORITY] = 1
        fat_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        fat_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 20]
        fat_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.subcutaneous_fat()
        fat_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        fat_dictionary[Tags.SIMULATE_DEFORMED_LAYERS] = False
        fat_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        fat_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 10,
                                                        0, 15]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 10,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.1
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_2_dictionary = Settings()
        vessel_2_dictionary[Tags.PRIORITY] = 3
        vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 2.5,
                                                        0, 15]
        vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 2.5,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.5
        vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_3_dictionary = Settings()
        vessel_3_dictionary[Tags.PRIORITY] = 3
        vessel_3_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 2.5,
                                                        0, 15]
        vessel_3_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 2.5,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_3_dictionary[Tags.STRUCTURE_RADIUS_MM] = 1.
        vessel_3_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_3_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_3_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        vessel_4_dictionary = Settings()
        vessel_4_dictionary[Tags.PRIORITY] = 3
        vessel_4_dictionary[Tags.STRUCTURE_START_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 10,
                                                        0, 15]
        vessel_4_dictionary[Tags.STRUCTURE_END_MM] = [self.VOLUME_TRANSDUCER_DIM_IN_MM / 2 - 10,
                                                      self.VOLUME_PLANAR_DIM_IN_MM, 15]
        vessel_4_dictionary[Tags.STRUCTURE_RADIUS_MM] = 1.5
        vessel_4_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        vessel_4_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_4_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["fat"] = fat_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        tissue_dict["vessel_3"] = vessel_3_dictionary
        tissue_dict["vessel_4"] = vessel_4_dictionary
        return tissue_dict


    def perform_test(self):

        self.device.update_settings_for_use_of_model_based_volume_creator(self.settings)

        SIMUATION_PIPELINE = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            #GaussianNoise(self.settings, "noise_initial_pressure"),
            KWaveAdapter(self.settings),
            DelayAndSumAdapter(self.settings)
        ]

        simulate(SIMUATION_PIPELINE, self.settings, self.device)

        self.initial_pressure = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_INITIAL_PRESSURE,
                                                self.settings[Tags.WAVELENGTH])

        self.reconstructed_image_homo = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                        self.settings[Tags.WAVELENGTH])

        self.homogenous_speed_of_sound = self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND]

        # Heterogenous Reconstruction
        self.speed_of_sound_map_hetero = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_SPEED_OF_SOUND)
        self.settings.get_reconstruction_settings()[Tags.DATA_FIELD_SPEED_OF_SOUND] = self.speed_of_sound_map_hetero
        self.settings[Tags.SOS_HETEROGENOUS] = True
        DelayAndSumAdapter(self.settings).run(self.device)

        self.reconstructed_image_hetero = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                          self.settings[Tags.WAVELENGTH])

    # overwrite function of ReconstructionAlgorithmTestBaseClass
    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        # visualize detector
        #visualise_device(self.pa_device)

        slice = 20

        fig, axes = plt.subplots(2, 2)#, figsize=(12,12))
        axes[0,0].set_title(f"Initial pressure")
        im = axes[0,0].imshow(np.rot90(self.initial_pressure[:, slice, :], 3), cmap="gnuplot")
        plt.colorbar(im, ax=axes[0,0])
   

        axes[0,1].set_title(f"""Hetero SoS map\nmean = {
            self.speed_of_sound_map_hetero[:,slice,:].mean():.1f}""")
        im = axes[0,1].imshow(np.rot90(self.speed_of_sound_map_hetero[:, slice, :], 3))
        plt.colorbar(im, ax=axes[0,1])

        axes[1,0].set_title(f"DAS Reconstruction\nwith SoS={self.homogenous_speed_of_sound}")
        im = axes[1,0].imshow(np.rot90(self.reconstructed_image_homo, 3), cmap="gnuplot")
        plt.colorbar(im, ax=axes[1,0])

        axes[1,1].set_title(f"hDAS Reconstruction")
        im = axes[1,1].imshow(np.rot90(self.reconstructed_image_hetero, 3), cmap="gnuplot")
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
            plt.savefig(save_path + f"minimal_pipeline_test_with_heterogenous_sos.png", dpi=300)
        plt.close()

        plt.figure()
        plt.title(f"hDAS - DAS")
        difference = np.rot90(self.reconstructed_image_hetero, 3)-np.rot90(self.reconstructed_image_homo, 3)
        max_difference = np.abs(difference).max()
        plt.imshow(difference, cmap = "seismic", vmin = -max_difference, vmax = max_difference)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + f"minimal_pipeline_test_with_heterogenous_sos_difference.png", dpi=300)
        plt.close()
    
    
    def test_convenience_function(self):

        #TODO ???
        pass

if __name__ == '__main__':
    test = hDASTissue()
    test.run_test(show_figure_on_screen=True)
