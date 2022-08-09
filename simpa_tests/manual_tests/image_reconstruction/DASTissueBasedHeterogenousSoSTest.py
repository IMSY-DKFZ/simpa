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
from simpa_tests.manual_tests import ReconstructionAlgorithmTestBaseClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DASTissueBasedHeterogenousSoSTest(ReconstructionAlgorithmTestBaseClass):
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm. To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """

    def test_reconstruction_of_simulation(self):

        self.device.update_settings_for_use_of_model_based_volume_creator(self.settings)

        # Simulation and Homogenous Reconstruction

        SIMUATION_PIPELINE = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            GaussianNoise(self.settings, "noise_initial_pressure"),
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
    

    # overwrite tissue creation function
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
        fat_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

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
        tissue_dict["fat"] = fat_dictionary
        #tissue_dict["epidermis"] = epidermis_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        tissue_dict["vessel_2"] = vessel_2_dictionary
        tissue_dict["vessel_3"] = vessel_3_dictionary
        return tissue_dict


    # overwrite function of ReconstructionAlgorithmTestBaseClass
    def visualise_result(self, show_figure_on_screen=True, save_path=None):

        # visualize detector
        #visualise_device(self.pa_device)

        print("\n\n CHRISTOPH:\n", "p0 shape:", self.initial_pressure.shape, "\nsos shape:", self.speed_of_sound_map_hetero.shape)
        print("assumend homogenous sos:", self.homogenous_speed_of_sound)


        slice = 20

        fig, axes = plt.subplots(2, 2)#, figsize=(12,12))
        axes[0,0].set_title(f"Initial pressure")
        im = axes[0,0].imshow(np.rot90(self.initial_pressure[:, slice, :], 3))
        plt.colorbar(im, ax=axes[0,0])
   

        axes[0,1].set_title(f"""Hetero SoS map\nmean = {
            self.speed_of_sound_map_hetero[:,slice,:].mean():.1f}""")
        im = axes[0,1].imshow(np.rot90(self.speed_of_sound_map_hetero[:, slice, :], 3))
        plt.colorbar(im, ax=axes[0,1])

        axes[1,0].set_title(f"DAS Reconstruction\nwith SoS={self.homogenous_speed_of_sound}")
        im = axes[1,0].imshow(np.rot90(self.reconstructed_image_homo, 3))
        plt.colorbar(im, ax=axes[1,0])

        axes[1,1].set_title(f"hDAS Reconstruction")
        im = axes[1,1].imshow(np.rot90(self.reconstructed_image_hetero, 3))
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
    test = DASTissueBasedHeterogenousSoSTest()
    test.run_test(show_figure_on_screen=False)
