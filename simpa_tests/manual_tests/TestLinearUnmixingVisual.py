"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, TISSUE_LIBRARY, Settings
from simpa.core.simulation import simulate
from simpa.core.processing_components.multispectral import linear_unmixing as lu
import numpy as np
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter
from simpa.utils.path_manager import PathManager
from simpa.io_handling import load_data_field
import matplotlib.pyplot as plt
from simpa.log import Logger
import os
from simpa.core.device_digital_twins import PencilBeamIlluminationGeometry


class TestLinearUnmixingVisual:
    """
    This test is a manual test, so visual confirmation is needed.
    """

    def setup(self):
        """
        This function lays the foundation for the manual test.
        """

        self.logger = Logger()
        # TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
        #  point to the correct file in the PathManager().
        self.path_manager = PathManager()

        RANDOM_SEED = 471
        self.VISUAL_WAVELENGTHS = [750, 800, 850]  # the performance is checked using three wavelengths

        np.random.seed(RANDOM_SEED)

        # Initialize global settings and prepare for simulation pipeline including volume creation
        general_settings = {
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "LinearUnmixingManualTest_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: self.path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: 0.25,
            Tags.DIM_VOLUME_Z_MM: 60,
            Tags.DIM_VOLUME_X_MM: 60,
            Tags.DIM_VOLUME_Y_MM: 30,
            Tags.WAVELENGTHS: self.VISUAL_WAVELENGTHS
        }
        self.settings = Settings(general_settings)
        self.settings.set_volume_creation_settings({
            Tags.SIMULATE_DEFORMED_LAYERS: True,
            Tags.STRUCTURES: self.create_example_tissue()
        })

        # Set component settings for linear unmixing.
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.PROPERTY_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_OXYHEMOGLOBIN_WAVELENGTHS: self.VISUAL_WAVELENGTHS,
            Tags.LINEAR_UNMIXING_DEOXYHEMOGLOBIN_WAVELENGTHS: self.VISUAL_WAVELENGTHS,
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: self.VISUAL_WAVELENGTHS
        }

        # Define device for simulation
        self.device = PencilBeamIlluminationGeometry()

        # Run simulation pipeline for all wavelengths in Tag.WAVELENGTHS
        pipeline = [
            VolumeCreationModelModelBasedAdapter(self.settings)
        ]
        simulate(pipeline, self.settings, self.device)

        # Run linear unmixing component with above specified settings
        lu.LinearUnmixingProcessingComponent(self.settings, "linear_unmixing").run()

    def perform_test(self):
        """
        This function visualizes the linear unmixing result, represented by the blood oxygen saturation.
        The user has to check if the test was successful.
        """

        self.logger.info("Testing linear unmixing...")

        # Load blood oxygen saturation
        lu_results = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        sO2 = lu_results["sO2"]

        # Load reference absorption for the first wavelength
        mua = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.PROPERTY_ABSORPTION_PER_CM,
                              wavelength=self.VISUAL_WAVELENGTHS[0])

        # Visualize linear unmixing result
        # The shape of the linear unmixing result should take after the reference absorption
        y_dim = int(mua.shape[1] / 2)
        plt.figure(figsize=(15, 15))
        plt.suptitle("Linear Unmixing - Visual Test")
        plt.subplot(121)
        plt.title("Absorption coefficients")
        plt.imshow(np.rot90(mua[:, y_dim, :], -1))
        plt.colorbar(fraction=0.05)
        plt.subplot(122)
        plt.title("Blood oxygen saturation")
        plt.imshow(np.rot90(sO2[:, y_dim, :], -1))
        plt.colorbar(fraction=0.05)
        plt.show()

        # clean up file after testing
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def create_example_tissue(self):
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        muscle_dictionary = Settings()
        muscle_dictionary[Tags.PRIORITY] = 1
        muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 10]
        muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        vessel_1_dictionary = Settings()
        vessel_1_dictionary[Tags.PRIORITY] = 3
        vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [self.settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                        10,
                                                        self.settings[Tags.DIM_VOLUME_Z_MM] / 2]
        vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [self.settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                      12,
                                                      self.settings[Tags.DIM_VOLUME_Z_MM] / 2]
        vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
        vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood(oxygenation=0.99)
        vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        epidermis_dictionary = Settings()
        epidermis_dictionary[Tags.PRIORITY] = 8
        epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 9]
        epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 10]
        epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
        epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
        epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        tissue_dict = Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["muscle"] = muscle_dictionary
        tissue_dict["epidermis"] = epidermis_dictionary
        tissue_dict["vessel_1"] = vessel_1_dictionary
        return tissue_dict


if __name__ == '__main__':
    test = TestLinearUnmixingVisual()
    test.setup()
    test.perform_test()
