# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags, Settings
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os
from simpa_tests.manual_tests import ManualIntegrationTestClass
from simpa_tests.test_utils.tissue_models import create_simple_tissue_model


class TestLinearUnmixingVisual(ManualIntegrationTestClass):
    """
    This test is a manual test, so visual confirmation is needed.
    """

    def setup(self):
        """
        This function lays the foundation for the manual test.
        """

        self.logger = sp.Logger()
        # TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
        #  point to the correct file in the PathManager().
        self.path_manager = sp.PathManager()

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
            Tags.STRUCTURES: create_simple_tissue_model(60, 60)
        })

        # Set component settings for linear unmixing.
        # We are interested in the blood oxygen saturation, so we have to execute linear unmixing with
        # the chromophores oxy- and deoxyhemoglobin and we have to set the tag LINEAR_UNMIXING_COMPUTE_SO2
        self.settings["linear_unmixing"] = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.LINEAR_UNMIXING_SPECTRA:
                sp.get_simpa_internal_absorption_spectra_by_names([Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
                                                                   Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN]),
            Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
            Tags.WAVELENGTHS: [800, 850]
        }

        # Define device for simulation
        self.device = sp.PencilBeamIlluminationGeometry()

        # Run simulation pipeline for all wavelengths in Tag.WAVELENGTHS
        self.pipeline = [
            sp.ModelBasedVolumeCreationAdapter(self.settings)
        ]

    def perform_test(self):
        """
        This function visualizes the linear unmixing result, represented by the blood oxygen saturation.
        The user has to check if the test was successful.
        """

        sp.simulate(self.pipeline, self.settings, self.device)

        # Run linear unmixing component with above specified settings
        sp.LinearUnmixing(self.settings, "linear_unmixing").run()

        self.logger.info("Testing linear unmixing...")

        # Load blood oxygen saturation
        self.lu_results = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.LINEAR_UNMIXING_RESULT)
        self.sO2 = self.lu_results["sO2"]

        # Load reference absorption for the first wavelength
        self.mua = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_ABSORPTION_PER_CM,
                                   wavelength=self.VISUAL_WAVELENGTHS[0])

    def tear_down(self):
        # clean up file after testing
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

    def visualise_result(self, show_figure_on_screen=True, save_path=None):
        # Visualize linear unmixing result
        # The shape of the linear unmixing result should take after the reference absorption

        ground_truth_sO2 = sp.load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_OXYGENATION)

        y_dim = int(self.mua.shape[1] / 2)
        plt.figure(figsize=(9, 3))
        plt.suptitle("Linear Unmixing - Visual Test")
        plt.subplot(131)
        plt.title("Ground Truth sO2 [%]")
        plt.imshow(np.rot90(ground_truth_sO2[:, y_dim, :] * 100, -1), vmin=0, vmax=100)
        plt.colorbar(fraction=0.05)
        plt.subplot(132)
        plt.title("Estimated sO2 [%]")
        plt.imshow(np.rot90(self.sO2[:, y_dim, :] * 100, -1), vmin=0, vmax=100)
        plt.colorbar(fraction=0.05)
        plt.subplot(133)
        plt.title("Absolute Difference")
        plt.imshow(np.rot90(np.abs(self.sO2 * 100 - ground_truth_sO2 * 100)[:, y_dim, :], -1), cmap="Reds", vmin=0)
        plt.colorbar()
        plt.tight_layout()
        if show_figure_on_screen:
            plt.show()
        else:
            if save_path is None:
                save_path = ""
            plt.savefig(save_path + "linear_unmixing_test.png")
        plt.close()


if __name__ == '__main__':
    test = TestLinearUnmixingVisual()
    test.run_test(show_figure_on_screen=False)
