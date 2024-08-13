# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import apply_b_mode
from simpa.utils import Tags
from simpa.io_handling import load_data_field
from simpa.core.simulation import simulate
from simpa import KWaveAdapter, MCXAdapter, \
    SignedDelayMultiplyAndSumAdapter, ModelBasedVolumeCreationAdapter
from simpa.core.processing_components.monospectral.noise import GaussianNoise
from simpa import reconstruct_signed_delay_multiply_and_sum_pytorch
from simpa_tests.manual_tests import ReconstructionAlgorithmTestBaseClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SignedDelayMultiplyAndSumReconstruction(ReconstructionAlgorithmTestBaseClass):
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm. To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """

    def test_reconstruction_of_simulation(self):

        self.device.update_settings_for_use_of_model_based_volume_creator(self.settings)

        SIMUATION_PIPELINE = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            GaussianNoise(self.settings, "noise_initial_pressure"),
            KWaveAdapter(self.settings),
            SignedDelayMultiplyAndSumAdapter(self.settings)
        ]

        simulate(SIMUATION_PIPELINE, self.settings, self.device)

        self.reconstructed_image_pipeline = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                            self.settings[Tags.WAVELENGTH])

    def test_convenience_function(self):
        # Load simulated time series data
        time_series_sensor_data = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                  Tags.DATA_FIELD_TIME_SERIES_DATA, self.settings[Tags.WAVELENGTH])

        reconstruction_settings = self.settings.get_reconstruction_settings()

        # reconstruct via convenience function
        self.reconstructed_image_convenience = reconstruct_signed_delay_multiply_and_sum_pytorch(time_series_sensor_data, self.device.get_detection_geometry(), reconstruction_settings[Tags.DATA_FIELD_SPEED_OF_SOUND],  1.0 / (
            self.device.get_detection_geometry().sampling_frequency_MHz * 1000), self.settings[Tags.SPACING_MM], reconstruction_settings[Tags.RECONSTRUCTION_MODE], reconstruction_settings[Tags.RECONSTRUCTION_APODIZATION_METHOD])

        # apply envelope detection method if set
        if reconstruction_settings[Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION]:
            self.reconstructed_image_convenience = apply_b_mode(
                self.reconstructed_image_convenience, reconstruction_settings[Tags.RECONSTRUCTION_BMODE_METHOD])


if __name__ == '__main__':
    test = SignedDelayMultiplyAndSumReconstruction()
    test.run_test(show_figure_on_screen=False)
