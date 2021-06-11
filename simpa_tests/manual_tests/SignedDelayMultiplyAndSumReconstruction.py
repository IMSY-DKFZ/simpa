"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling import load_data_field, load_hdf5
from simpa.core.simulation import simulate
from simpa.core.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import AcousticForwardModelKWaveAdapter
from simpa.core.optical_simulation_module.optical_forward_model_mcx_adapter import OpticalForwardModelMcxAdapter
from simpa.core.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter
from simpa.core.volume_creation_module.volume_creation_module_model_based_adapter import \
    VolumeCreationModelModelBasedAdapter
from simpa.core.processing_components.noise import GaussianNoiseProcessingComponent
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
            VolumeCreationModelModelBasedAdapter(self.settings),
            OpticalForwardModelMcxAdapter(self.settings),
            GaussianNoiseProcessingComponent(self.settings, "noise_initial_pressure"),
            AcousticForwardModelKWaveAdapter(self.settings),
            ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter(self.settings)
        ]

        simulate(SIMUATION_PIPELINE, self.settings, self.device)

        reconstructed_image_path = generate_dict_path(
            Tags.RECONSTRUCTED_DATA,
            wavelength=self.settings[Tags.WAVELENGTH])

        reconstructed_image = load_hdf5(
            self.settings[Tags.SIMPA_OUTPUT_PATH],
            reconstructed_image_path)[Tags.RECONSTRUCTED_DATA]

        self.plot_reconstruction_compared_with_initial_pressure(reconstructed_image,
                                                                "Reconstructed image using adapter")

    def test_convenience_function(self):
        # Load simulated time series data
        time_series_sensor_data = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                  Tags.TIME_SERIES_DATA, self.settings[Tags.WAVELENGTH])

        # reconstruct image using convenience function
        reconstructed_image = reconstruct_signed_delay_multiply_and_sum_pytorch(time_series_sensor_data,
                                                                                self.device.get_detection_geometry(),
                                                                                self.settings
                                                                                )

        self.plot_reconstruction_compared_with_initial_pressure(reconstructed_image,
                                                                "Reconstructed image using convenience function")

if __name__ == '__main__':
    test = SignedDelayMultiplyAndSumReconstruction()
    test.test_reconstruction_of_simulation()
    test.test_convenience_function()
