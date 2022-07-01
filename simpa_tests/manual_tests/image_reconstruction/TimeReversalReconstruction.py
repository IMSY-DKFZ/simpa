# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags

from simpa.io_handling import load_data_field
from simpa.core.simulation import simulate
from simpa import KWaveAdapter, MCXAdapter, \
    TimeReversalAdapter, ModelBasedVolumeCreationAdapter, GaussianNoise
from simpa import reconstruct_delay_and_sum_pytorch
from simpa_tests.manual_tests import ReconstructionAlgorithmTestBaseClass

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TimeReversalReconstruction(ReconstructionAlgorithmTestBaseClass):
    """
    This test runs a simulation creating an example volume of geometric shapes and reconstructs it with the Delay and
    Sum algorithm. To verify that the test was successful a user has to evaluate the displayed reconstruction.
    """

    def test_reconstruction_of_simulation(self):

        self.device.update_settings_for_use_of_model_based_volume_creator(self.settings)

        SIMULATION_PIPELINE = [
            ModelBasedVolumeCreationAdapter(self.settings),
            MCXAdapter(self.settings),
            GaussianNoise(self.settings, "noise_initial_pressure"),
            KWaveAdapter(self.settings),
            TimeReversalAdapter(self.settings)
        ]

        simulate(SIMULATION_PIPELINE, self.settings, self.device)

        self.reconstructed_image_pipeline = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                                            self.settings[Tags.WAVELENGTH])

    def test_convenience_function(self):
        self.reconstructed_image_convenience = None


if __name__ == '__main__':
    test = TimeReversalReconstruction()
    test.run_test(show_figure_on_screen=False)
