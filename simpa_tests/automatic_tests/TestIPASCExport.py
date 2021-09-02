"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
import os

from simpa.core.simulation import simulate
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.utils import Tags, Settings
from simpa_tests.test_utils import create_test_structure_parameters
from simpa.core import VolumeCreationModelModelBasedAdapter
from simpa.core.optical_simulation_module.optical_forward_model_test_adapter import OpticalForwardModelTestAdapter
from simpa.core.acoustic_forward_module.acoustic_forward_model_test_adapter import AcousticForwardModelTestAdapter


class TestDeviceUUID(unittest.TestCase):

    def setUp(self) -> None:
        self.VOLUME_WIDTH_IN_MM = 4
        self.VOLUME_HEIGHT_IN_MM = 3
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711

        self.settings = {
            # These parameters set the general propeties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestName_" + str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: ".",
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.WAVELENGTHS: [800],
            Tags.DO_IPASC_EXPORT: True
        }

        self.settings = Settings(self.settings)
        self.settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: create_test_structure_parameters()
            }
        )
        self.settings.set_optical_settings({
            "test": "test"
        })
        self.settings.set_acoustic_settings({
            "test": "test"
        })

        self.full_simulation_pipeline = [
            VolumeCreationModelModelBasedAdapter(self.settings),
            OpticalForwardModelTestAdapter(self.settings),
            AcousticForwardModelTestAdapter(self.settings),
        ]

        self.optical_simulation_pipeline = [
            VolumeCreationModelModelBasedAdapter(self.settings),
            OpticalForwardModelTestAdapter(self.settings)
        ]

        self.expected_ipasc_output_path = None

    def tearDown(self) -> None:
        pass

    def clean_up(self):
        print(f"Attempting to clean {self.settings[Tags.SIMPA_OUTPUT_PATH]}")
        if (os.path.exists(self.settings[Tags.SIMPA_OUTPUT_PATH]) and
                os.path.isfile(self.settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

        print(f"Attempting to clean {self.expected_ipasc_output_path}")
        if (os.path.exists(self.expected_ipasc_output_path) and
                os.path.isfile(self.expected_ipasc_output_path)):
            # Delete the created file
            os.remove(self.expected_ipasc_output_path)

    def test_ipasc_adapter_does_sensible_things(self):
        pass

    def test_file_is_not_created_on_only_optical_simulation(self):
        simulate(self.optical_simulation_pipeline, self.settings, RSOMExplorerP50(0.1, 1, 1))
        self.expected_ipasc_output_path = self.settings[Tags.SIMPA_OUTPUT_PATH].replace(".hdf5", "_ipasc.hfd5")
        self.assertTrue(not os.path.exists(self.expected_ipasc_output_path))
        self.clean_up()

    def test_file_is_created_on_acoustic_simulation(self):
        simulate(self.full_simulation_pipeline, self.settings, RSOMExplorerP50(0.1, 1, 1))
        self.expected_ipasc_output_path = self.settings[Tags.SIMPA_OUTPUT_PATH].replace(".hdf5", "_ipasc.hfd5")
        self.assertTrue(os.path.exists(self.expected_ipasc_output_path))
        self.clean_up()

    def test_file_is_created_on_full_simulation(self):
        pass

