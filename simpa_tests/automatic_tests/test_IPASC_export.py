# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import os
import numpy as np

from simpa.core.simulation import simulate
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.utils import Tags, Settings
from simpa_tests.test_utils import create_test_structure_parameters
from simpa import ModelBasedVolumeCreationAdapter
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_test_adapter import \
    OpticalForwardModelTestAdapter
from simpa.core.simulation_modules.acoustic_forward_module.acoustic_forward_model_test_adapter import \
    AcousticForwardModelTestAdapter
from simpa.core.simulation_modules.reconstruction_module.reconstruction_module_test_adapter import \
    ReconstructionModuleTestAdapter

from ipasc_tool import load_data as load_ipasc
from simpa.io_handling import load_hdf5 as load_simpa
from simpa.io_handling import load_data_field as load_simpa_datafield


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
            Tags.WAVELENGTHS: [800, 801],
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
        self.settings.set_reconstruction_settings({
            "test": "test"
        })

        self.acoustic_simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            OpticalForwardModelTestAdapter(self.settings),
            AcousticForwardModelTestAdapter(self.settings),
        ]

        self.optical_simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            OpticalForwardModelTestAdapter(self.settings)
        ]

        self.full_simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(self.settings),
            OpticalForwardModelTestAdapter(self.settings),
            AcousticForwardModelTestAdapter(self.settings),
            ReconstructionModuleTestAdapter(self.settings)
        ]

        self.device = RSOMExplorerP50(0.1, 12, 12)

        self.expected_ipasc_output_path = None
        self.expected_ipasc_output_path = None

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

    def assert_ipasc_file_binary_contents_is_matching_simpa_simulation(self, simpa_path, ipasc_path):
        if not (os.path.exists(simpa_path) and os.path.isfile(simpa_path)):
            raise AssertionError("simpa_path not valid")
        if not (os.path.exists(ipasc_path) and os.path.isfile(ipasc_path)):
            raise AssertionError("ipasc_path not valid")

        simpa_data = load_simpa(simpa_path)
        settings = Settings(simpa_data[Tags.SETTINGS])
        wavelengths = settings[Tags.WAVELENGTHS]
        simpa_time_series = []
        for wl in wavelengths:
            simpa_time_series.append(load_simpa_datafield(simpa_path, Tags.DATA_FIELD_TIME_SERIES_DATA, wl))
        simpa_time_series = np.reshape(np.asarray(simpa_time_series), (-1, ))

        ipasc_data = load_ipasc(ipasc_path)
        ipasc_time_series = np.reshape(ipasc_data.binary_time_series_data, (-1, ))

        time_series_sum = np.abs(np.sum(simpa_time_series-ipasc_time_series))
        self.assertAlmostEqual(time_series_sum, 0.00000, places=4, msg=f"Expected {0} but was {time_series_sum}")

        simpa_positions = self.device.get_detection_geometry().get_detector_element_positions_base_mm()
        simpa_orientations = self.device.get_detection_geometry().get_detector_element_orientations()

        ipasc_positions = ipasc_data.get_detector_position() * 1000
        ipasc_orientations = ipasc_data.get_detector_orientation()

        positions_sum = np.sum(np.abs(simpa_positions - ipasc_positions))
        orientations_sum = np.abs(np.sum(simpa_orientations - ipasc_orientations))
        self.assertAlmostEqual(positions_sum, 0.00000, places=4, msg=f"Expected {0} but was {positions_sum}")
        self.assertAlmostEqual(orientations_sum, 0.00000, places=4, msg=f"Expected {0} but was {orientations_sum}")

    def test_file_is_not_created_on_only_optical_simulation(self):
        simulate(self.optical_simulation_pipeline, self.settings, self.device)
        self.expected_ipasc_output_path = self.settings[Tags.SIMPA_OUTPUT_PATH].replace(".hdf5", "_ipasc.hdf5")
        self.assertTrue(not os.path.exists(self.expected_ipasc_output_path))
        self.clean_up()

    def test_file_is_created_on_acoustic_simulation(self):
        simulate(self.acoustic_simulation_pipeline, self.settings, self.device)
        self.expected_ipasc_output_path = self.settings[Tags.SIMPA_OUTPUT_PATH].replace(".hdf5", "_ipasc.hdf5")
        self.assertTrue(os.path.exists(self.expected_ipasc_output_path))
        self.assert_ipasc_file_binary_contents_is_matching_simpa_simulation(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                                            self.expected_ipasc_output_path)
        self.clean_up()

    def test_file_is_created_on_full_simulation(self):
        simulate(self.full_simulation_pipeline, self.settings, self.device)
        self.expected_ipasc_output_path = self.settings[Tags.SIMPA_OUTPUT_PATH].replace(".hdf5", "_ipasc.hdf5")
        self.assertTrue(os.path.exists(self.expected_ipasc_output_path))
        self.assert_ipasc_file_binary_contents_is_matching_simpa_simulation(self.settings[Tags.SIMPA_OUTPUT_PATH],
                                                                            self.expected_ipasc_output_path)
        self.clean_up()
