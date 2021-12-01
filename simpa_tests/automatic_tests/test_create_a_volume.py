# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.core.simulation import simulate
import os
from simpa_tests.test_utils import create_test_structure_parameters
from simpa import ModelBasedVolumeCreationAdapter
from simpa.core.device_digital_twins import RSOMExplorerP50


class TestCreateVolume(unittest.TestCase):

    def test_create_volume(self):

        random_seed = 4711
        basic_settings = {
            Tags.WAVELENGTHS: [800, 801],
            Tags.RANDOM_SEED: random_seed,
            Tags.VOLUME_NAME: "FlowPhantom_" + str(random_seed).zfill(6),
            Tags.SIMULATION_PATH: ".",
            Tags.SPACING_MM: 0.3,
            Tags.DIM_VOLUME_Z_MM: 5,
            Tags.DIM_VOLUME_X_MM: 4,
            Tags.DIM_VOLUME_Y_MM: 3
        }

        settings = Settings(basic_settings)
        settings.set_volume_creation_settings({Tags.STRUCTURES: create_test_structure_parameters()})

        simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(settings)
        ]

        simulate(simulation_pipeline, settings, RSOMExplorerP50(0.1, 1, 1))

        if (os.path.exists(settings[Tags.SIMPA_OUTPUT_PATH]) and
           os.path.isfile(settings[Tags.SIMPA_OUTPUT_PATH])):
            os.remove(settings[Tags.SIMPA_OUTPUT_PATH])
