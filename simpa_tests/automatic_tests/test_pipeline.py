# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.core.simulation import simulate
import numpy as np
from simpa_tests.test_utils import create_test_structure_parameters
import os
from simpa import ModelBasedVolumeCreationAdapter
from simpa.core.simulation_modules.optical_simulation_module.optical_forward_model_test_adapter import \
    OpticalForwardModelTestAdapter
from simpa.core.simulation_modules.acoustic_forward_module.acoustic_forward_model_test_adapter import \
    AcousticForwardModelTestAdapter
from simpa.core.device_digital_twins import RSOMExplorerP50


class TestPipeline(unittest.TestCase):

    def setUp(self):

        self.VOLUME_WIDTH_IN_MM = 4
        self.VOLUME_HEIGHT_IN_MM = 3
        self.SPACING = 0.25
        self.RANDOM_SEED = 4711

    def test_pipeline(self):
        # Seed the numpy random configuration prior to creating the settings file in
        # order to ensure that the same volume
        # is generated with the same random seed every time.

        np.random.seed(self.RANDOM_SEED)

        settings = {
            # These parameters set the general propeties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestName_"+str(self.RANDOM_SEED),
            Tags.SIMULATION_PATH: ".",
            Tags.SPACING_MM: self.SPACING,
            Tags.DIM_VOLUME_Z_MM: self.VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM,
            Tags.DIM_VOLUME_Y_MM: self.VOLUME_WIDTH_IN_MM,

            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [800],


            # The following parameters tell the script that we do not want any extra
            # modelling steps
            # Add the volume_creation_module to be simulated to the tissue
        }

        settings = Settings(settings)
        settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: create_test_structure_parameters()
            }
        )
        settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_TEST,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
        })
        settings.set_acoustic_settings({
                Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
                Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_TEST,
                Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
                Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
        })

        simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(settings),
            OpticalForwardModelTestAdapter(settings),
            AcousticForwardModelTestAdapter(settings),
        ]

        simulate(simulation_pipeline, settings, RSOMExplorerP50(0.1, 1, 1))

        if (os.path.exists(settings[Tags.SIMPA_OUTPUT_PATH]) and
                os.path.isfile(settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(settings[Tags.SIMPA_OUTPUT_PATH])
