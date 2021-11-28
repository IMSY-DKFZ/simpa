# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import os
import numpy as np

from simpa.core.processing_components.monospectral.noise import *
from simpa.utils import Tags, Settings
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.core.simulation import simulate
from simpa.utils import TISSUE_LIBRARY
from simpa.io_handling import load_data_field
from simpa import ModelBasedVolumeCreationAdapter


class TestNoiseModels(unittest.TestCase):

    @staticmethod
    def create_background_parameters(background_value):
        background_structure_dictionary = dict()
        background_structure_dictionary[Tags.PRIORITY] = 0
        background_structure_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(
            mua=background_value, mus=background_value, g=0.5
        )
        background_structure_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
        return {Tags.BACKGROUND: background_structure_dictionary}

    def validate_noise_model_results(self, noise_model, noise_model_settings,
                                     background_value,
                                     expected_mean, expected_std,
                                     error_margin=0.05):
        np.random.seed(self.RANDOM_SEED)

        settings = {
            # These parameters set the general propeties of the simulated volume
            Tags.RANDOM_SEED: self.RANDOM_SEED,
            Tags.VOLUME_NAME: "TestName_" + str(self.RANDOM_SEED),
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
                Tags.STRUCTURES: self.create_background_parameters(background_value=background_value)
            }
        )

        settings["noise_model_settings"] = noise_model_settings

        simulation_pipeline = [
            ModelBasedVolumeCreationAdapter(settings),
            noise_model(settings, "noise_model_settings")
        ]

        try:
            simulate(simulation_pipeline, settings, RSOMExplorerP50(0.1, 1, 1))

            absorption = load_data_field(file_path=settings[Tags.SIMPA_OUTPUT_PATH],
                                         data_field=Tags.DATA_FIELD_ABSORPTION_PER_CM,
                                         wavelength=800)
            actual_mean = np.mean(absorption)
            actual_std = np.std(absorption)
            self.assertTrue(np.abs(actual_mean - expected_mean) < 1e-10 or
                            np.abs(actual_mean - expected_mean) / expected_mean < error_margin,
                            f"The mean was not as expected. Expected {expected_mean} but was {actual_mean}")
            self.assertTrue(np.abs(actual_std - expected_std) < 1e-10 or
                            np.abs(actual_std - expected_std) / expected_std < error_margin,
                            f"The mean was not as expected. Expected {expected_std} but was {actual_std}")
        finally:
            if (os.path.exists(settings[Tags.SIMPA_OUTPUT_PATH]) and
                    os.path.isfile(settings[Tags.SIMPA_OUTPUT_PATH])):
                # Delete the created file
                os.remove(settings[Tags.SIMPA_OUTPUT_PATH])

    def setUp(self):

        self.VOLUME_WIDTH_IN_MM = 10
        self.VOLUME_HEIGHT_IN_MM = 10
        self.SPACING = 1
        self.RANDOM_SEED = 4711

    @unittest.expectedFailure
    def test_gaussian_noise_no_data_field(self):
        noise_model = GaussianNoise

        # Test fails without set data field
        settings = {}
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=2.0,
                                          expected_mean=2,
                                          expected_std=1)

    def test_gaussian_noise_correct_mean_and_standard_deviations(self):
        noise_model = GaussianNoise

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=1.0,
                                          expected_std=0.1)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=2.0,
                                          expected_std=0.1)

        # Test multiplicative version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.0,
                                          expected_std=0.0)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.0,
                                          expected_std=0.1)

        # Test with using non-negativity contraint
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=1.0,
                                          expected_std=0.1)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=2.0,
                                          expected_std=0.1)

    @unittest.expectedFailure
    def test_gamma_noise_no_data_field(self):
        noise_model = GammaNoise

        # Test fails without set data field
        settings = {}
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=2.0,
                                          expected_mean=2,
                                          expected_std=1)

    def test_gamma_noise_correct_mean_and_standard_deviations(self):
        noise_model = GammaNoise

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_SCALE: 1,
            Tags.NOISE_SHAPE: 0.1,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.1,
                                          expected_std=np.sqrt(1 * 0.1))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.1,
                                          expected_std=np.sqrt(1 * 0.1))

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_SCALE: 2.5,
            Tags.NOISE_SHAPE: 1.7,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=2.5 * 1.7,
                                          expected_std=np.sqrt(2.5 * 2.5 * 1.7))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.0 + 2.5 * 1.7,
                                          expected_std=np.sqrt(2.5 * 2.5 * 1.7))

        # Test multiplicative version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_SCALE: 1.0,
            Tags.NOISE_SHAPE: 1.7,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.0,
                                          expected_std=0.0)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.0 * 1.7,
                                          expected_std=np.sqrt(1.0 * 1.0 * 1.7))

    @unittest.expectedFailure
    def test_poisson_noise_no_data_field(self):
        noise_model = PoissonNoise

        # Test fails without set data field
        settings = {}
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=2.0,
                                          expected_mean=2,
                                          expected_std=1)

    def test_poisson_noise_correct_mean_and_standard_deviations(self):
        noise_model = PoissonNoise

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 1.7,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=1.7,
                                          expected_std=np.sqrt(1.7))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=2.7,
                                          expected_std=np.sqrt(1.7))

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 33.7,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=33.7,
                                          expected_std=np.sqrt(33.7))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=34.7,
                                          expected_std=np.sqrt(33.7))

        # Test multiplicative version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MEAN: 1.7,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE
        }
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.0,
                                          expected_std=0.0)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.7,
                                          expected_std=np.sqrt(1.7))

    @unittest.expectedFailure
    def test_saltandpepper_noise_no_data_field(self):
        noise_model = SaltAndPepperNoise

        # Test fails without set data field
        settings = {}
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=2.0,
                                          expected_mean=2,
                                          expected_std=1)

    def test_saltandpepper_noise_correct_mean_and_standard_deviations(self):
        noise_model = SaltAndPepperNoise

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MIN: 0,
            Tags.NOISE_MAX: 2,
            Tags.NOISE_FREQUENCY: 0.1
        }

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.1,
                                          expected_std=np.sqrt((0.95 * 0 ** 2 + 0.05 * 2 ** 2) - 0.1**2))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=1.0,
                                          expected_std=np.sqrt((0.9 * 1 ** 2 + 0.05 * 2 ** 2 + 0.05 * 0 ** 2) - 1.0**2))

    @unittest.expectedFailure
    def test_uniform_noise_no_data_field(self):
        noise_model = UniformNoise

        # Test fails without set data field
        settings = {}
        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=2.0,
                                          expected_mean=2,
                                          expected_std=1)

    def test_uniform_noise_correct_mean_and_standard_deviations(self):
        noise_model = UniformNoise

        # Test additive version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MIN: 5,
            Tags.NOISE_MAX: 12,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE
        }

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=8.5,
                                          expected_std=np.sqrt(1/12 * (12 - 5) ** 2))

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=9.5,
                                          expected_std=np.sqrt(1/12 * (12 - 5) ** 2))

        # Test multiplicative version
        settings = {
            Tags.DATA_FIELD: Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.NOISE_MIN: 1,
            Tags.NOISE_MAX: 3,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE
        }

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=0.0,
                                          expected_mean=0.0,
                                          expected_std=0.0)

        self.validate_noise_model_results(noise_model=noise_model,
                                          noise_model_settings=settings,
                                          background_value=1.0,
                                          expected_mean=2.0,
                                          expected_std=np.sqrt(1 / 12 * (3 - 1) ** 2))
