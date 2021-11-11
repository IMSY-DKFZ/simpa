# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined
import numpy as np


class SaltAndPepperNoise(ProcessingComponent):
    """
    Applies salt and pepper noise to the defined data field.
    The noise will be applied to all wavelengths.

    The noise will be 50% salt and 50% pepper noise, but both can be set to the same value using the
    NOISE_MIN and NOISE_MAX fields.

    Component Settings::

       Tags.NOISE_MIN (default: min(data_field))
       Tags.NOISE_MAX (default: max(data_field))
       Tags.NOISE_FREQUENCY (default: 0.01)
       Tags.DATA_FIELD (required)
    """

    def run(self, device):
        self.logger.info("Applying Salt And Pepper Noise Model...")

        if Tags.DATA_FIELD not in self.component_settings.keys():
            msg = f"The field {Tags.DATA_FIELD} must be set in order to use the" \
                  f"salt_and_pepper_noise component."
            self.logger.critical(msg)
            raise KeyError(msg)

        data_field = self.component_settings[Tags.DATA_FIELD]

        wavelength = self.global_settings[Tags.WAVELENGTH]
        data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        min_noise = np.min(data_array)
        max_noise = np.max(data_array)
        noise_frequency = 0.01

        if Tags.NOISE_FREQUENCY in self.component_settings.keys():
            noise_frequency = self.component_settings[Tags.NOISE_FREQUENCY]

        if Tags.NOISE_MIN in self.component_settings.keys():
            min_noise = self.component_settings[Tags.NOISE_MIN]

        if Tags.NOISE_MAX in self.component_settings.keys():
            max_noise = self.component_settings[Tags.NOISE_MAX]

        self.logger.debug(f"Noise model min: {min_noise}")
        self.logger.debug(f"Noise model max: {max_noise}")
        self.logger.debug(f"Noise model frequency: {noise_frequency}")

        shape = np.shape(data_array)
        num_data_points = int(np.round(np.prod(shape) * noise_frequency / 2))

        coords_min = tuple([np.random.randint(0, i - 1, int(num_data_points)) for i in shape])
        coords_max = tuple([np.random.randint(0, i - 1, int(num_data_points)) for i in shape])

        data_array[coords_min] = min_noise
        data_array[coords_max] = max_noise

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(data_array)

        save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Salt And Pepper Noise Model...[Done]")
