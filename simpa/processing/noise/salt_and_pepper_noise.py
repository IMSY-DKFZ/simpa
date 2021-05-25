"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.utils import EPS
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.simulation_components import ProcessingComponent

import numpy as np


class SaltAndPepperNoiseProcessingComponent(ProcessingComponent):
    """
    Applies salt and pepper noise to the defined data field.
    The noise will be applied to all wavelengths.

    The noise will be 50% salt and 50% pepper noise, but both can be set to the same value using the
    NOISE_MIN and NOISE_MAX fields.

    Component Settings:
       **Tags.NOISE_MIN (default: min(data_field))
       **Tags.NOISE_MAX (default: max(data_field))
       **Tags.NOISE_FREQUENCY (default: 0.01)
       **data_field (required)
    """

    def run(self, device):
        self.logger.info("Applying Salt And Pepper Noise Model...")
        mode = Tags.NOISE_MODE_ADDITIVE

        if Tags.DATA_FIELD not in self.component_settings.keys():
            self.logger.critical()
            raise KeyError(f"The field {Tags.DATA_FIELD} must be set in order to use the"+
                           f"salt_and_pepper_noise component.")

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

        if Tags.NOISE_MODE in self.component_settings.keys():
            mode = self.component_settings[Tags.NOISE_MODE]

        self.logger.debug(f"Noise model mode: {mode}")
        self.logger.debug(f"Noise model min: {min_noise}")
        self.logger.debug(f"Noise model max: {max_noise}")
        self.logger.debug(f"Noise model frequency: {noise_frequency}")

        num_data_points = int(np.round(np.mult(np.shape(data_array)) * noise_frequency / 2))
        coords_min = np.random.choice(data_array, num_data_points)
        coords_max = np.random.choice(data_array, num_data_points)

        data_array[coords_min] = min_noise
        data_array[coords_max] = max_noise

        save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Salt And Pepper Noise Model...[Done]")
