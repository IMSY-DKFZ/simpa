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


class GammaNoiseProcessingComponent(ProcessingComponent):
    """
    Applies Gaussian noise to the defined data field.
    The noise will be applied to all wavelengths.

    Component Settings:
       **Tags.NOISE_SHAPE (default: 2)
       **Tags.NOISE_SCALE (default: 2)
       **Tags.NOISE_MODE (default: Tags.NOISE_MODE_ADDITIVE)
       **data_field (required)
    """

    def run(self, device):
        self.logger.info("Applying Gamma Noise Model...")
        shape = 2
        scale = 2
        mode = Tags.NOISE_MODE_ADDITIVE

        if Tags.DATA_FIELD not in self.component_settings.keys():
            msg = f"The field {Tags.DATA_FIELD} must be set in order to use the gamma_noise field."
            self.logger.critical(msg)
            raise KeyError(msg)

        data_field = self.component_settings[Tags.DATA_FIELD]

        if Tags.NOISE_SHAPE in self.component_settings.keys():
            shape = self.component_settings[Tags.NOISE_SHAPE]

        if Tags.NOISE_SCALE in self.component_settings.keys():
            scale = self.component_settings[Tags.NOISE_SCALE]

        if Tags.NOISE_MODE in self.component_settings.keys():
            mode = self.component_settings[Tags.NOISE_MODE]

        self.logger.debug(f"Noise model mode: {mode}")
        self.logger.debug(f"Noise model shape: {shape}")
        self.logger.debug(f"Noise model scale: {scale}")

        wavelength = self.global_settings[Tags.WAVELENGTH]
        data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        if mode == Tags.NOISE_MODE_ADDITIVE:
            data_array = data_array + np.random.gamma(shape, scale, size=np.shape(data_array))
        elif mode == Tags.NOISE_MODE_MULTIPLICATIVE:
            data_array = data_array * np.random.gamma(shape, scale, size=np.shape(data_array))

        save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Gamma Noise Model...[Done]")
