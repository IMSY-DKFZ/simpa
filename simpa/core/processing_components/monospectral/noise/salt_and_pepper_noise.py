# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined
import numpy as np
import torch


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
        data_tensor = torch.as_tensor(data_array, dtype=torch.float32, device=self.torch_device)

        min_noise = torch.min(data_tensor).item()
        max_noise = torch.max(data_tensor).item()
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

        dist = torch.distributions.uniform.Uniform(torch.tensor(-1.0, dtype=torch.float32, device=self.torch_device),
                                                   torch.tensor(1.0, dtype=torch.float32, device=self.torch_device))
        sample = dist.sample(data_tensor.shape)
        sample_cutoff = 1.0 - noise_frequency
        data_tensor[sample > sample_cutoff] = min_noise
        data_tensor[-sample > sample_cutoff] = max_noise

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(data_tensor)

        save_data_field(data_tensor.cpu().numpy().astype(np.float64, copy=False),
                        self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Salt And Pepper Noise Model...[Done]")
