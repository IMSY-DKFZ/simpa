# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined
import numpy as np
import torch


class UniformNoise(ProcessingComponent):
    """
    Applies uniform noise to the defined data field.
    The noise will be applied to all wavelengths.

    The noise will be uniformly distributed between [min, max[.

    Component Settings::

       Tags.NOISE_MIN (default: 0)
       Tags.NOISE_MAX (default: 1)
       Tags.NOISE_MODE (default: Tags.NOISE_MODE_ADDITIVE)
       Tags.DATA_FIELD (required)
    """

    def run(self, device):
        self.logger.info("Applying Uniform Noise Model...")
        min_noise = 0
        max_noise = 1
        mode = Tags.NOISE_MODE_ADDITIVE

        if Tags.DATA_FIELD not in self.component_settings.keys():
            msg = f"The field {Tags.DATA_FIELD} must be set in order to use the uniform_noise component."
            self.logger.critical(msg)
            raise KeyError(msg)

        data_field = self.component_settings[Tags.DATA_FIELD]

        if Tags.NOISE_MIN in self.component_settings.keys():
            min_noise = self.component_settings[Tags.NOISE_MIN]

        if Tags.NOISE_MAX in self.component_settings.keys():
            max_noise = self.component_settings[Tags.NOISE_MAX]

        if Tags.NOISE_MODE in self.component_settings.keys():
            mode = self.component_settings[Tags.NOISE_MODE]

        self.logger.debug(f"Noise model mode: {mode}")
        self.logger.debug(f"Noise model min: {min_noise}")
        self.logger.debug(f"Noise model max: {max_noise}")

        wavelength = self.global_settings[Tags.WAVELENGTH]
        data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)
        data_tensor = torch.as_tensor(data_array, dtype=torch.float32, device=self.torch_device)
        dist = torch.distributions.uniform.Uniform(torch.tensor(min_noise, dtype=torch.float32, device=self.torch_device),
                                                   torch.tensor(max_noise, dtype=torch.float32, device=self.torch_device))

        if mode == Tags.NOISE_MODE_ADDITIVE:
            data_tensor += dist.sample(data_tensor.shape)
        elif mode == Tags.NOISE_MODE_MULTIPLICATIVE:
            data_tensor *= dist.sample(data_tensor.shape)

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(data_tensor)

        save_data_field(data_tensor.cpu().numpy().astype(np.float64, copy=False),
                        self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Uniform Noise Model...[Done]")
