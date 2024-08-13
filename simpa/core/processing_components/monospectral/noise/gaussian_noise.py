# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.utils import EPS
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined
import numpy as np
import torch


class GaussianNoise(ProcessingComponent):
    """
    Applies Gaussian noise to the defined data field.
    The noise will be applied to all wavelengths.
    Component Settings::

       Tags.NOISE_MEAN (default: 0)
       Tags.NOISE_STD (default: 1)
       Tags.NOISE_MODE (default: Tags.NOISE_MODE_ADDITIVE)
       Tags.NOISE_NON_NEGATIVITY_CONSTRAINT (default: False)
       Tags.DATA_FIELD (required)
    """

    def run(self, device):
        self.logger.info("Applying Gaussian Noise Model...")
        mean = 0
        std = 1
        mode = Tags.NOISE_MODE_ADDITIVE
        non_negative = False

        if Tags.DATA_FIELD not in self.component_settings.keys():
            msg = f"The field {Tags.DATA_FIELD} must be set in order to use the gaussian_noise field."
            self.logger.critical(msg)
            raise KeyError(msg)

        data_field = self.component_settings[Tags.DATA_FIELD]

        if Tags.NOISE_MEAN in self.component_settings.keys():
            mean = self.component_settings[Tags.NOISE_MEAN]

        if Tags.NOISE_STD in self.component_settings.keys():
            std = self.component_settings[Tags.NOISE_STD]

        if Tags.NOISE_MODE in self.component_settings.keys():
            mode = self.component_settings[Tags.NOISE_MODE]

        if Tags.NOISE_NON_NEGATIVITY_CONSTRAINT in self.component_settings.keys():
            non_negative = self.component_settings[Tags.NOISE_NON_NEGATIVITY_CONSTRAINT]

        self.logger.debug(f"Noise model mode: {mode}")
        self.logger.debug(f"Noise model mean: {mean}")
        self.logger.debug(f"Noise model std: {std}")
        self.logger.debug(f"Noise model non-negative: {non_negative}")

        wavelength = self.global_settings[Tags.WAVELENGTH]
        data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)
        data_tensor = torch.as_tensor(data_array, dtype=torch.float32, device=self.torch_device)
        dist = torch.distributions.normal.Normal(torch.tensor(mean, dtype=torch.float32, device=self.torch_device),
                                                 torch.tensor(std, dtype=torch.float32, device=self.torch_device))

        if mode == Tags.NOISE_MODE_ADDITIVE:
            data_tensor += dist.sample(data_tensor.shape)
        elif mode == Tags.NOISE_MODE_MULTIPLICATIVE:
            data_tensor *= dist.sample(data_tensor.shape)

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(data_tensor)

        if non_negative:
            data_tensor[data_tensor < EPS] = EPS
        save_data_field(data_tensor.cpu().numpy().astype(np.float64, copy=False),
                        self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Gaussian Noise Model...[Done]")
