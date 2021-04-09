# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.utils import Tags
from simpa.utils import EPS
from simpa.io_handling import load_data_field, save_data_field
from simpa.core import SimulationModule

import numpy as np


class GaussianNoiseModel(SimulationModule):
    """
        Applies Gaussian noise to the defined data field.
        The noise will be applied to all wavelengths.
        :param kwargs:
           **Tags.NOISE_MEAN (default: 0)
           **Tags.NOISE_STD (default: 1)
           **Tags.NOISE_MODE (default: Tags.NOISE_MODE_ADDITIVE)
           **Tags.NOISE_NON_NEGATIVITY_CONSTRAINT (default: False)
           **data_field (required)
        """

    def run(self):
        self.logger.info("Applying Gaussian Noise Model...")
        mean = 0
        std = 1
        mode = Tags.NOISE_MODE_ADDITIVE
        non_negative = False

        if Tags.DATA_FIELD not in self.component_settings.keys():
            self.logger.critical()
            raise KeyError(f"The field {Tags.DATA_FIELD} must be set in order to use the gaussian_noise field.")

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

        if mode == Tags.NOISE_MODE_ADDITIVE:
            data_array = data_array + np.random.normal(mean, std, size=np.shape(data_array))
        elif mode == Tags.NOISE_MODE_MULTIPLICATIVE:
            data_array = data_array * np.random.normal(mean, std, size=np.shape(data_array))

        if non_negative:
            data_array[data_array < EPS] = EPS
        save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Applying Gaussian Noise Model...[Done]")
