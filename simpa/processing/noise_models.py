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

from simpa.utils import Settings
from simpa.utils import Tags
from simpa.utils import EPS
from simpa.io_handling import load_data_field, save_data_field

import numpy as np


def gaussian_noise(settings: Settings, **kwargs):
    """
    Applies Gaussian noise to the defined data field.
    The noise will be applied to all wavelengths.

    :param settings:
    :param kwargs:
       **Tags.NOISE_MEAN (default: 0)
       **Tags.NOISE_STD (default: 1)
       **Tags.NOISE_MODE (default: Tags.NOISE_MODE_ADDITIVE)
       **Tags.NOISE_NON_NEGATIVITY_CONSTRAINT (default: False)
       **data_field (required)
    """
    mean = 0
    std = 1
    mode = Tags.NOISE_MODE_ADDITIVE
    non_negativ = False

    if Tags.DATA_FIELD not in kwargs.keys():
        raise KeyError("The field " + Tags.DATA_FIELD + " must be set in order to use the gaussian_noise field.")

    data_field = kwargs[Tags.DATA_FIELD]

    if Tags.NOISE_MEAN in kwargs.keys():
        mean = kwargs[Tags.NOISE_MEAN]

    if Tags.NOISE_STD in kwargs.keys():
        std = kwargs[Tags.NOISE_STD]

    if Tags.NOISE_MODE in kwargs.keys():
        mode = kwargs[Tags.NOISE_MODE]

    if Tags.NOISE_NON_NEGATIVITY_CONSTRAINT in kwargs.keys():
        non_negativ = kwargs[Tags.NOISE_NON_NEGATIVITY_CONSTRAINT]

    for wavelength in settings[Tags.WAVELENGTHS]:
        data_array = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        if mode == Tags.NOISE_MODE_ADDITIVE:
            data_array = data_array + np.random.normal(mean, std, size=np.shape(data_array))
        elif mode == Tags.NOISE_MODE_MULTIPLICATIVE:
            data_array = data_array * np.random.normal(mean, std, size=np.shape(data_array))

        if non_negativ:
            data_array[data_array < EPS] = EPS
        save_data_field(data_array, settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)
