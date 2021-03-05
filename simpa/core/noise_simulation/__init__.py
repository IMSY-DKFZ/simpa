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
from simpa.utils import StandardProperties
from abc import abstractmethod
import numpy as np
import pandas as pd
import json


class NoiseModelAdapterBase:
    """
    This class functions as a base class that can be used to easily define different
    noise models by extending the apply_noise_model function.
    """

    @abstractmethod
    def apply_noise_model(self, time_series_data: np.ndarray, settings: dict) -> np.ndarray:
        """
        Applies the defined noise model to the input time series data.

        :param time_series_data: the data the noise should be applied to.
        :param settings: the settings dictionary that contains the simulation instructions.
        :return: a numpy array of the same shape as the input data.
        """
        pass


class GaussianNoiseModel(NoiseModelAdapterBase):
    """
    The purpose of the GaussianNoiseModel class is to apply an additive gaussian noise to the input data.

    The mean and standard deviation of the model can be defined either by using the
    Tags.NOISE_MEAN and Tags.NOISE_STD tags, but they can also be set using a pandas dataframe
    that contains mean and standard deviation of noise for different wavelengths and
    temperatures. This can be done using the Tags.NOISE_MODEL_PATH tag.
    """

    def apply_noise_model(self, time_series_data, settings):
        mean_noise = 0
        std_noise = 400

        if Tags.NOISE_MEAN in settings:
            mean_noise = float(settings[Tags.NOISE_MEAN])

        if Tags.NOISE_STD in settings:
            std_noise = float(settings[Tags.NOISE_STD][str(settings[Tags.WAVELENGTH])]) * np.max(time_series_data)

        if Tags.NOISE_MODEL_PATH in settings:
            if Tags.WAVELENGTH in settings:
                wavelength = settings[Tags.WAVELENGTH]
            else:
                wavelength = 800
            if Tags.MEDIUM_TEMPERATURE_CELCIUS in settings:
                temperature = settings[Tags.MEDIUM_TEMPERATURE_CELCIUS]
            else:
                temperature = StandardProperties.BODY_TEMPERATURE_CELCIUS

            df = pd.read_csv(settings[Tags.NOISE_MODEL_PATH], sep="\t", index_col=0)
            noise_parameters = df.loc[int(wavelength), str(int(temperature))]
            mean_noise = json.loads(noise_parameters)[0]
            std_noise = json.loads(noise_parameters)[1]

        time_series_data = time_series_data + np.random.normal(mean_noise, std_noise, size=np.shape(time_series_data))

        return time_series_data
