from ippai.simulate import Tags, StandardProperties
from abc import abstractmethod
import numpy as np
import pandas as pd
import json


class NoiseModelAdapterBase:

    @abstractmethod
    def apply_noise_model(self, data, settings):
        """

        :param data:
        :param settings:
        :return:
        """
        pass


class GaussianNoise(NoiseModelAdapterBase):
    """
    This class is reponsible to apply an additive gaussian noise to the input data.
    """
    def apply_noise_model(self, data, settings):
        mean_noise = 0
        std_noise = 1

        if Tags.NOISE_MEAN in settings:
            mean_noise = float(settings[Tags.NOISE_MEAN])

        if Tags.NOISE_STD in settings:
            std_noise = float(settings[Tags.NOISE_STD])

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

        data = data + np.random.normal(mean_noise, std_noise, size=np.shape(data))

        return data
