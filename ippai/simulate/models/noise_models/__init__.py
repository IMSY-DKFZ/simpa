from ippai.simulate import Tags
from abc import abstractmethod
import numpy as np


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

        data = data + np.random.normal(mean_noise, std_noise, size=np.shape(data))

        return data
