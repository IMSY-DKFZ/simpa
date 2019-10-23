from ippai.simulate import Tags
import numpy as np
from abc import abstractmethod


class ReconstructionAdapterBase:

    @abstractmethod
    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        """
        A deriving class needs to implement this method according to its model.

        :param time_series_sensor_data: the time series sensor data
        :param settings: Setting dictionary
        :return: a reconstructed photoacoustic image
        """
        pass

    def simulate(self, settings, acoustic_data_path):
        """

        :param settings:
        :param acoustic_data_path:
        :return:
        """
        print("Performing reconstruction...")

        time_series_sensor_data = np.load(acoustic_data_path)["time_series_data"]

        reconstructed_image = self.reconstruction_algorithm(time_series_sensor_data, settings)
        return reconstructed_image

        print("Performing reconstruction...Done]")
