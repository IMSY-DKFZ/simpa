from ippai.simulate import Tags
from ippai.io_handling.io_hdf5 import load_hdf5
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

        time_series_sensor_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], acoustic_data_path)[Tags.TIME_SERIES_DATA]

        reconstructed_image = self.reconstruction_algorithm(time_series_sensor_data, settings)
        return reconstructed_image

        print("Performing reconstruction...Done]")
