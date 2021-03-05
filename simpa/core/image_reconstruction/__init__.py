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
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import load_data_field
from abc import abstractmethod


class ReconstructionAdapterBase:
    """
    TODO
    """

    @abstractmethod
    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        """
        A deriving class needs to implement this method according to its model.

        :param time_series_sensor_data: the time series sensor data
        :param settings: Setting dictionary
        :return: a reconstructed photoacoustic image
        """
        pass

    def simulate(self, settings):
        """

        :param settings:
        :return:
        """
        print("Performing reconstruction...")

        time_series_sensor_data = load_data_field(settings[Tags.SIMPA_OUTPUT_PATH], Tags.TIME_SERIES_DATA, settings[Tags.WAVELENGTH])

        reconstructed_image = self.reconstruction_algorithm(time_series_sensor_data, settings)
        print("Performing reconstruction...[Done]")
        return reconstructed_image
