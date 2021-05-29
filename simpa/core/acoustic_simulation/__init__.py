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

from abc import abstractmethod
import numpy as np


class AcousticForwardAdapterBase:
    """
    This class should be used as a base for implementations of acoustic forward models.
    """

    @abstractmethod
    def forward_model(self, settings) -> np.ndarray:
        """
        This method performs the acoustic forward modeling given the initial pressure
        distribution and the acoustic tissue properties contained in the settings file.
        A deriving class needs to implement this method according to its model.

        :param settings: Setting dictionary
        :return: time series pressure data
        """
        pass

    def simulate(self, settings) -> np.ndarray:
        """
        Call this method to invoke the simulation process.

        :param settings: the settings dictionary containing all simulation parameters.
        :return: a numpy array containing the time series pressure data per detection element
        """
        print("Simulating the acoustic forward process...")

        time_series_pressure_data = self.forward_model(settings=settings)

        print("Simulating the acoustic forward process...[Done]")
        return time_series_pressure_data
