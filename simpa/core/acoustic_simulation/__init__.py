# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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


class AcousticForwardAdapterBase:
    """
    Use this class as a base for implementations of optical forward models.
    """

    @abstractmethod
    def forward_model(self, settings):
        """
        A deriving class needs to implement this method according to its model.

        :param settings: Setting dictionary
        :return: Fluence in units of J/cm^2
        """
        pass

    def simulate(self, optical_properties_path, settings):
        """
        Call this method to invoke the simulation process.
        TODO

        A adapter that implements the forward_model method, will take acoustic properties as input
        and return the time series pressure data as output.

        :param optical_properties_path: path to a *.npz file that contains the following tags:
            Tags.PROPERTY_ABSORPTION_PER_CM -> contains the optical absorptions in units of one per centimeter
            Tags.PROPERTY_SCATTERING_PER_CM -> contains the optical scattering in units of one per centimeter
            Tags.PROPERTY_ANISOTROPY -> contains the dimensionless optical scattering anisotropy
        :param settings:
        :return:
        """
        print("Simulating the acoustic forward process...")

        time_series_pressure_data = self.forward_model(settings=settings)

        print("Simulating the acoustic forward process...[Done]")
        return time_series_pressure_data
