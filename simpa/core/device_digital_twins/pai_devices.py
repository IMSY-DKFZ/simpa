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
from simpa.utils.settings_generator import Settings


class PAIDeviceBase:
    """
    This class represents a PAI device including the detection and illumination geometry.
    """

    @abstractmethod
    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        """
        It might be that certain device geometries need a certain dimensionality of the simulated PAI volume, or that
        it required the existence of certain Tags in the global global_settings.
        To this end, a  PAI device should use this method to inform the user about a mismatch of the desired device and
        throw a ValueError if that is the case.

        :raises ValueError: raises a value error if the prerequisites are not matched.
        :returns : True if the prerequisites are met.
        """
        pass

    @abstractmethod
    def adjust_simulation_volume_and_settings(self, global_settings: Settings) -> Settings:
        """
        In case that the PAI device needs space for the arrangement of detectors or illuminators in the volume,
        this method will update the volume accordingly.
        """
        pass

    @abstractmethod
    def get_illuminator_definition(self, global_settings: Settings):
        """
        TODO
        """
        pass

    @abstractmethod
    def get_detector_element_positions_mm(self, global_settings: Settings):
        """
        TODO
        """
        pass

    @abstractmethod
    def get_detector_element_orientations(self, global_settings: Settings):
        """
        TODO
        """
        pass