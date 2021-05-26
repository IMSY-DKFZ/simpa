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

from abc import abstractmethod, ABC
from simpa.log import Logger
from simpa.utils import Settings
from simpa.utils import Tags
import numpy as np


class DigitalDeviceTwinBase:
    """
    This class represents a device that can be used for illumination, detection or both.
    """

    def __init__(self, device_position_mm: np.ndarray = None):
        if device_position_mm is None:
            self.device_position_mm = np.array([0, 0, 0])
        else:
            self.device_position_mm = device_position_mm
        self.logger = Logger()

    @abstractmethod
    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        """
        It might be that certain device geometries need a certain dimensionality of the simulated PAI volume, or that
        it required the existence of certain Tags in the global global_settings.
        To this end, a  PAI device should use this method to inform the user about a mismatch of the desired device and
        throw a ValueError if that is the case.

        :raises ValueError: raises a value error if the prerequisites are not matched.
        :returns: True if the prerequisites are met, False if they are not met, but no exception has been raised.

        """
        pass

    @abstractmethod
    def get_field_of_view_extent_mm(self) -> np.ndarray:
        """
        Returns the field of view extent of this imaging device.
        It is defined as a numpy array of the shape [xs, xe, ys, ye, zs, ze],
        where x, y, and z denote the coordinate axes and s and e denote the start and end
        positions. These coordinates are defined relatively to the probe position.
        """
        pass

    def get_field_of_view_mm(self) -> np.ndarray:
        """
        returns the absolute field of view in mm where the probe position is already
        accounted for.
        It is defined as a numpy array of the shape [xs, xe, ys, ye, zs, ze],
        where x, y, and z denote the coordinate axes and s and e denote the start and end
        positions.
        """
        position = self.device_position_mm
        field_of_view_extent = self.get_field_of_view_extent_mm()

        return np.asarray([position[0] + field_of_view_extent[0],
                           position[0] + field_of_view_extent[1],
                           position[1] + field_of_view_extent[2],
                           position[1] + field_of_view_extent[3],
                           position[2] + field_of_view_extent[4],
                           position[2] + field_of_view_extent[5]
                           ])


class PhotoacousticDevice(ABC, DigitalDeviceTwinBase):

    def __init__(self,  device_position_mm: np.ndarray = None):
        super(PhotoacousticDevice, self).__init__(device_position_mm=device_position_mm)
        self.detection_geometry = None
        self.illumination_geometries = []

    def set_detection_geometry(self, detection_geometry):
        self.detection_geometry = detection_geometry

    def add_illumination_geometry(self, illumination_geometry):
        self.illumination_geometries.append(illumination_geometry)

    def get_detection_geometry(self):
        return self.detection_geometry

    def get_illumination_geometry(self):
        """
        :return: None, if no illumination geometry was defined,
            an instance of IlluminationGeometryBase if exactly one geometry was defined,
            a list of IlluminationGeometryBase instances if more than one device was defined.
        """
        if len(self.illumination_geometries) == 0:
            return None

        if len(self.illumination_geometries) == 1:
            return self.illumination_geometries[0]

        return self.illumination_geometries

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        # TODO also integrate the illumination bounds -> maximum extent, smallest for start, biggest for end
        return self.detection_geometry.get_field_of_view_extent_mm()

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        _result = True
        if self.detection_geometry is not None \
                and not self.detection_geometry.check_settings_prerequisites(global_settings):
            _result = False
        for illumination_geometry in self.illumination_geometries:
            if illumination_geometry is not None \
                    and not illumination_geometry.check_settings_prerequisites(global_settings):
                _result = False
        return _result

    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        """
        Returns the default probe position in case none was given in the Settings dict.
        """
        return np.asarray([0, 0, 0])

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings):
        """
        This method can be overwritten by a photoacoustic device if the device poses special constraints to the
        volume that should be considered by the model-based volume creator.
        """
        pass
