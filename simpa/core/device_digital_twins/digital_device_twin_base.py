"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from abc import abstractmethod, ABC
from simpa.log import Logger
from simpa.utils import Settings
from simpa.utils import Tags
import numpy as np


class DigitalDeviceTwinBase:
    """
    This class represents a device that can be used for illumination, detection or both.
    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """
        Constructor of the base class for all digital devices.
        :param device_position_mm: Each device has an internal position which serves as origin for internal
        representations of e.g. detector element positions or illuminator positions.
        """
        if device_position_mm is None:
            self.device_position_mm = np.array([0, 0, 0])
        else:
            self.device_position_mm = device_position_mm

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-10, 10, -10, 10, -10, 10])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

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

    def get_field_of_view_mm(self) -> np.ndarray:
        """
        returns the absolute field of view in mm where the probe position is already
        accounted for.
        It is defined as a numpy array of the shape [xs, xe, ys, ye, zs, ze],
        where x, y, and z denote the coordinate axes and s and e denote the start and end
        positions.
        """
        position = self.device_position_mm
        field_of_view_extent = self.field_of_view_extent_mm

        field_of_view = np.asarray([position[0] + field_of_view_extent[0],
                                    position[0] + field_of_view_extent[1],
                                    position[1] + field_of_view_extent[2],
                                    position[1] + field_of_view_extent[3],
                                    position[2] + field_of_view_extent[4],
                                    position[2] + field_of_view_extent[5]
                                    ])
        if min(field_of_view) < 0:
            self.logger.warning(f"The field of view of the chosen device is not fully within the simulated volume, "
                                f"field of view is: {field_of_view}")
            field_of_view[field_of_view < 0] = 0

        return field_of_view


class PhotoacousticDevice(ABC, DigitalDeviceTwinBase):

    def __init__(self,  device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        super(PhotoacousticDevice, self).__init__(device_position_mm=device_position_mm,
                                                  field_of_view_extent_mm=field_of_view_extent_mm)
        self.detection_geometry = None
        self.illumination_geometries = []

    def set_detection_geometry(self, detection_geometry):
        if detection_geometry is None:
            msg = "The given detection_geometry must not be None!"
            self.logger.critical(msg)
            raise ValueError(msg)
        self.detection_geometry = detection_geometry

    def add_illumination_geometry(self, illumination_geometry):
        if illumination_geometry is None:
            msg = "The given illumination_geometry must not be None!"
            self.logger.critical(msg)
            raise ValueError(msg)
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

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings):
        """
        This method can be overwritten by a photoacoustic device if the device poses special constraints to the
        volume that should be considered by the model-based volume creator.
        """
        pass
