# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from abc import ABC
from simpa.core.device_digital_twins import DigitalDeviceTwinBase


class PhotoacousticDevice(DigitalDeviceTwinBase, ABC):
    """Base class of a photoacoustic device. It consists of one detection geometry that describes the geometry of the
    single detector elements and a list of illuminators.

    A Photoacoustic Device can be initialized as follows::

        import simpa as sp
        import numpy as np

        # Initialise a PhotoacousticDevice with its position and field of view
        device = sp.PhotoacousticDevice(device_position_mm=np.array([10, 10, 0]),
            field_of_view_extent_mm=np.array([-20, 20, 0, 0, 0, 20]))

        # Option 1) Set the detection geometry position relative to the PhotoacousticDevice
        device.set_detection_geometry(sp.DetectionGeometry(),
            detector_position_relative_to_pa_device=np.array([0, 0, -10]))

        # Option 2) Set the detection geometry position absolute
        device.set_detection_geometry(
            sp.DetectionGeometryBase(device_position_mm=np.array([10, 10, -10])))

        # Option 1) Add the illumination geometry position relative to the PhotoacousticDevice
        device.add_illumination_geometry(sp.IlluminationGeometry(),
            illuminator_position_relative_to_pa_device=np.array([0, 0, 0]))

        # Option 2) Add the illumination geometry position absolute
        device.add_illumination_geometry(
            sp.IlluminationGeometryBase(device_position_mm=np.array([10, 10, 0]))

    Attributes:
        detection_geometry (DetectionGeometryBase): Geometry of the detector elements.
        illumination_geometries (list): List of illuminations defined by :py:class:`IlluminationGeometryBase`.
    """

    def __init__(self, device_position_mm=None, field_of_view_extent_mm=None):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(PhotoacousticDevice, self).__init__(device_position_mm=device_position_mm,
                                                  field_of_view_extent_mm=field_of_view_extent_mm)
        self.detection_geometry = None
        self.illumination_geometries = []

    def set_detection_geometry(self, detection_geometry,
                               detector_position_relative_to_pa_device=None):
        """Sets the detection geometry for the PA device. The detection geometry can be instantiated with an absolute
        position or it can be instantiated without the device_position_mm argument but a position relative to the
        position of the PhotoacousticDevice. If both absolute and relative positions are given, the absolute position
        is chosen as position of the detection geometry.

        :param detection_geometry: Detection geometry of the PA device.
        :type detection_geometry: DetectionGeometryBase
        :param detector_position_relative_to_pa_device: Position of the detection geometry relative to the PA device.
        :type detector_position_relative_to_pa_device: ndarray
        :raises ValueError: if the detection_geometry is None

        """
        if detection_geometry is None:
            msg = "The given detection_geometry must not be None!"
            self.logger.critical(msg)
            raise ValueError(msg)
        if np.linalg.norm(detection_geometry.device_position_mm) == 0 and \
                detector_position_relative_to_pa_device is not None:
            detection_geometry.device_position_mm = np.add(self.device_position_mm,
                                                           detector_position_relative_to_pa_device)
        self.detection_geometry = detection_geometry

    def add_illumination_geometry(self, illumination_geometry, illuminator_position_relative_to_pa_device=None):
        """Adds an illuminator to the PA device. The illumination geometry can be instantiated with an absolute
        position or it can be instantiated without the device_position_mm argument but a position relative to the
        position of the PhotoacousticDevice. If both absolute and relative positions are given, the absolute position
        is chosen as position of the illumination geometry.

        :param illumination_geometry: Geometry of the illuminator.
        :type illumination_geometry: IlluminationGeometryBase
        :param illuminator_position_relative_to_pa_device: Position of the illuminator relative to the PA device.
        :type illuminator_position_relative_to_pa_device: ndarray
        :raises ValueError: if the illumination_geometry is None

        """
        if illumination_geometry is None:
            msg = "The given illumination_geometry must not be None!"
            self.logger.critical(msg)
            raise ValueError(msg)
        if np.linalg.norm(illumination_geometry.device_position_mm) == 0:
            if illuminator_position_relative_to_pa_device is not None:
                illumination_geometry.device_position_mm = np.add(self.device_position_mm,
                                                                  illuminator_position_relative_to_pa_device)
            else:
                illumination_geometry.device_position_mm = self.device_position_mm
        self.illumination_geometries.append(illumination_geometry)

    def get_detection_geometry(self):
        """
        :return: None if no detection geometry was set or an instance of DetectionGeometryBase.
        :rtype: None, DetectionGeometryBase
        """
        return self.detection_geometry

    def get_illumination_geometry(self):
        """
        :return: None, if no illumination geometry was defined,
            an instance of IlluminationGeometryBase if exactly one geometry was defined,
            a list of IlluminationGeometryBase instances if more than one device was defined.
        :rtype: None, IlluminationGeometryBase
        """
        if len(self.illumination_geometries) == 0:
            return None

        if len(self.illumination_geometries) == 1:
            return self.illumination_geometries[0]

        return self.illumination_geometries

    def check_settings_prerequisites(self, global_settings) -> bool:
        _result = True
        if self.detection_geometry is not None \
                and not self.detection_geometry.check_settings_prerequisites(global_settings):
            _result = False
        for illumination_geometry in self.illumination_geometries:
            if illumination_geometry is not None \
                    and not illumination_geometry.check_settings_prerequisites(global_settings):
                _result = False
        return _result

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings):
        pass

    def update_settings_for_use_of_segmentation_based_volume_creator(self, global_settings):
        pass

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"PhotoacousticDevice": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = PhotoacousticDevice(
            device_position_mm=dictionary_to_deserialize["device_position_mm"],
            field_of_view_extent_mm=dictionary_to_deserialize["field_of_view_extent_mm"])
        det_geometry = dictionary_to_deserialize["detection_geometry"]
        if det_geometry != "None":
            deserialized_device.set_detection_geometry(dictionary_to_deserialize["detection_geometry"])
        if "illumination_geometries" in dictionary_to_deserialize:
            for illumination_geometry in dictionary_to_deserialize["illumination_geometries"]:
                deserialized_device.illumination_geometries.append(illumination_geometry)

        return deserialized_device
