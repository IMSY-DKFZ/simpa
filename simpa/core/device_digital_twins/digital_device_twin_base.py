# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.log import Logger
import numpy as np
import hashlib
import uuid
from simpa.utils.serializer import SerializableSIMPAClass
from simpa.utils.calculate import are_equal


class DigitalDeviceTwinBase(SerializableSIMPAClass):
    """
    This class represents a device that can be used for illumination, detection or a combined photoacoustic device
    which has representations of both.
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
        if device_position_mm is None:
            self.device_position_mm = np.array([0, 0, 0])
        else:
            self.device_position_mm = device_position_mm

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-10, 10, -10, 10, -10, 10])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

        self.logger = Logger()

    def __eq__(self, other):
        """
        Checks each key, value pair in the devices.
        """
        if isinstance(other, DigitalDeviceTwinBase):
            if self.__dict__.keys() != other.__dict__.keys():
                return False
            for self_key, self_value in self.__dict__.items():
                other_value = other.__dict__[self_key]
                if not are_equal(self_value, other_value):
                    return False
            return True
        return False

    @abstractmethod
    def check_settings_prerequisites(self, global_settings) -> bool:
        """
        It might be that certain device geometries need a certain dimensionality of the simulated PAI volume, or that
        it requires the existence of certain Tags in the global global_settings.
        To this end, a  PAI device should use this method to inform the user about a mismatch of the desired device and
        throw a ValueError if that is the case.

        :param global_settings: Settings for the entire simulation pipeline.
        :type global_settings: Settings

        :raises ValueError: raises a value error if the prerequisites are not matched.
        :returns: True if the prerequisites are met, False if they are not met, but no exception has been raised.
        :rtype: bool

        """
        pass

    @abstractmethod
    def update_settings_for_use_of_model_based_volume_creator(self, global_settings):
        """
        This method can be overwritten by a PA device if the device poses special constraints to the
        volume that should be considered by the model-based volume creator.

        :param global_settings: Settings for the entire simulation pipeline.
        :type global_settings: Settings
        """
        pass

    def get_field_of_view_mm(self) -> np.ndarray:
        """
        Returns the absolute field of view in mm where the probe position is already
        accounted for.
        It is defined as a numpy array of the shape [xs, xe, ys, ye, zs, ze],
        where x, y, and z denote the coordinate axes and s and e denote the start and end
        positions.

        :return: Absolute field of view in mm where the probe position is already accounted for.
        :rtype: ndarray
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

    def generate_uuid(self):
        """
        Generates a universally unique identifier (uuid) for each device.
        :return:
        """
        class_dict = self.__dict__
        m = hashlib.md5()
        m.update(str(class_dict).encode('utf-8'))
        return str(uuid.UUID(m.hexdigest()))

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {"DigitalDeviceTwinBase": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = DigitalDeviceTwinBase(
            device_position_mm=dictionary_to_deserialize["device_position_mm"],
            field_of_view_extent_mm=dictionary_to_deserialize["field_of_view_extent_mm"])
        return deserialized_device
