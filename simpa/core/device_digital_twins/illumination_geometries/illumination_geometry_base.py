# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.core.device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase
from simpa.utils import Settings
from numpy import ndarray
import numpy as np


class IlluminationGeometryBase(DigitalDeviceTwinBase):
    """
    This class is the base class for representing all illumination geometries.
    """

    def __init__(self, device_position_mm=None, source_direction_vector=None, field_of_view_extent_mm=None):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of illuminator positions.
        :type device_position_mm: ndarray

        :param source_direction_vector: Direction of the illumination source.
        :type source_direction_vector: ndarray

        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(IlluminationGeometryBase, self).__init__(device_position_mm=device_position_mm,
                                                       field_of_view_extent_mm=field_of_view_extent_mm)

        if source_direction_vector is None:
            self.source_direction_vector = [0, 0, 1]
        else:
            self.source_direction_vector = source_direction_vector
        self.normalized_source_direction_vector = self.source_direction_vector / np.linalg.norm(
            self.source_direction_vector)

    @abstractmethod
    def get_mcx_illuminator_definition(self, global_settings) -> dict:
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions.
        :type global_settings: Settings

        :return: Dictionary that includes all parameters needed for mcx.
        :rtype: dict
        """
        pass

    def check_settings_prerequisites(self, global_settings) -> bool:
        return True

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings) -> Settings:
        return global_settings

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"IlluminationGeometryBase": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = IlluminationGeometryBase()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
