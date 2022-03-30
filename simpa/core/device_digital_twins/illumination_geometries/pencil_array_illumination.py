# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class PencilArrayIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a pencil array illumination geometry.
    The device position is defined as the middle of the array.
    """
    def __init__(self, pitch_mm=0.5, number_illuminators_x=100, number_illuminators_y=100, device_position_mm=None,
                 field_of_view_extent_mm=None):
        """
        :param pitch_mm: Defines the x and y distance between the illumination positions
        :type pitch_mm: float
        :param number_illuminators_x: Defines the number of illuminators in the x direction
        :type number_illuminators_x: int
        :param number_illuminators_y: Defines the number of illuminators in the y direction
        :type number_illuminators_y: int
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(PencilArrayIlluminationGeometry, self).__init__(device_position_mm=device_position_mm,
                                                              field_of_view_extent_mm=field_of_view_extent_mm)

        self.pitch_mm = pitch_mm
        self.number_illuminators_x = number_illuminators_x
        self.number_illuminators_y = number_illuminators_y

    def get_mcx_illuminator_definition(self, global_settings) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_PENCILARRAY

        spacing = global_settings[Tags.SPACING_MM]

        device_position = list(self.device_position_mm / spacing + 0.5)

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [(self.number_illuminators_x * self.pitch_mm) / spacing,
                         0,
                         0, self.number_illuminators_x]

        source_param2 = [0,
                         (self.number_illuminators_y * self.pitch_mm) / spacing,
                         0, self.number_illuminators_y]

        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"PencilArrayIlluminationGeometry": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = PencilArrayIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
