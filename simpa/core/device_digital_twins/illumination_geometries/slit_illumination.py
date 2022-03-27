# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class SlitIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a slit illumination geometry.
    The device position is defined as the middle of the slit.
    """
    def __init__(self, slit_vector_mm=None, direction_vector_mm=None, device_position_mm=None,
                 field_of_view_extent_mm=None):
        """
        :param slit_vector_mm: Defines the slit in vector form. For example a slit along the x-axis with length 5mm
            would be defined as [5, 0, 0].
        :type slit_vector_mm: list
        :param direction_vector_mm: Direction vector in which the slit illuminates.
            Defined analogous to the slit vector.
        :type direction_vector_mm: list
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(SlitIlluminationGeometry, self).__init__(device_position_mm=device_position_mm,
                                                       field_of_view_extent_mm=field_of_view_extent_mm)

        if slit_vector_mm is None:
            slit_vector_mm = [5, 0, 0]

        if direction_vector_mm is None:
            direction_vector_mm = [0, 0, 1]

        self.slit_vector_mm = slit_vector_mm
        direction_vector_mm[0] = direction_vector_mm[0] / np.linalg.norm(direction_vector_mm)
        direction_vector_mm[1] = direction_vector_mm[1] / np.linalg.norm(direction_vector_mm)
        direction_vector_mm[2] = direction_vector_mm[2] / np.linalg.norm(direction_vector_mm)
        self.direction_vector_norm = direction_vector_mm

    def get_mcx_illuminator_definition(self, global_settings) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_SLIT

        spacing = global_settings[Tags.SPACING_MM]

        device_position = (self.device_position_mm/spacing) + 0.5 - 0.5 * np.array(self.slit_vector_mm)/spacing

        self.logger.debug(device_position)

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [self.slit_vector_mm[0]/spacing,
                         self.slit_vector_mm[1]/spacing,
                         self.slit_vector_mm[2]/spacing, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": list(device_position),
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"SlitIlluminationGeometry": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = SlitIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
