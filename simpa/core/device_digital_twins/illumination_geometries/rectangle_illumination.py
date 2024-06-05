# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import typing

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags
import numpy as np
from simpa.utils.serializer import SerializableSIMPAClass


class RectangleIlluminationGeometry(IlluminationGeometryBase):
    """
    Defines a rectangle illumination geometry.
    The device position is defined as the UPPER LEFT CORNER of the rectangle.

    Note: To create a global light which illuminates the entire tissue evenly (= creating a planar illumination geometry),
    create the following geometry using the tissue length and width:

    >>> global_light = RectangleIlluminationGeometry(length_mm=tissue_length_mm, width_mm=tissue_width_mm)
    """

    def __init__(self,
                 length_mm: int = 10,
                 width_mm: int = 10,
                 device_position_mm: typing.Optional[np.ndarray] = None,
                 source_direction_vector: typing.Optional[np.ndarray] = None,
                 field_of_view_extent_mm: typing.Optional[np.ndarray] = None):
        """
        :param length_mm: The length of the rectangle in mm.
        :param width_mm: The width of the rectangle in mm.
        :param device_position_mm: The device position in mm, the UPPER LEFT CORNER of the rectangle.
        If None, the position is defined as [0, 0, 0].
        :param source_direction_vector: Direction of the illumination source.
        If None, the direction is defined as [0, 0, 1].
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        """
        if device_position_mm is None:
            device_position_mm = np.zeros(3)

        if source_direction_vector is None:
            source_direction_vector = np.array([0, 0, 1])

        super(RectangleIlluminationGeometry, self).__init__(device_position_mm=device_position_mm,
                                                            source_direction_vector=source_direction_vector,
                                                            field_of_view_extent_mm=field_of_view_extent_mm)

        assert length_mm > 0
        assert width_mm > 0

        self.length_mm = length_mm
        self.width_mm = width_mm

    def get_mcx_illuminator_definition(self, global_settings: Settings) -> dict:
        """
        Returns the illumination parameters for MCX simulations.

        :param global_settings: The global settings.

        :return: The illumination parameters as a dictionary.
        """
        assert isinstance(global_settings, Settings), type(global_settings)

        source_type = Tags.ILLUMINATION_TYPE_PLANAR

        spacing = global_settings[Tags.SPACING_MM]

        device_position = list(np.rint(self.device_position_mm / spacing))

        self.logger.debug(device_position)

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [np.rint(self.width_mm / spacing) + 1, 0, 0]
        source_param2 = [0, np.rint(self.length_mm / spacing) + 1, 0]

        # If Pos=[10, 12, 0], Param1=[10, 0, 0], Param2=[0, 20, 0],
        # then illumination covers: x in [10, 20], y in [12, 32]
        # (https://github.com/fangq/mcx/discussions/220)
        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        """
        Serializes the object into a dictionary.

        :return: The dictionary representing the serialized object.
        """
        serialized_device = self.__dict__
        device_dict = {RectangleIlluminationGeometry.__name__: serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict) -> SerializableSIMPAClass:
        """
        Deserializes the provided dict into an object of this type.
        :param dictionary_to_deserialize: The dictionary to deserialize.

        :return: The deserialized object from the dictionary.
        """
        assert isinstance(dictionary_to_deserialize, dict), type(dictionary_to_deserialize)

        deserialized_device = RectangleIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value

        return deserialized_device
