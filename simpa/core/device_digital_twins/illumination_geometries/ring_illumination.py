# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import typing

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags
import numpy as np
from simpa.utils.serializer import SerializableSIMPAClass


class RingIlluminationGeometry(IlluminationGeometryBase):
    """
    Defines a ring illumination geometry.
    The device position is defined as the center of the ring.

    Note: To create a ring light which illuminates a square tissue with the same center and any inner radius r,
    create the following geometry using the tissue width:

    >>> ring_light = RingIlluminationGeometry(inner_radius_in_mm=r,
                                              outer_radius_in_mm=tissue_width / 2.,
                                              device_position_mm=np.array([tissue_width / 2., tissue_width / 2., 0]))
    """

    def __init__(self,
                 outer_radius_in_mm: float = 1,
                 inner_radius_in_mm: float = 0,
                 lower_angular_bound: float = 0,
                 upper_angular_bound: float = 0,
                 device_position_mm: typing.Optional[np.ndarray] = None,
                 source_direction_vector: typing.Optional[np.ndarray] = None,
                 field_of_view_extent_mm: typing.Optional[np.ndarray] = None):
        """
        :param outer_radius_in_mm: The outer radius of the ring in mm.
        :param inner_radius_in_mm: The inner radius of the ring in mm. If 0, should match the disk illumination.
        :param lower_angular_bound: The lower angular bound in radians. If both bounds are 0, than no bound is applied.
        Note that the bound of 0 starts from the x-axis on the right side and is applied clockwise.
        :param upper_angular_bound: The upper angular bound in radians. If both bounds are 0, than no bound is applied.
        Note that the bound is applied clockwise in relation to the lower bound.
        :param device_position_mm: The device position in mm, the center of the ring.
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

        super(RingIlluminationGeometry, self).__init__(device_position_mm=device_position_mm,
                                                       source_direction_vector=source_direction_vector,
                                                       field_of_view_extent_mm=field_of_view_extent_mm)

        assert inner_radius_in_mm >= 0, f"The inner radius has to be 0 or positive, not {inner_radius_in_mm}!"
        assert outer_radius_in_mm >= inner_radius_in_mm, \
            f"The outer radius ({outer_radius_in_mm}) has to be at least as large " \
            f"as the inner radius ({inner_radius_in_mm})!"

        assert lower_angular_bound >= 0, f"The lower angular bound has to be 0 or positive, not {lower_angular_bound}!"
        assert upper_angular_bound >= lower_angular_bound, \
            f"The outer radius ({upper_angular_bound}) has to be at least as large " \
            f"as the inner radius ({lower_angular_bound})!"

        self.outer_radius_in_mm = outer_radius_in_mm
        self.inner_radius_in_mm = inner_radius_in_mm
        self.lower_angular_bound = lower_angular_bound
        self.upper_angular_bound = upper_angular_bound

    def get_mcx_illuminator_definition(self, global_settings: Settings) -> dict:
        """
        Returns the illumination parameters for MCX simulations.
        :param global_settings: The global settings.
        :return: The illumination parameters as a dictionary.
        """
        assert isinstance(global_settings, Settings), type(global_settings)

        source_type = Tags.ILLUMINATION_TYPE_RING

        spacing = global_settings[Tags.SPACING_MM]

        device_position = list(self.device_position_mm / spacing + 1)  # No need to round

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [self.outer_radius_in_mm / spacing,
                         self.inner_radius_in_mm / spacing,
                         self.lower_angular_bound,
                         self.upper_angular_bound]

        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1
        }

    def serialize(self) -> dict:
        """
        Serializes the object into a dictionary.
        :return: The dictionary representing the serialized object.
        """
        serialized_device = self.__dict__
        device_dict = {RingIlluminationGeometry.__name__: serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict) -> SerializableSIMPAClass:
        """
        Deserializes the provided dict into an object of this type.
        :param dictionary_to_deserialize: The dictionary to deserialize.
        :return: The deserialized object from the dictionary.
        """
        assert isinstance(dictionary_to_deserialize, dict), type(dictionary_to_deserialize)

        deserialized_device = RingIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value

        return deserialized_device
