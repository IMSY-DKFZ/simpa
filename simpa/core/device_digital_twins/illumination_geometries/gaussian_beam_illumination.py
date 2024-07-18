# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from math import log, sqrt
from collections.abc import Sized
from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Tags


class GaussianBeamIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a Gaussian beam illumination geometry.
    The position is defined as the middle of the beam.
    """

    def __init__(self, beam_radius_mm=None, focal_length_mm=None, device_position_mm=None,
                 field_of_view_extent_mm=None):
        """
        :param beam_radius_mm: Initial radius of the gaussian beam at half maximum (full width at half maximum (FWHM))
        in mm.
        :type beam_radius_mm: int, float
        :param focal_length_mm: Focal length of the gaussian beam in mm. Can be positive (focussed beam), negative
        (cone-shaped beam) or None (collimated beam).
        :type focal_length_mm: int, float
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(GaussianBeamIlluminationGeometry, self).__init__(device_position_mm=device_position_mm,
                                                               field_of_view_extent_mm=field_of_view_extent_mm)

        if beam_radius_mm is None:
            beam_radius_mm = 0

        self.beam_radius_mm = beam_radius_mm
        self.focal_length_mm = focal_length_mm

    def get_mcx_illuminator_definition(self, global_settings) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_GAUSSIAN

        spacing = global_settings[Tags.SPACING_MM]

        device_position = list(self.device_position_mm / spacing + 0.5)

        source_direction = list(self.normalized_source_direction_vector)
        if self.focal_length_mm is not None:  # the focal length can optionally be added as 4th parameter
            source_direction.append(self.focal_length_mm / spacing)

        # mcx takes the beam_radius in at the 1/e^2 (2 * sigma) threshold, but we use FWHM (sqrt(2*ln2))*sigma)
        # by multiplying the input radius by (2 / sqrt(2 * log(2))) we can convert it
        source_param1 = [self.beam_radius_mm / spacing * 2 / sqrt(2 * log(2)), 0, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"GaussianBeamIlluminationGeometry": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = GaussianBeamIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            if not isinstance(value, Sized) and value != 'None':
                deserialized_device.__dict__[key] = value
        return deserialized_device
