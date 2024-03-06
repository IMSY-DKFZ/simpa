# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class MSOTInVisionIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT InVision photoacoustic device.
    """

    def __init__(self, invision_position=None):
        super().__init__()

        if invision_position is None:
            self.invision_position = [0, 0, 0]
        else:
            self.invision_position = invision_position

        det_sep_half = 24.74 / 2
        detector_iso_distance = 74.05 / 2
        detector_width = 2 * 6.12

        self.device_positions_mm = list()
        self.source_direction_vectors = list()
        self.slit_vectors_mm = list()

        for index in [0, 1, 2, 3, 4]:
            for y_offset_factor in [+1, -1]:
                angle = index * 2.0 * np.pi / 5.0
                illumination_angle = -0.41608649 * y_offset_factor
                v = np.array([-np.sin(angle), np.sin(illumination_angle), -np.cos(angle)])
                v /= np.linalg.norm(v)
                slit_vector = np.array([np.cos(angle), 0, -np.sin(angle)]) * detector_width
                slit_middle_on_circle = np.array([np.sin(angle), 0.0, np.cos(angle)]) * detector_iso_distance
                y_offset = np.array([0.0, det_sep_half * y_offset_factor, 0.0])
                pos = np.array(self.invision_position) + slit_middle_on_circle + y_offset - 0.5*slit_vector
                self.device_positions_mm.append(pos)
                self.source_direction_vectors.append(v)
                self.slit_vectors_mm.append(slit_vector)

        divergence_angle = 0.165806  # full beam divergence angle measured at Full Width at Half Maximum (FWHM)
        full_width_at_half_maximum = 2.0 * np.tan(0.5 * divergence_angle)  # FWHM of beam divergence
        # standard deviation of gaussian with FWHM
        self.sigma = full_width_at_half_maximum / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def get_mcx_illuminator_definition(self, global_settings):
        self.logger.debug(self.invision_position)
        source_type = Tags.ILLUMINATION_TYPE_SLIT

        spacing = global_settings[Tags.SPACING_MM]

        source_positions = list(list(pos / spacing + 1) for pos in self.device_positions_mm)

        source_directions = list(list(v) for v in self.source_direction_vectors)

        source_param1s = list(list(slit_vector / spacing) + [0.0] for slit_vector in self.slit_vectors_mm)

        source_param2s = [[self.sigma, 0, 0, 0]]*len(self.device_positions_mm)

        return {
            "Type": source_type,
            "Pos": source_positions,
            "Dir": source_directions,
            "Param1": source_param1s,
            "Param2": source_param2s
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"MSOTInVisionIlluminationGeometry": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTInVisionIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
