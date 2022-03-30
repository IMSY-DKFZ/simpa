# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class MSOTInVisionIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT InVision photoacoustic device.
    """

    def __init__(self, invision_position=None, geometry_id=0):
        """
        :param geometry_id: ID of the specific InVision illuminator.
        :type geometry_id: int
        """
        super().__init__()

        if invision_position is None:
            self.invision_position = [0, 0, 0]
        else:
            self.invision_position = invision_position

        self.geometry_id = geometry_id

        angle = 0.0
        det_sep_half = 24.74 / 2
        detector_iso_distance = 74.05 / 2
        illumination_angle = -0.41608649

        if geometry_id == 0:
            angle = 0.0
        elif geometry_id == 1:
            angle = 0.0
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif geometry_id == 2:
            angle = 1.25664
        elif geometry_id == 3:
            angle = 1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif geometry_id == 4:
            angle = -1.25664
        elif geometry_id == 5:
            angle = -1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif geometry_id == 6:
            angle = 2.51327
        elif geometry_id == 7:
            angle = 2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif geometry_id == 8:
            angle = -2.51327
        elif geometry_id == 9:
            angle = -2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle

        self.device_position_mm = [self.invision_position[0] + np.sin(angle) * detector_iso_distance,
                                   self.invision_position[1] + det_sep_half,
                                   self.invision_position[2] + np.cos(angle) * detector_iso_distance]

        self.source_direction_vector = np.array([-np.sin(angle),
                                                 np.sin(illumination_angle),
                                                 np.cos(angle)])

        self.normalized_source_direction_vector = self.source_direction_vector / np.linalg.norm(
            self.source_direction_vector)

    def get_mcx_illuminator_definition(self, global_settings):
        self.logger.debug(self.invision_position)
        source_type = Tags.ILLUMINATION_TYPE_MSOT_INVISION

        spacing = global_settings[Tags.SPACING_MM]

        source_position = list(np.array(self.device_position_mm) / spacing + 1)

        source_direction = list(self.normalized_source_direction_vector)

        source_param1 = [spacing, self.geometry_id, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": source_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
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
