"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class MSOTInVisionIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT Acuity (Echo) photoacoustic device.
    """

    def __init__(self, geometry_id: int = 0):
        """
        Initializes the illumination source.
        """
        super().__init__()
        self.geometry_id = geometry_id

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        pass

    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm: np.ndarray):
        self.logger.debug(probe_position_mm)
        source_type = Tags.ILLUMINATION_TYPE_MSOT_INVISION

        spacing = global_settings[Tags.SPACING_MM]

        angle = 0.0
        det_sep_half = 24.74 / (2 * spacing)
        detector_iso_distance = 74.05 / (2 * spacing)
        illumination_angle = -0.41608649

        if self.geometry_id == 0:
            angle = 0.0
        elif self.geometry_id == 1:
            angle = 0.0
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif self.geometry_id == 2:
            angle = 1.25664
        elif self.geometry_id == 3:
            angle = 1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif self.geometry_id == 4:
            angle = -1.25664
        elif self.geometry_id == 5:
            angle = -1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif self.geometry_id == 6:
            angle = 2.51327
        elif self.geometry_id == 7:
            angle = 2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif self.geometry_id == 8:
            angle = -2.51327
        elif self.geometry_id == 9:
            angle = -2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle

        source_position = [probe_position_mm[0]/spacing + 1 + np.sin(angle) * detector_iso_distance,
                           probe_position_mm[1]/spacing + 1 + det_sep_half,
                           probe_position_mm[2]/spacing + 1 + np.cos(angle) * detector_iso_distance]

        length = np.sqrt(np.sin(angle) ** 2 + np.sin(illumination_angle) ** 2 + np.cos(angle) ** 2)
        source_direction = [-np.sin(angle) / length,
                            np.sin(illumination_angle) / length,
                            np.cos(angle) / length]

        source_param1 = [spacing, self.geometry_id, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": source_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }