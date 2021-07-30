"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class PencilArrayIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a slit illumination geometry.
    The device position is defined as the middle of the slit.
    """
    def __init__(self, pitch_mm=0.5,
                 number_illuminators_x=100,
                 number_illuminators_y=100):
        """
        Initializes a slit illumination source.
        :param pitch_mm: Defines the x and y distance between the illumination positions
        :param number_illuminators_x: Defines the number of illuminators in the x direction
        :param number_illuminators_y: Defines the number of illuminators in the y direction
        """
        super().__init__()

        self.pitch_mm = pitch_mm
        self.number_illuminators_x = number_illuminators_x
        self.number_illuminators_y = number_illuminators_y

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        pass

    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_PENCILARRAY

        spacing = global_settings[Tags.SPACING_MM]

        device_position = probe_position_mm / spacing + 0.5

        source_direction = [0, 0, 1]

        source_param1 = [(self.number_illuminators_x * self.pitch_mm) / spacing,
                         0,
                         0, self.number_illuminators_x]

        source_param2 = [0,
                         (self.number_illuminators_y * self.pitch_mm) / spacing,
                         0, self.number_illuminators_y]

        return {
            "Type": source_type,
            "Pos": list(device_position),
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }