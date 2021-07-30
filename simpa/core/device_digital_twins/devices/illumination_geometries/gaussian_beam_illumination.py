"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class GaussianBeamIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a Gaussian beam illumination geometry.
    The position is defined as the middle of the beam.
    """

    def __init__(self, beam_radius_mm=None):
        super(GaussianBeamIlluminationGeometry, self).__init__()
        if beam_radius_mm is None:
            beam_radius_mm = 0

        self.beam_radius_mm = beam_radius_mm

    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_GAUSSIAN

        spacing = global_settings[Tags.SPACING_MM]

        device_position = probe_position_mm / spacing + 0.5

        source_direction = [0, 0, 1]

        source_param1 = [self.beam_radius_mm, 0, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": list(device_position),
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }