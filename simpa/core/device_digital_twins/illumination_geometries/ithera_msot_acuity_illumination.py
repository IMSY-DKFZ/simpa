# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class MSOTAcuityIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT Acuity (Echo) photoacoustic device.
    The position is defined as the middle of the illumination slit.
    """

    def __init__(self):
        """
        Initializes the illumination source.
        """
        super().__init__()

    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm):

        source_type = Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO
        spacing = global_settings[Tags.SPACING_MM]
        source_position = [probe_position_mm[0]/spacing + 0.5,
                           probe_position_mm[1]/spacing + 0.5 - 16.46 / spacing,
                           probe_position_mm[2]/spacing + 0.5 + 5] # FIXME: This seems to be a bug

        # source_direction = [0, 0.381070, 0.9245460]       earlier calculation
        source_direction = [0, 0.356091613, 0.934451049]       # new calculation TODO: Check for correctness

        source_param1 = [30 / spacing, 0, 0, 0]

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
        del serialized_device["logger"]
        return {"MSOTAcuityIlluminationGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
