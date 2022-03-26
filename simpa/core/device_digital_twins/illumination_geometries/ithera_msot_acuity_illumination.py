# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags
import numpy as np


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
                           probe_position_mm[1]/spacing + 0.5,
                           probe_position_mm[2]/spacing + 0.5]

        # y position relative to the membrane:
        # The laser is located 43.2 mm  behind the membrane with an angle of 22.4 degrees.
        # However, the incident of laser and image plane is located 2.8 behind the membrane (outside of the device).
        y_pos_relative_to_membrane = np.tan(np.deg2rad(22.4)) * (43.2 + 2.8)

        direction_vector = np.array([0, y_pos_relative_to_membrane, 43.2 + 2.8])
        source_direction = list(direction_vector/np.linalg.norm(direction_vector))

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
        return {"MSOTAcuityIlluminationGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
