# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from simpa.core.device_digital_twins import IlluminationGeometryBase
from simpa.utils import Settings, Tags


class MSOTAcuityIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT Acuity (Echo) photoacoustic device.
    """

    def __init__(self):
        """
        Initializes the illumination source.
        """
        super().__init__()

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        pass

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