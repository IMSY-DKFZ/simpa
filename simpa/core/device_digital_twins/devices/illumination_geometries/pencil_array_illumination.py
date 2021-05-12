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
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }