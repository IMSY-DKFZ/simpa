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


class SlitIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents a slit illumination geometry.
    The device position is defined as the middle of the slit.
    """
    def __init__(self, slit_vector_mm: list = None, direction_vector_mm: list = None):
        """
        Initializes a slit illumination source.
        :param slit_vector_mm: Defines the slit in vector form. For example a slit along the x-axis with length 5mm
            would be defined as [5, 0, 0]
        :param direction_vector_mm: Direction vector in which the slit illuminates.
            Defined analogous to the slit vector.
        """
        super().__init__()

        if slit_vector_mm is None:
            slit_vector_mm = [5, 0, 0]

        if direction_vector_mm is None:
            direction_vector_mm = [0, 0, 1]

        self.slit_vector_mm = slit_vector_mm
        direction_vector_mm[0] = direction_vector_mm[0] / np.linalg.norm(direction_vector_mm)
        direction_vector_mm[1] = direction_vector_mm[1] / np.linalg.norm(direction_vector_mm)
        direction_vector_mm[2] = direction_vector_mm[2] / np.linalg.norm(direction_vector_mm)
        self.direction_vector_norm = direction_vector_mm

    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm) -> dict:
        source_type = Tags.ILLUMINATION_TYPE_SLIT

        spacing = global_settings[Tags.SPACING_MM]

        device_position = [0, 0, 0]
        device_position[0] = (probe_position_mm[0]/spacing) + 0.5 - 0.5 * self.slit_vector_mm[0]/spacing
        device_position[1] = (probe_position_mm[1]/spacing) + 0.5 - 0.5 * self.slit_vector_mm[1]/spacing
        device_position[2] = (probe_position_mm[2]/spacing) + 0.5 - 0.5 * self.slit_vector_mm[2]/spacing

        self.logger.debug(device_position)

        source_direction = self.direction_vector_norm

        source_param1 = [self.slit_vector_mm[0]/spacing,
                         self.slit_vector_mm[1]/spacing,
                         self.slit_vector_mm[2]/spacing, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }