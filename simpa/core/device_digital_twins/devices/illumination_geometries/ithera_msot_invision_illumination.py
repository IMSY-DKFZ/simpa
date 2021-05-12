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

    def get_mcx_illuminator_definition(self, global_settings: Settings):
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :return:
        """
        source_type = Tags.ILLUMINATION_TYPE_MSOT_INVISION

        nx = global_settings[Tags.DIM_VOLUME_X_MM]
        ny = global_settings[Tags.DIM_VOLUME_Y_MM]
        nz = global_settings[Tags.DIM_VOLUME_Z_MM]
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

        source_position = [round(nx / (spacing * 2.0)) + 0.5 + np.sin(angle) * detector_iso_distance,
                           round(ny / (spacing * 2.0)) + 0.5 + det_sep_half,
                           round(nz / (spacing * 2.0)) + 0.5 + np.cos(angle) * detector_iso_distance]

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