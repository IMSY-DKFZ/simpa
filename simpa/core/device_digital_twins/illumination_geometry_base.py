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

from abc import abstractmethod, ABC
from simpa.core.device_digital_twins.digital_device_base import DigitalDeviceBase
from simpa.utils import Settings, Tags
import numpy as np
from simpa.log import Logger


class IlluminationGeometryBase(DigitalDeviceBase):
    """
    This class represents an illumination geometry.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_illuminator_definition(self, global_settings: Settings) -> dict:
        """
        Defines the illumination geometry of the device in the settings dictionary.
        """
        pass


class SlitIlluminationGeometry(IlluminationGeometryBase, ABC):
    """
    This class represents a slit illumination geometry.
    The device position is defined as the middle of the slit.
    """
    def __init__(self, slit_vector_mm: np.ndarray, direction: np.ndarray):
        """
        Initializes a slit illumination source.
        :param slit_vector_mm: Defines the slit in vector form. For example a slit along the x-axis with length 5mm
        would be defined as np.array([5, 0, 0])
        :param direction: Direction vector in which the slit illuminates.
        """
        super().__init__()

        self.slit_vector_mm = slit_vector_mm
        self.direction = direction / np.linalg.norm(direction)

    def get_illuminator_definition(self, global_settings: Settings) -> dict:
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :return: Dictionary that includes all parameters needed for mcx.
        """
        source_type = Tags.ILLUMINATION_TYPE_SLIT

        nx = global_settings[Tags.DIM_VOLUME_X_MM]
        ny = global_settings[Tags.DIM_VOLUME_Y_MM]
        nz = global_settings[Tags.DIM_VOLUME_Z_MM]
        spacing = global_settings[Tags.SPACING_MM]

        if Tags.DIGITAL_DEVICE_POSITION in global_settings:
            device_position = global_settings[Tags.DIGITAL_DEVICE_POSITION]
        else:
            device_position = [round(nx / (spacing * 2.0)) + 0.5,
                               round(ny / (spacing * 2.0)) + 0.5,
                               spacing]  # The z-position

        device_position = np.subtract(device_position, 0.5*self.slit_vector_mm)

        source_direction = [0, 0, 1]

        source_param1 = [self.slit_vector_mm[0]/spacing, self.slit_vector_mm[1]/spacing,
                         self.slit_vector_mm[2]/spacing, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": device_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }
