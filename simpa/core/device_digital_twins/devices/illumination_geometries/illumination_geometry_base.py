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

from abc import abstractmethod
from simpa.core.device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase
from simpa.utils import Settings
import numpy as np


class IlluminationGeometryBase(DigitalDeviceTwinBase):
    """
    This class represents an illumination geometry.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_mcx_illuminator_definition(self, global_settings: Settings, probe_position_mm: np.ndarray) -> dict:
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :param probe_position_mm: the position of the probe in the volume
        :return: Dictionary that includes all parameters needed for mcx.
        """
        pass

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        return True

    def adjust_simulation_volume_and_settings(self, global_settings: Settings) -> Settings:
        return global_settings
