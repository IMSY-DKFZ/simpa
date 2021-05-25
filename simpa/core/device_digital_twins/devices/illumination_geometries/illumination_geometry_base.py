"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

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
