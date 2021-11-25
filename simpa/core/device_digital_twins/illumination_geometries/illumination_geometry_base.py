# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod
from simpa.core.device_digital_twins.digital_device_twin_base import DigitalDeviceTwinBase
from simpa.utils import Settings
from numpy import ndarray


class IlluminationGeometryBase(DigitalDeviceTwinBase):
    """
    This class represents an illumination geometry.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_mcx_illuminator_definition(self, global_settings, probe_position_mm) -> dict:
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :type global_settings: Settings

        :param probe_position_mm: the position of the probe in the volume
        :type probe_position_mm: ndarray

        :return: Dictionary that includes all parameters needed for mcx.
        :rtype: dict
        """
        pass

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        return True

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings) -> Settings:
        return global_settings
