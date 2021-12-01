# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
from abc import abstractmethod

from simpa.core.device_digital_twins import DigitalDeviceTwinBase
from simpa.log import Logger
from simpa.utils import Settings


class SimulationModule:
    """
    Defines a simulation module that is callable via the SIMPA core.simulation.simulate method.
    """

    def __init__(self, global_settings):
        """
         :param global_settings: The SIMPA settings dictionary
         :type global_settings: Settings
        """
        self.logger = Logger()
        self.global_settings = global_settings

    @abstractmethod
    def run(self, digital_device_twin: DigitalDeviceTwinBase):
        """
        Executes the respective simulation module

        :param digital_device_twin: The digital twin that can be used by the digital device_twin.
        """
        pass
