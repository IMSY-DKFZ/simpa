# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
from abc import abstractmethod

from simpa.core.device_digital_twins import DigitalDeviceTwinBase
from simpa.log import Logger
from simpa.utils import Settings
from simpa.utils.processing_device import get_processing_device

class PipelineModule:
    """
    Defines a pipeline module (either simulation or processing module) that implements a run method and can be called by running the pipeline's simulate method.
    """
    def __init__(self, global_settings: Settings):
        """
         :param global_settings: The SIMPA settings dictionary
         :type global_settings: Settings
        """
        self.logger = Logger()
        self.global_settings = global_settings
        self.torch_device = get_processing_device(self.global_settings)

    @abstractmethod
    def run(self, digital_device_twin: DigitalDeviceTwinBase):
        """
        Executes the respective simulation module

        :param digital_device_twin: The digital twin that can be used by the digital device_twin.
        """
        pass
  