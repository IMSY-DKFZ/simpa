# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod

from simpa.core import PipelineModule
from simpa.utils import Settings, Tags
from typing import List

class SimulationModule(PipelineModule):
    """
    Defines a simulation module that is a step in the simulation pipeline. 
    Each simulation module can only be one of Volume Creation, Light Propagation Modeling, Acoustic Wave Propagation Modeling, Image Reconstruction.
    """

    def __init__(self, global_settings: Settings):
        """
         :param global_settings: The SIMPA settings dictionary
         :type global_settings: Settings
        """
        super(SimulationModule, self).__init__(global_settings=global_settings)
        self.component_settings = self.load_component_settings()
        if self.component_settings is None:
            raise ValueError("The component settings should not be None at this point")

    @abstractmethod
    def load_component_settings(self) -> Settings:
        """
        :return: Loads component settings corresponding to this simulation component
        """
        pass

    def get_additional_flags(self) -> List[str]:
        """Reads the list of additional flags from the corresponding component settings Tags.ADDITIONAL_FLAGS

        :return: List[str]: list of additional flags 
        """
        cmd = []
        if Tags.ADDITIONAL_FLAGS in self.component_settings:
            for flag in self.component_settings[Tags.ADDITIONAL_FLAGS]:
                cmd.append(str(flag))
        return cmd