# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod

from simpa.core import PipelineModule
from simpa.utils import Settings

class SimulationModule(PipelineModule):
    """
    Defines a simulation module that is a step in the simulation pipeline.
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
        
    def before_running(self):
        """
        Called before running the run method from PipelineModule
        """
        self.logger.info(f"Starts running {self.__class__.__name__}")

    def after_running(self):
        """
        Called after running the run method from PipelineModule
        """
        self.logger.info(f"Ended running {self.__class__.__name__}")

    
