# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import abstractmethod

from simpa.core import PipelineModule
from simpa.utils import Settings

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

        self.component_settings = self.get_default_component_settings()
        self.component_settings.update(self.get_user_set_component_settings()) # adds and if necessary overrides default component settings by user given settings
        if self.component_settings is None:
            raise ValueError("The component settings should not be None at this point")

    @abstractmethod
    def get_user_set_component_settings(self) -> Settings:
        """
        :return: Loads component settings corresponding to this simulation component set by the user
        """
        pass

    @abstractmethod
    def get_default_component_settings(self) -> Settings:
        """Return the default component settings corresponding to this simulation component

        :return: default component settings of this component
        :rtype: Settings
        """
        pass

        

    
