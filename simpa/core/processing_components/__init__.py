# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import ABC
from simpa.core import SimulationModule


class ProcessingComponent(SimulationModule, ABC):
    """
    Defines a simulation component, which can be used to pre- or post-process simulation data.
    """

    def __init__(self, global_settings, component_settings_key: str):
        """
        Initialises the ProcessingComponent.

        :param component_settings_key: The key where the component settings are stored in
        """
        super(ProcessingComponent, self).__init__(global_settings=global_settings)
        self.component_settings = global_settings[component_settings_key]
