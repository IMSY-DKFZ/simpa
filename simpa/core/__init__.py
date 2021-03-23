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

from simpa.log import Logger
from simpa.utils import Settings
from abc import abstractmethod


class SimulationComponent:
    """
    Defines a simulation component that is callable via the SIMPA core.simulation.simulate method.
    """

    def __init__(self, global_settings: Settings, component_settings_key: str):
        """
        Initialises the SimulationComponent given the global settings dictionary.
         :param global_settings: The SIMPA settings dictionary
         :param component_settings_key: the key to lookup the specific settings for this Component.
        """
        self.logger = Logger()
        self.global_settings = global_settings
        self.component_settings = global_settings[component_settings_key]

    @abstractmethod
    def run(self):
        """
        Executes the respective simulation component
        """
        pass




