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
from abc import abstractmethod, ABC


class SimulationModule(ABC):
    """
    Defines a simulation module that is callable via the SIMPA core.simulation.simulate method.
    """

    def __init__(self):
        """
        Initialises the SimulationModule.
        """
        self.logger = Logger()

    @abstractmethod
    def run(self, global_settings: Settings):
        """
        Executes the respective simulation module

        :param global_settings: The global SIMPA settings dictionary
        """
        pass


class ProcessingComponent(SimulationModule, ABC):
    """
    Defines a simulation component, which can be used to pre- or post-process simulation data.
    """

    def __init__(self, component_settings_key: str):
        """
        Initialises the ProcessingComponent.

        :param component_settings_key: The key where the component settings are stored in
        """
        super().__init__()
        self.component_settings_key = component_settings_key


