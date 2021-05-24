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

from abc import abstractmethod
import numpy as np
from simpa.core.simulation_components import SimulationModule
from simpa.utils import Tags, Settings
from simpa.io_handling.io_hdf5 import save_hdf5
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.core.device_digital_twins import PhotoacousticDevice, DetectionGeometryBase


class AcousticForwardModelBaseAdapter(SimulationModule):
    """
    This method is the entry method for running an acoustic forward model.
    It is invoked in the *simpa.core.simulation.simulate* method, but can also be called
    individually for the purposes of performing acoustic forward modeling only or in a different context.

    The concrete will be chosen based on the::

        Tags.ACOUSTIC_MODEL

    tag in the settings dictionary.

    :param settings: The settings dictionary containing key-value pairs that determine the simulation.
        Here, it must contain the Tags.ACOUSTIC_MODEL tag and any tags that might be required by the specific
        acoustic model.
    :raises AssertionError: an assertion error is raised if the Tags.ACOUSTIC_MODEL tag is not given or
        points to an unknown acoustic forward model.
    """

    def __init__(self, global_settings: Settings):
        super(AcousticForwardModelBaseAdapter, self).__init__(global_settings=global_settings)
        self.component_settings = global_settings.get_acoustic_settings()

    @abstractmethod
    def forward_model(self, detection_geometry) -> np.ndarray:
        """
        This method performs the acoustic forward modeling given the initial pressure
        distribution and the acoustic tissue properties contained in the settings file.
        A deriving class needs to implement this method according to its model.

        :return: time series pressure data
        """
        pass

    def run(self, digital_device_twin):
        """
        Call this method to invoke the simulation process.

        :param digital_device_twin:
        :return: a numpy array containing the time series pressure data per detection element
        """

        self.logger.info("Simulating the acoustic forward process...")

        _device = None
        if isinstance(digital_device_twin, DetectionGeometryBase):
            _device = digital_device_twin
        elif isinstance(digital_device_twin, PhotoacousticDevice):
            _device = digital_device_twin.get_detection_geometry()
        else:
            raise TypeError(f"The optical forward modelling does not support devices of type {type(digital_device_twin)}")

        time_series_data = self.forward_model(_device)

        acoustic_output_path = generate_dict_path(Tags.TIME_SERIES_DATA, wavelength=self.global_settings[Tags.WAVELENGTH])

        save_hdf5(time_series_data, self.global_settings[Tags.SIMPA_OUTPUT_PATH], acoustic_output_path)

        self.logger.info("Simulating the acoustic forward process...[Done]")
