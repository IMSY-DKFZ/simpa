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

from simpa.utils import Tags
from simpa.core.device_digital_twins.devices.detection_geometries.detection_geometry_base import DetectionGeometryBase
from simpa.core.device_digital_twins.digital_device_twin_base import PhotoacousticDevice
from simpa.io_handling.io_hdf5 import load_data_field
from abc import abstractmethod
from simpa.core.simulation_components import SimulationModule
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import save_hdf5
import numpy as np
from simpa.utils import Settings


class ReconstructionAdapterBase(SimulationModule):
    """
    This class is the main entry point to perform image reconstruction using the SIMPA toolkit.
    All information necessary for the respective reconstruction method must be contained in the
    respective settings dictionary.
    """

    def __init__(self, global_settings: Settings):
        super(ReconstructionAdapterBase, self).__init__(global_settings=global_settings)
        self.component_settings = global_settings.get_reconstruction_settings()

    @abstractmethod
    def reconstruction_algorithm(self, time_series_sensor_data, detection_geometry) -> np.ndarray:
        """
        A deriving class needs to implement this method according to its model.

        :param time_series_sensor_data: the time series sensor data
        :param detection_geometry:
        :return: a reconstructed photoacoustic image
        """
        pass

    def run(self, device):
        self.logger.info("Performing reconstruction...")

        time_series_sensor_data = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                                                  Tags.TIME_SERIES_DATA, self.global_settings[Tags.WAVELENGTH])

        _device = None
        if isinstance(device, DetectionGeometryBase):
            _device = device
        elif isinstance(device, PhotoacousticDevice):
            _device = device.get_detection_geometry()
        else:
            raise TypeError(f"Type {type(device)} is not supported for performing image reconstruction.")
        reconstruction = self.reconstruction_algorithm(time_series_sensor_data, _device)

        reconstruction_output_path = generate_dict_path(Tags.RECONSTRUCTED_DATA, self.global_settings[Tags.WAVELENGTH])

        save_hdf5({Tags.RECONSTRUCTED_DATA: reconstruction}, self.global_settings[Tags.SIMPA_OUTPUT_PATH],
                  reconstruction_output_path)

        self.logger.info("Performing reconstruction...[Done]")
