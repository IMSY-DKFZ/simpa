"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

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
