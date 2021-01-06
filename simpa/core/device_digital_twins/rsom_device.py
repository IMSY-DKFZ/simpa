# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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

from simpa.core.device_digital_twins.pai_devices import PAIDeviceBase
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SegmentationClasses
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.structure_library import HorizontalLayerStructure, Background
from simpa.utils.deformation_manager import get_functional_from_deformation_settings
import numpy as np
from simpa.utils import create_deformation_settings


class RSOMExplorerP50(PAIDeviceBase):

    def __init__(self, element_spacing_mm=0.04):
        self.center_frequency_Hz = 50e6
        self.bandwidth_percent = 100
        self.sampling_frequency_MHz = 500
        self.detector_element_length_mm = 1
        self.detector_element_width_mm = 1
        self.number_detector_elements = 1
        self.element_spacing_mm = element_spacing_mm

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        return True  # Realistically, every volume can be imaged with the RSOM system.

    def adjust_simulation_volume_and_settings(self, global_settings: Settings):
        global_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ] = self.center_frequency_Hz
        global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = self.sampling_frequency_MHz
        global_settings[Tags.SENSOR_BANDWIDTH_PERCENT] = self.bandwidth_percent
        self.detector_element_length_mm = global_settings[Tags.SPACING_MM]
        self.detector_element_width_mm = global_settings[Tags.SPACING_MM]
        if self.element_spacing_mm < global_settings[Tags.SPACING_MM]:
            self.element_spacing_mm = global_settings[Tags.SPACING_MM]
        return global_settings

    def get_illuminator_definition(self):
        pass

    def get_detector_element_positions_mm(self, global_settings: Settings):

        num_elements_x = np.round(global_settings[Tags.DIM_VOLUME_X_MM] / self.element_spacing_mm).astype(int)
        num_elements_y = np.round(global_settings[Tags.DIM_VOLUME_Y_MM] / self.element_spacing_mm).astype(int)
        self.number_detector_elements = num_elements_x * num_elements_y

        detector_element_positions_mm = np.zeros((self.number_detector_elements, 3))

        for x in range(num_elements_x):
            for y in range(num_elements_y):
                detector_element_positions_mm[x + y*num_elements_x] = [x * self.element_spacing_mm,
                                                                       y * self.element_spacing_mm,
                                                                       0]

        return detector_element_positions_mm

    def get_detector_element_orientations(self, global_settings: Settings):
        num_elements_x = np.round(global_settings[Tags.DIM_VOLUME_X_MM] / self.element_spacing_mm).astype(int)
        num_elements_y = np.round(global_settings[Tags.DIM_VOLUME_Y_MM] / self.element_spacing_mm).astype(int)
        self.number_detector_elements = num_elements_x * num_elements_y

        detector_element_orientations = np.zeros((self.number_detector_elements, 3))

        for x in range(num_elements_x):
            for y in range(num_elements_y):
                detector_element_orientations[x + y * num_elements_x] = [0, 0, 1]

        return detector_element_orientations


if __name__ == "__main__":
    device = RSOMExplorerP50()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 12
    settings[Tags.DIM_VOLUME_Y_MM] = 12
    settings[Tags.DIM_VOLUME_Z_MM] = 2.8
    settings[Tags.SPACING_MM] = 0.02
    settings[Tags.STRUCTURES] = {}
    # settings[Tags.DIGITAL_DEVICE_POSITION] = [50, 50, 50]
    settings = device.adjust_simulation_volume_and_settings(settings)
    # print(settings[Tags.DIM_VOLUME_Z_MM])

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))
    print(x_dim, z_dim)

    positions = device.get_detector_element_positions_mm(settings)
    detector_elements = device.get_detector_element_orientations(global_settings=settings)
    print(np.shape(positions))
    print(np.shape(detector_elements))
    # detector_elements[:, 1] = detector_elements[:, 1] + device.probe_height_mm
    positions = np.round(positions / device.element_spacing_mm).astype(int)

    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 1], marker='x')
    plt.show()
