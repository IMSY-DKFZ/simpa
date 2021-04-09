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

from simpa.core.device_digital_twins.pai_devices import PAIDeviceBase
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags
import numpy as np


class SlitIlluminationLinearDetector(PAIDeviceBase):
    """
    This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

    """

    def __init__(self):
        super().__init__()
        self.pitch_mm = 0.01
        self.number_detector_elements = 256
        self.detector_element_width_mm = 0.24
        self.detector_element_length_mm = 13
        self.center_frequency_Hz = 3.96e6
        self.bandwidth_percent = 55
        self.sampling_frequency_MHz = 40
        self.probe_height_mm = 43.2
        self.probe_width_mm = self.number_detector_elements * self.pitch_mm

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.VOLUME_CREATOR] != Tags.VOLUME_CREATOR_VERSATILE:
            if global_settings[Tags.DIM_VOLUME_Z_MM] <= (self.probe_height_mm + 1):
                self.logger.error("Volume z dimension is too small to encompass the device in simulation!"
                                  "Must be at least {} mm but was {} mm"
                                  .format((self.probe_height_mm + 1),
                                          global_settings[Tags.DIM_VOLUME_Z_MM]))
                return False
            if global_settings[Tags.DIM_VOLUME_X_MM] <= self.probe_width_mm:
                self.logger.error("Volume x dimension is too small to encompass MSOT device in simulation!"
                                     "Must be at least {} mm but was {} mm"
                                     .format(self.probe_width_mm, global_settings[Tags.DIM_VOLUME_X_MM]))
                return False

        global_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ] = self.center_frequency_Hz
        global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = self.sampling_frequency_MHz
        global_settings[Tags.SENSOR_BANDWIDTH_PERCENT] = self.bandwidth_percent

        return True

    def adjust_simulation_volume_and_settings(self, global_settings: Settings):
        return global_settings

    def get_illuminator_definition(self, global_settings: Settings):
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :return:
        """
        source_type = Tags.ILLUMINATION_TYPE_SLIT

        nx = global_settings[Tags.DIM_VOLUME_X_MM]
        ny = global_settings[Tags.DIM_VOLUME_Y_MM]
        nz = global_settings[Tags.DIM_VOLUME_Z_MM]
        spacing = global_settings[Tags.SPACING_MM]

        source_position = [round(nx / (spacing * 2.0)) + 0.5,
                           round(ny / (spacing * 2.0)) + 0.5,
                           spacing]     # The z-position

        source_direction = [0, 0, 1]

        source_param1 = [10 / spacing, 0, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": source_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def get_detector_element_positions_base_mm(self) -> np.ndarray:

        detector_positions = np.zeros((self.number_detector_elements, 3))

        det_elements = np.arange(-int(self.number_detector_elements / 2),
                                 int(self.number_detector_elements / 2))
        if self.number_detector_elements % 2 == 0:
            # eg for 256 elements: go from -127.5, -126.5, ..., 0, .., 126.5, 177.5 instead of between -128 and 127
            det_elements = np.add(det_elements, 0.5)

        detector_positions[:, 0] = det_elements

        return detector_positions

    def get_detector_element_positions_accounting_for_device_position_mm(self, global_settings: Settings) -> np.ndarray:
        abstract_element_positions = self.get_detector_element_positions_base_mm()

        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])

        if Tags.DIGITAL_DEVICE_POSITION in global_settings and global_settings[Tags.DIGITAL_DEVICE_POSITION]:
            device_position = np.asarray(global_settings[Tags.DIGITAL_DEVICE_POSITION])
        else:
            device_position = np.array([sizes_mm[0] / 2, sizes_mm[1] / 2, self.probe_height_mm])

        return np.add(abstract_element_positions, device_position)

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_orientations = np.zeros((self.number_detector_elements, 3)) - 1
        return detector_orientations


if __name__ == "__main__":
    device = SlitIlluminationLinearDetector()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 20
    settings[Tags.DIM_VOLUME_Y_MM] = 50
    settings[Tags.DIM_VOLUME_Z_MM] = 20
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}
    settings[Tags.VOLUME_CREATOR] = Tags.VOLUME_CREATOR_VERSATILE
    # settings[Tags.DIGITAL_DEVICE_POSITION] = [50, 50, 50]
    settings = device.adjust_simulation_volume_and_settings(settings)

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))

    positions = device.get_detector_element_positions_accounting_for_device_position_mm(settings)
    detector_elements = device.get_detector_element_orientations(global_settings=settings)
    # detector_elements[:, 1] = detector_elements[:, 1] + device.probe_height_mm
    positions = np.round(positions/settings[Tags.SPACING_MM]).astype(int)
    position_map = np.zeros((x_dim, z_dim))
    position_map[positions[:, 0], positions[:, 2]] = 1
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
    # plt.imshow(map)
    # plt.show()
