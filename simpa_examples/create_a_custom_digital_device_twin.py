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
# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simpa.core.device_digital_twins.digital_device_base import PhotoacousticDevice
from simpa.core.device_digital_twins.illumination_geometry_base import SlitIlluminationGeometry
from simpa.core.device_digital_twins.detection_geometry_base import LinearDetector

from simpa.utils import Settings, Tags
import numpy as np


class ExampleDeviceSlitIlluminationLinearDetector(PhotoacousticDevice):
    """
    This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

    """

    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        return np.asarray([0, 0, 0])

    def __init__(self):
        super().__init__()
        self.set_detection_geometry(LinearDetector())
        self.add_illumination_geometry(SlitIlluminationGeometry())


if __name__ == "__main__":
    device = ExampleDeviceSlitIlluminationLinearDetector()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 20
    settings[Tags.DIM_VOLUME_Y_MM] = 50
    settings[Tags.DIM_VOLUME_Z_MM] = 20
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}
    # settings[Tags.DIGITAL_DEVICE_POSITION] = [50, 50, 50]
    settings = device.adjust_simulation_volume_and_settings(settings)

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))

    positions = device.get_detection_geometry().get_detector_element_positions_accounting_for_device_position_mm(settings)
    detector_elements = device.get_detection_geometry().get_detector_element_orientations(global_settings=settings)
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
