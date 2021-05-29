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


class RSOMExplorerP50(PAIDeviceBase):
    """
    This class represents an approximation of the Raster-scanning Optoacoustic Mesoscopy (RSOM) device
    built by iThera Medical (Munich, Germany). Please refer to the companie's website for more information
    (https://www.ithera-medical.com/products/rsom-explorer-p50/).

    Since simulating thousands of individual forward modeling steps to obtain a single raster-scanned image
    is computationally not feasible, we approximate the process with a device design that has detection elements
    across the entire field of view. Because of this limitation we also need to approximate the light source
    with a homogeneous illumination across the field of view.

    The digital device is modeled based on the reported specifications of the RSOM Explorer P50 system.
    Technical details of the system can be found in the dissertation of Mathias Schwarz
    (https://mediatum.ub.tum.de/doc/1324031/1324031.pdf) and you can find more details on
    use cases of the device in the following literature sources::

        Yew, Yik Weng, et al. "Raster-scanning optoacoustic mesoscopy (RSOM) imaging
        as an objective disease severity tool in atopic dermatitis patients."
        Journal of the American Academy of Dermatology (2020).

        Hindelang, B., et al. "Non-invasive imaging in dermatology and the unique
        potential of raster-scan optoacoustic mesoscopy."
        Journal of the European Academy of Dermatology and Venereology
        33.6 (2019): 1051-1061.

    """

    def __init__(self, element_spacing_mm=0.02):
        self.center_frequency_Hz = float(50e6)
        self.bandwidth_percent = 100.0
        self.sampling_frequency_MHz = 500.0
        self.detector_element_length_mm = 1
        self.detector_element_width_mm = 1
        self.number_detector_elements = 1
        self.num_elements_x = 0
        self.num_elements_y = 0
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

        self.num_elements_x = np.round(global_settings[Tags.DIM_VOLUME_X_MM] / self.element_spacing_mm).astype(int)
        self.num_elements_y = np.round(global_settings[Tags.DIM_VOLUME_Y_MM] / self.element_spacing_mm).astype(int)
        self.number_detector_elements = self.num_elements_x * self.num_elements_y

        return global_settings

    def get_illuminator_definition(self, global_settings: Settings):
        pass

    def get_detector_element_positions_base_mm(self):
        detector_element_positions_mm = np.zeros((self.number_detector_elements, 3))
        for x in range(self.num_elements_x):
            for y in range(self.num_elements_y):
                detector_element_positions_mm[x + y*self.num_elements_x] = \
                    [(x - self.num_elements_x/2) * self.element_spacing_mm,
                     (y - self.num_elements_y/2) * self.element_spacing_mm,
                     0]
        return detector_element_positions_mm

    def get_detector_element_positions_accounting_for_device_position_mm(self, global_settings: Settings):

        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])

        detector_element_positions_mm = self.get_detector_element_positions_base_mm()

        if Tags.DIGITAL_DEVICE_POSITION in global_settings and global_settings[Tags.DIGITAL_DEVICE_POSITION]:
            device_position = np.asarray(global_settings[Tags.DIGITAL_DEVICE_POSITION])
        else:
            device_position = np.array([sizes_mm[0] / 2, sizes_mm[1] / 2, 0])

        return np.add(detector_element_positions_mm, device_position)

    def get_detector_element_orientations(self, global_settings: Settings):
        detector_element_orientations = np.zeros((self.number_detector_elements, 3))

        for x in range(self.num_elements_x):
            for y in range(self.num_elements_y):
                detector_element_orientations[x + y * self.num_elements_x] = [0, 0, 1]

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

    positions = device.get_detector_element_positions_accounting_for_device_position_mm(settings)
    detector_elements = device.get_detector_element_orientations(global_settings=settings)
    print(np.shape(positions))
    print(np.shape(detector_elements))
    # detector_elements[:, 1] = detector_elements[:, 1] + device.probe_height_mm
    positions = np.round(positions / device.element_spacing_mm).astype(int)

    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 1], marker='x')
    plt.show()
