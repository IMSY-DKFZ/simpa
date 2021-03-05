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
import numpy as np

from simpa.core.device_digital_twins.pai_devices import PAIDeviceBase
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags


class InVision256TF(PAIDeviceBase):
    """
    This class represents a digital twin of the InVision 256-TF, manufactured by iThera Medical, Munich, Germany
    (https://www.ithera-medical.com/products/msot-invision/). It is based on the real specifications of the device, but
    due to the limitations of the possibilities how to represent a device in the software frameworks,
    constitutes only an approximation.

    Some important publications that showcase the use cases of the InVision series devices are::

        Joseph, James, et al. "Evaluation of precision in optoacoustic tomography
        for preclinical imaging in living subjects."
        Journal of Nuclear Medicine 58.5 (2017): 807-814.

        Merčep, Elena, et al. "Whole-body live mouse imaging by hybrid
        reflection-mode ultrasound and optoacoustic tomography."
        Optics letters 40.20 (2015): 4643-4646.

    """

    def __init__(self):
        self.pitch_mm = 0.74
        self.radius_mm = 40
        self.number_detector_elements = 256
        self.detector_element_width_mm = 0.64
        self.detector_element_length_mm = 15
        self.center_frequency_Hz = 5e6
        self.bandwidth_percent = 55
        self.sampling_frequency_MHz = 40
        self.focus_in_field_of_view_mm = np.array([0, 0, 4])

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        pass

    def adjust_simulation_volume_and_settings(self, global_settings: Settings) -> Settings:
        global_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ] = self.center_frequency_Hz
        global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = self.sampling_frequency_MHz
        global_settings[Tags.SENSOR_BANDWIDTH_PERCENT] = self.bandwidth_percent
        return global_settings

    def get_illuminator_definition(self, global_settings: Settings):
        pass

    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        pitch_angle = self.pitch_mm / self.radius_mm
        print("pitch angle: ", pitch_angle)
        detector_radius = self.radius_mm
        detector_positions = np.zeros((self.number_detector_elements, 3))
        # go from -127.5, -126.5, ..., 0, .., 126.5, 177.5 instead of between -128 and 127
        det_elements = np.arange(-int(self.number_detector_elements / 2) + 0.5,
                                 int(self.number_detector_elements / 2) + 0.5)
        detector_positions[:, 0] = np.sin(pitch_angle * det_elements - np.pi/2) * detector_radius
        detector_positions[:, 2] = np.cos(pitch_angle * det_elements - np.pi/2) * detector_radius

        return detector_positions

    def get_detector_element_positions_accounting_for_device_position_mm(self, global_settings: Settings) -> np.ndarray:
        abstract_element_positions = self.get_detector_element_positions_base_mm()

        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])

        if Tags.DIGITAL_DEVICE_POSITION in global_settings and global_settings[Tags.DIGITAL_DEVICE_POSITION]:
            device_position = np.asarray(global_settings[Tags.DIGITAL_DEVICE_POSITION])
        else:
            device_position = np.asarray([sizes_mm[0] / 2, sizes_mm[1] / 2, sizes_mm[2] / 2])

        return np.add(abstract_element_positions, device_position)

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_positions = self.get_detector_element_positions_base_mm()
        detector_orientations = np.subtract(self.focus_in_field_of_view_mm, detector_positions)
        norm = np.linalg.norm(detector_orientations, axis=-1)
        for dim in range(3):
            detector_orientations[:, dim] = detector_orientations[:, dim] / norm
        return detector_orientations


if __name__ == "__main__":
    device = InVision256TF()
    settings = Settings()
    settings[Tags.DIM_VOLUME_X_MM] = 100
    settings[Tags.DIM_VOLUME_Y_MM] = 20
    settings[Tags.DIM_VOLUME_Z_MM] = 100
    settings[Tags.SPACING_MM] = 0.5
    settings[Tags.STRUCTURES] = {}
    settings[Tags.VOLUME_CREATOR] = Tags.VOLUME_CREATOR_VERSATILE
    settings = device.adjust_simulation_volume_and_settings(settings)

    x_dim = int(round(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]))
    z_dim = int(round(settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM]))
    print(x_dim, z_dim)

    positions = device.get_detector_element_positions_accounting_for_device_position_mm(settings)
    print("Positions in mm:", positions)
    detector_elements = device.get_detector_element_orientations(global_settings=settings)
    print(np.shape(positions[:, 0]))
    print(np.shape(positions[:, 2]))
    print(np.shape(detector_elements[:, 0]))
    print(np.shape(detector_elements[:, 2]))
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.show()
