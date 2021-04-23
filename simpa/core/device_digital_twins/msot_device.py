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

from simpa.core.device_digital_twins.pai_device_base import PAIDeviceBase
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
import numpy as np


class MSOTAcuityEcho(PAIDeviceBase):
    """
    This class represents a digital twin of the MSOT Acuity Echo, manufactured by iThera Medical, Munich, Germany
    (https://www.ithera-medical.com/products/msot-acuity/). It is based on the real specifications of the device, but
    due to the limitations of the possibilities how to represent a device in the software frameworks,
    constitutes only an approximation.

    Some important publications that showcase the use cases of the MSOT Acuity and Acuity Echo device are::

        Regensburger, Adrian P., et al. "Detection of collagens by multispectral optoacoustic
        tomography as an imaging biomarker for Duchenne muscular dystrophy."
        Nature Medicine 25.12 (2019): 1905-1915.

        Knieling, Ferdinand, et al. "Multispectral Optoacoustic Tomography for Assessment of
        Crohn's Disease Activity."
        The New England journal of medicine 376.13 (2017): 1292.

    """

    def __init__(self):
        super().__init__()
        self.pitch_mm = 0.34
        self.radius_mm = 40
        self.number_detector_elements = 256
        self.detector_element_width_mm = 0.24
        self.detector_element_length_mm = 13
        self.center_frequency_Hz = 3.96e6
        self.bandwidth_percent = 55
        self.sampling_frequency_MHz = 40
        self.probe_height_mm = 43.2
        self.probe_width_mm = 2*np.sin(self.pitch_mm/self.radius_mm * 128)*self.radius_mm
        self.mediprene_membrane_height_mm = 1
        self.focus_in_field_of_view_mm = np.array([0, 0, 8])

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.VOLUME_CREATOR] != Tags.VOLUME_CREATOR_VERSATILE:
            if global_settings[Tags.DIM_VOLUME_Z_MM] <= (self.probe_height_mm + self.mediprene_membrane_height_mm + 1):
                self.logger.error("Volume z dimension is too small to encompass MSOT device in simulation!"
                                     "Must be at least {} mm but was {} mm"
                                     .format((self.probe_height_mm + self.mediprene_membrane_height_mm + 1),
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
        global_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ] = self.center_frequency_Hz
        global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = self.sampling_frequency_MHz
        global_settings[Tags.SENSOR_BANDWIDTH_PERCENT] = self.bandwidth_percent
        return global_settings

    def get_illuminator_definition(self, global_settings: Settings):
        """
        IMPORTANT: This method creates a dictionary that contains tags as they are expected for the
        mcx simulation tool to represent the illumination geometry of this device.

        :param global_settings: The global_settings instance containing the simulation instructions
        :return:
        """
        source_type = Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO

        nx = global_settings[Tags.DIM_VOLUME_X_MM]
        ny = global_settings[Tags.DIM_VOLUME_Y_MM]
        nz = global_settings[Tags.DIM_VOLUME_Z_MM]
        spacing = global_settings[Tags.SPACING_MM]

        source_position = [round(nx / (spacing * 2.0)) + 0.5,
                           round(ny / (spacing * 2.0) - 17.81 / spacing) + 0.5,
                           spacing]     # The z-position

        source_direction = [0, 0.381070, 0.9245460]

        source_param1 = [30 / spacing, 0, 0, 0]

        source_param2 = [0, 0, 0, 0]

        return {
            "Type": source_type,
            "Pos": source_position,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def get_detector_element_positions_base_mm(self) -> np.ndarray:

        pitch_angle = self.pitch_mm / self.radius_mm
        self.logger.debug(f"pitch angle: {pitch_angle}")
        detector_radius = self.radius_mm

        # if distortion is not None:
        #     focus[0] -= np.round(distortion[1] / (2 * global_settings[Tags.SPACING_MM]))

        detector_positions = np.zeros((self.number_detector_elements, 3))
        # go from -127.5, -126.5, ..., 0, .., 126.5, 177.5 instead of between -128 and 127
        det_elements = np.arange(-int(self.number_detector_elements / 2) + 0.5,
                                 int(self.number_detector_elements / 2) + 0.5)
        detector_positions[:, 0] = self.focus_in_field_of_view_mm[0] \
            + np.sin(pitch_angle * det_elements) * detector_radius
        detector_positions[:, 2] = self.focus_in_field_of_view_mm[2] \
            - np.sqrt(detector_radius ** 2 - (np.sin(pitch_angle*det_elements) * detector_radius) ** 2)

        return detector_positions

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_positions = self.get_detector_element_positions_base_mm()
        detector_orientations = np.subtract(self.focus_in_field_of_view_mm, detector_positions)
        norm = np.linalg.norm(detector_orientations, axis=-1)
        for dim in range(3):
            detector_orientations[:, dim] = detector_orientations[:, dim]/norm
        return detector_orientations

    def get_default_probe_position(self, global_settings: Settings) -> np.ndarray:
        sizes_mm = np.asarray([global_settings[Tags.DIM_VOLUME_X_MM],
                               global_settings[Tags.DIM_VOLUME_Y_MM],
                               global_settings[Tags.DIM_VOLUME_Z_MM]])
        return np.array([sizes_mm[0] / 2, sizes_mm[1] / 2, self.probe_height_mm])


if __name__ == "__main__":
    device = MSOTAcuityEcho()
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
