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

from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils import Settings, Tags


class CurvedArrayDetectionGeometry(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a curved detection geometry
    """

    def __init__(self, pitch_mm=0.5,
                 radius_mm=40,
                 number_detector_elements=256,
                 detector_element_width_mm=0.24,
                 detector_element_length_mm=13,
                 center_frequency_hz=3.96e6,
                 bandwidth_percent=55,
                 sampling_frequency_mhz=40,
                 angular_origin_offset=np.pi,
                 device_position_mm=None):
        """
        :param angular_origin_offset: TODO
        """

        super().__init__(number_detector_elements=number_detector_elements,
                         detector_element_width_mm=detector_element_width_mm,
                         detector_element_length_mm=detector_element_length_mm,
                         center_frequency_hz=center_frequency_hz,
                         bandwidth_percent=bandwidth_percent,
                         sampling_frequency_mhz=sampling_frequency_mhz,
                         probe_width_mm=2 * np.sin(pitch_mm / radius_mm * 128) * radius_mm,
                         device_position_mm=device_position_mm)

        self.pitch_mm = pitch_mm
        self.radius_mm = radius_mm
        self.angular_origin_offset = angular_origin_offset

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        return np.asarray([-self.probe_width_mm/2,
                           self.probe_width_mm/2,
                           0, 0,
                           0, 100])

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.DIM_VOLUME_Z_MM] <= (self.radius_mm + 1):
            self.logger.error("Volume z dimension is too small to encompass the device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format((self.radius_mm + 1),
                                      global_settings[Tags.DIM_VOLUME_Z_MM]))
            return False
        if global_settings[Tags.DIM_VOLUME_X_MM] <= self.probe_width_mm:
            self.logger.error("Volume x dimension is too small to encompass MSOT device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format(self.probe_width_mm, global_settings[Tags.DIM_VOLUME_X_MM]))
            return False
        return True

    def get_detector_element_positions_base_mm(self) -> np.ndarray:

        pitch_angle = self.pitch_mm / self.radius_mm
        self.logger.debug(f"pitch angle: {pitch_angle}")
        detector_radius = self.radius_mm
        detector_positions = np.zeros((self.number_detector_elements, 3))
        det_elements = np.arange(-int(self.number_detector_elements / 2) + 0.5,
                                 int(self.number_detector_elements / 2) + 0.5)
        detector_positions[:, 0] = np.sin(pitch_angle * det_elements - self.angular_origin_offset) * detector_radius
        detector_positions[:, 2] = np.cos(pitch_angle * det_elements - self.angular_origin_offset) * detector_radius

        return detector_positions

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_positions = self.get_detector_element_positions_base_mm()
        detector_orientations = np.subtract(0, detector_positions)
        norm = np.linalg.norm(detector_orientations, axis=-1)
        for dim in range(3):
            detector_orientations[:, dim] = detector_orientations[:, dim] / norm
        return detector_orientations
