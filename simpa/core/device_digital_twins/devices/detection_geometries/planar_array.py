"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np

from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils import Settings, Tags


class PlanarArrayDetectionGeometry(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a linear detection geometry.

    """

    def __init__(self, pitch_mm=0.5,
                 number_detector_elements_x=100,
                 number_detector_elements_y=100,
                 detector_element_width_mm=0.24,
                 detector_element_length_mm=0.5,
                 center_frequency_hz=3.96e6,
                 bandwidth_percent=55,
                 sampling_frequency_mhz=40,
                 device_position_mm: np.ndarray = None):
        super().__init__(number_detector_elements=number_detector_elements_x * number_detector_elements_y,
                         detector_element_width_mm=detector_element_width_mm,
                         detector_element_length_mm=detector_element_length_mm,
                         center_frequency_hz=center_frequency_hz,
                         bandwidth_percent=bandwidth_percent,
                         sampling_frequency_mhz=sampling_frequency_mhz,
                         probe_width_mm=number_detector_elements_x * pitch_mm,
                         device_position_mm=device_position_mm)
        self.pitch_mm = pitch_mm
        self.number_detector_elements_x = number_detector_elements_x
        self.number_detector_elements_y = number_detector_elements_y
        self.probe_depth_mm = number_detector_elements_y * pitch_mm

    def get_field_of_view_extent_mm(self) -> np.ndarray:
        return np.asarray([-self.number_detector_elements_x*self.pitch_mm/2,
                           self.number_detector_elements_x*self.pitch_mm/2,
                           -self.number_detector_elements_y * self.pitch_mm / 2,
                           self.number_detector_elements_y * self.pitch_mm / 2,
                           0, 100])

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.DIM_VOLUME_X_MM] <= self.probe_width_mm:
            self.logger.error(f"Volume x dimension is too small to encompass RSOM device in simulation!"
                              f"Must be at least {self.probe_width_mm} mm but "
                              f"was {global_settings[Tags.DIM_VOLUME_X_MM]} mm")
            return False
        if global_settings[Tags.DIM_VOLUME_Y_MM] <= self.probe_depth_mm:
            self.logger.error(f"Volume y dimension is too small to encompass RSOM device in simulation!"
                              f"Must be at least {self.probe_depth_mm} mm but "
                              f"was {global_settings[Tags.DIM_VOLUME_X_MM]} mm")
            return False
        return True

    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        detector_element_positions_mm = np.zeros((self.number_detector_elements, 3))
        for x in range(self.number_detector_elements_x):
            for y in range(self.number_detector_elements_y):
                detector_element_positions_mm[x + y*self.number_detector_elements_x] = \
                    [(x - self.number_detector_elements_x/2) * self.pitch_mm,
                     (y - self.number_detector_elements_y/2) * self.pitch_mm,
                     0]
        return detector_element_positions_mm

    def get_detector_element_orientations(self, global_settings: Settings) -> np.ndarray:
        detector_element_orientations = np.zeros((self.number_detector_elements, 3))
        detector_element_orientations[:, 2] = 1
        return detector_element_orientations
