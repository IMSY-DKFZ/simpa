"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np

from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils import Settings, Tags


class SingleDetectionElement(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a random detection geometry of omnidirectional elements.
    The origin for this device is the center of the random array.

    """

    def __init__(self, device_position_mm: np.ndarray = None):
        """

        """
        super().__init__(number_detector_elements=1,
                         detector_element_width_mm=1,
                         detector_element_length_mm=1,
                         center_frequency_hz=1,
                         bandwidth_percent=500,
                         sampling_frequency_mhz=10,
                         device_position_mm=device_position_mm,
                         field_of_view_extent_mm=np.asarray([-10, 10, 0, 0, -10, 10]))

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        return True

    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        return np.copy(np.zeros((1, 3)))

    def get_detector_element_orientations(self) -> np.ndarray:
        detector_orientations = np.zeros((self.number_detector_elements, 3))
        detector_orientations[:, 2] = 1
        return detector_orientations

