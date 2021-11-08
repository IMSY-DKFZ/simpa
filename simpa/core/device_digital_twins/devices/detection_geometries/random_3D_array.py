"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np

from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils import Settings, Tags


class Random3DArrayDetectionGeometry(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a random detection geometry of omnidirectional elements.
    The origin for this device is the center of the random array.

    """

    def __init__(self,
                 extent_mm: float = 5.0,
                 number_detector_elements: int = 100,
                 detector_element_width_mm: float = 0.24,
                 detector_element_length_mm: float = 0.5,
                 center_frequency_hz: int = 3.96e6,
                 bandwidth_percent: int = 55,
                 sampling_frequency_mhz: float = 40,
                 device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None,
                 seed: int = None):
        """
        :param extent_mm:
        :param number_detector_elements:
        :param detector_element_width_mm:
        :param detector_element_length_mm:
        :param center_frequency_hz:
        :param bandwidth_percent:
        :param sampling_frequency_mhz:
        :param device_position_mm: Center of the linear array.
        """

        if seed is not None:
            np.random.seed(seed)

        self.detector_positions_base_mm = np.random.random((number_detector_elements, 3)) * 2 * extent_mm - extent_mm
        width = np.max(self.detector_positions_base_mm[:, 0]) - \
                np.min(self.detector_positions_base_mm[:, 0])
        self.probe_height_mm = np.max(self.detector_positions_base_mm[:, 2]) - \
                               np.min(self.detector_positions_base_mm[:, 2])
        self.probe_length_mm = np.max(self.detector_positions_base_mm[:, 1]) - \
                               np.min(self.detector_positions_base_mm[:, 1])

        if field_of_view_extent_mm is None:
            field_of_view_extent_mm = np.asarray([-width / 2,
                                                  width / 2,
                                                  -self.probe_length_mm / 2,
                                                  self.probe_length_mm / 2,
                                                  -self.probe_height_mm/2,
                                                  self.probe_height_mm/2])
        super().__init__(number_detector_elements=number_detector_elements,
                         detector_element_width_mm=detector_element_width_mm,
                         detector_element_length_mm=detector_element_length_mm,
                         center_frequency_hz=center_frequency_hz,
                         bandwidth_percent=bandwidth_percent,
                         sampling_frequency_mhz=sampling_frequency_mhz,
                         device_position_mm=device_position_mm,
                         field_of_view_extent_mm=field_of_view_extent_mm)

        self.probe_width_mm = width

    def check_settings_prerequisites(self, global_settings: Settings) -> bool:
        if global_settings[Tags.DIM_VOLUME_X_MM] < self.probe_width_mm + 1:
            self.logger.error("Volume x dimension is too small to encompass random device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format(self.probe_width_mm + 1, global_settings[Tags.DIM_VOLUME_X_MM]))
            return False
        if global_settings[Tags.DIM_VOLUME_Z_MM] < self.probe_height_mm + 1:
            self.logger.error("Volume z dimension is too small to encompass random device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format(self.probe_height_mm + 1, global_settings[Tags.DIM_VOLUME_Z_MM]))
            return False
        return True

    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        return np.copy(self.detector_positions_base_mm)

    def get_detector_element_orientations(self) -> np.ndarray:
        detector_orientations = np.zeros((self.number_detector_elements, 3))
        detector_orientations[:, 2] = 1
        return detector_orientations
