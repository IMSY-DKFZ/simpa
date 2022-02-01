# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np

from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils import Tags


class CurvedArrayDetectionGeometry(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a curved detection geometry. The origin for this device is the center (focus) of the curved array.
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
                 device_position_mm=None,
                 field_of_view_extent_mm=None):
        """

        :param pitch_mm: In-plane distance between the beginning of one detector element to the next detector element.
        :param radius_mm:
        :param number_detector_elements:
        :param detector_element_width_mm:
        :param detector_element_length_mm:
        :param center_frequency_hz:
        :param bandwidth_percent:
        :param sampling_frequency_mhz:
        :param angular_origin_offset:
        :param device_position_mm: Center (focus) of the curved array.
        """

        super(CurvedArrayDetectionGeometry, self).__init__(
             number_detector_elements=number_detector_elements,
             detector_element_width_mm=detector_element_width_mm,
             detector_element_length_mm=detector_element_length_mm,
             center_frequency_hz=center_frequency_hz,
             bandwidth_percent=bandwidth_percent,
             sampling_frequency_mhz=sampling_frequency_mhz,
             device_position_mm=device_position_mm)

        self.pitch_mm = pitch_mm
        self.radius_mm = radius_mm
        self.angular_origin_offset = angular_origin_offset

        detector_positions = self.get_detector_element_positions_base_mm()
        min_x_coordinate = np.min(detector_positions[:, 0])
        max_x_coordinate = np.max(detector_positions[:, 0])
        self.probe_width_mm = max_x_coordinate - min_x_coordinate

        min_z_coordinate = np.min(detector_positions[:, 2])
        max_z_coordinate = np.max(detector_positions[:, 2])
        self.probe_height_mm = max_z_coordinate - min_z_coordinate

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-self.probe_width_mm/2,
                                                       self.probe_width_mm/2,
                                                       0, 0, 0, 100])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

    def check_settings_prerequisites(self, global_settings) -> bool:
        if global_settings[Tags.DIM_VOLUME_Z_MM] < (self.probe_height_mm + 1):
            self.logger.error("Volume z dimension is too small to encompass the device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format((self.probe_height_mm + 1),
                                      global_settings[Tags.DIM_VOLUME_Z_MM]))
            return False
        if global_settings[Tags.DIM_VOLUME_X_MM] < (self.probe_width_mm + 1):
            self.logger.error("Volume x dimension is too small to encompass MSOT device in simulation!"
                              "Must be at least {} mm but was {} mm"
                              .format(self.probe_width_mm, global_settings[Tags.DIM_VOLUME_X_MM]))
            return False
        return True

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings):
        pass

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

    def get_detector_element_orientations(self) -> np.ndarray:
        detector_positions = self.get_detector_element_positions_base_mm()
        detector_orientations = np.subtract(0, detector_positions)
        norm = np.linalg.norm(detector_orientations, axis=-1)
        for dim in range(3):
            detector_orientations[:, dim] = detector_orientations[:, dim] / norm
        return detector_orientations

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {"CurvedArrayDetectionGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = CurvedArrayDetectionGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
