# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np

from simpa.core.device_digital_twins import DetectionGeometryBase
import os


class CupArrayDetectionGeometry(DetectionGeometryBase):
    """
    This class represents a digital twin of a ultrasound detection device
    with a cup detection geometry. The origin for this device is the center (focus) of the cup array.
    """

    def __init__(self, radius_mm=30,
                 pitch_mm=None,
                 number_detector_elements=384,
                 detector_element_width_mm=None,
                 detector_element_length_mm=None,
                 center_frequency_hz=8e6,
                 bandwidth_percent=50,
                 sampling_frequency_mhz=40,
                 angular_origin_offset=None,
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

        super(CupArrayDetectionGeometry, self).__init__(
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

        if field_of_view_extent_mm is None:
            self.logger.warning("Set field_of_view_extend_mm, while looping over first or second dimension - otherwise "
                                "memory issues could arise. Default FOV is 4mm*4mm*8.5mm, hence very small.")
            self.field_of_view_extent_mm = np.asarray([-2., 2., -2., 2., -6.5, 2.])  # -6.5 mm is level of membrane
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

    def check_settings_prerequisites(self, global_settings) -> bool:
        pass

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings):
        pass

    def get_detector_element_positions_base_mm(self) -> np.ndarray:
        py_file_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
        detector_positions = np.load(f'{py_file_path}/cup_positions_PType307', allow_pickle=True) * 1000  # m -> mm
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
        return {"CupArrayDetectionGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = CupArrayDetectionGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
