# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.core.device_digital_twins import CurvedArrayDetectionGeometry
from simpa.utils import Settings, Tags
import numpy as np


class TestCurvedArray(unittest.TestCase):

    def setUp(self):

        self.VOLUME_WIDTH_IN_MM = 10
        self.VOLUME_HEIGHT_IN_MM = 10
        self.SPACING = 1
        self.RANDOM_SEED = 4711

    def test_simple_detection_geometry(self):
        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=np.pi/2,
                                                          radius_mm=1,
                                                          number_detector_elements=4,
                                                          angular_origin_offset=np.pi/4
                                                          )

        detector_positions = detection_geometry.get_detector_element_positions_base_mm()
        correct_positions = [[0, 0, -1],
                             [-1, 0, 0],
                             [0, 0, 1],
                             [1, 0, 0]]

        assert len(detector_positions) == 4

        for element in range(len(detector_positions)):
            assert (np.abs(detector_positions[element] - np.array(correct_positions[element])) < 1e-10).all()

        detector_orientations = detection_geometry.get_detector_element_orientations()
        correct_orientations = -1 * np.array(correct_positions)

        for element in range(len(detector_positions)):
            assert (np.abs(detector_orientations[element] - np.array(correct_orientations[element])) < 1e-10).all()

    def test_simple_detection_geometry_prerequisite_check(self):
        self.VOLUME_WIDTH_IN_MM = 2 + self.SPACING
        self.VOLUME_HEIGHT_IN_MM = 2 + self.SPACING

        settings = Settings({Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM,
                             Tags.DIM_VOLUME_Z_MM: self. VOLUME_HEIGHT_IN_MM})

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=np.pi/2,
                                                          radius_mm=1,
                                                          number_detector_elements=4,
                                                          angular_origin_offset=np.pi/4
                                                          )

        self.assertTrue(detection_geometry.check_settings_prerequisites(settings))

        settings[Tags.DIM_VOLUME_X_MM] = self.VOLUME_WIDTH_IN_MM - self.SPACING
        self.assertFalse(detection_geometry.check_settings_prerequisites(settings))

        settings[Tags.DIM_VOLUME_X_MM] = self.VOLUME_WIDTH_IN_MM
        settings[Tags.DIM_VOLUME_Z_MM] = self.VOLUME_HEIGHT_IN_MM - self.SPACING
        self.assertFalse(detection_geometry.check_settings_prerequisites(settings))

        settings[Tags.DIM_VOLUME_X_MM] = self.VOLUME_WIDTH_IN_MM - self.SPACING
        settings[Tags.DIM_VOLUME_Z_MM] = self.VOLUME_HEIGHT_IN_MM - self.SPACING
        self.assertFalse(detection_geometry.check_settings_prerequisites(settings))
