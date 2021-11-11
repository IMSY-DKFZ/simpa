# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry
from simpa.utils import Settings, Tags
import numpy as np


class TestCurvedArray(unittest.TestCase):

    def setUp(self):

        self.VOLUME_WIDTH_IN_MM = 10
        self.VOLUME_HEIGHT_IN_MM = 10
        self.SPACING = 1
        self.RANDOM_SEED = 4711

    def test_simple_detection_geometry(self):
        detection_geometry = LinearArrayDetectionGeometry(pitch_mm=1,
                                                          number_detector_elements=4,
                                                          )

        detector_positions = detection_geometry.get_detector_element_positions_base_mm()
        print(detector_positions)
        correct_positions = [[-1.5, 0, 0],
                             [-0.5, 0, 0],
                             [0.5, 0, 0],
                             [1.5, 0, 0]]

        assert len(detector_positions) == 4

        for element in range(len(detector_positions)):
            assert (np.abs(detector_positions[element] - np.array(correct_positions[element])) < 1e-10).all()

        detector_orientations = detection_geometry.get_detector_element_orientations()

        for element in range(len(detector_positions)):
            assert (np.abs(detector_orientations[element] - np.array([0, 0, -1])) < 1e-10).all()

    def test_simple_detection_geometry_prerequisite_check(self):
        self.VOLUME_WIDTH_IN_MM = 3 + self.SPACING

        settings = Settings({Tags.DIM_VOLUME_X_MM: self.VOLUME_WIDTH_IN_MM})

        detection_geometry = LinearArrayDetectionGeometry(pitch_mm=1,
                                                          number_detector_elements=4,
                                                          )

        self.assertTrue(detection_geometry.check_settings_prerequisites(settings))

        settings[Tags.DIM_VOLUME_X_MM] = self.VOLUME_WIDTH_IN_MM - self.SPACING
        self.assertFalse(detection_geometry.check_settings_prerequisites(settings))
