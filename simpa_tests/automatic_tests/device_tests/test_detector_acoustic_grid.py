# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa.core.device_digital_twins import (CurvedArrayDetectionGeometry, LinearArrayDetectionGeometry, PlanarArrayDetectionGeometry)
from simpa.utils import Settings, Tags

class TestDetectorAcousticGrid(unittest.TestCase):
    @staticmethod
    def _settings(x_mm = 6, y_mm = 4, z_mm = 1, spacing_mm = 1):
        return Settings({Tags.DIM_VOLUME_X_MM: x_mm, Tags.DIM_VOLUME_Y_MM: y_mm, Tags.DIM_VOLUME_Z_MM: z_mm, Tags.SPACING_MM: spacing_mm})

    @staticmethod
    def _single_linear(position_mm, width_mm = 4, length_mm = 2):
        return LinearArrayDetectionGeometry(pitch_mm=1, number_detector_elements=1, detector_element_width_mm=width_mm, detector_element_length_mm=length_mm, device_position_mm=np.asarray(position_mm, dtype=float))

    def test_rectangular_element_corners_and_bounds(self):
        geometry = self._single_linear([3,2,0])
        corners_mm = geometry.get_detector_element_corners_mm()

        expected_corners_mm = np.array([
            [
                [1,1,0],
                [1,3,0],
                [5,1,0],
                [5,3,0],
            ]
        ])

        self.assertEqual(corners_mm.shape, (1,4,3))
        np.testing.assert_allclose(corners_mm, expected_corners_mm)

        lower_mm, upper_mm = (geometry.get_physical_detector_bounds_mm())
        np.testing.assert_allclose(lower_mm, [1,1,0])
        np.testing.assert_allclose(upper_mm, [5,3,0])

    def test_complete_aperture_inside_grid(self):
        geometry = self._single_linear([3,2,0])
        settings = self._settings(x_mm = 6, y_mm = 4, z_mm = 1)
        self.assertTrue(geometry.check_detector_elements_inside_acoustic_grid(settings))

    def test_inclusive_boundary(self):
        geometry = self._single_linear([2,1,0])
        settings = self._settings(x_mm=5, y_mm=3, z_mm=1)

        detector_lower_mm, detector_upper_mm = (geometry.get_physical_detector_bounds_mm())
        grid_lower_mm, grid_upper_mm = (geometry.get_acoustic_grid_bounds_mm(settings))

        np.testing.assert_allclose(detector_lower_mm, grid_lower_mm)
        np.testing.assert_allclose(detector_upper_mm, grid_upper_mm)
        self.assertTrue(geometry.check_detector_elements_inside_acoustic_grid(settings))

    def test_center_inside_but_incomplete_detector_is_rejected(self):
        geometry = self._single_linear([1,2,0])
        settings = self._settings()

        centers_mm = (geometry.get_detector_element_positions_accounting_for_device_position_mm())
        self.assertGreaterEqual(centers_mm[0,0],0)

        detector_lower_mm, _ = ( geometry.get_physical_detector_bounds_mm() )
        self.assertLess(detector_lower_mm[0], 0)

        self.assertFalse( geometry.check_detector_elements_inside_acoustic_grid(settings))

    def test_volume_boundary_beyond_gridpoint_rejected(self):
        geometry = self._single_linear([4.5,1,0], width_mm=1, length_mm=2)
        settings = self._settings(x_mm = 5, y_mm = 3, z_mm = 1, spacing_mm = 1)

        _, detector_upper_mm = (geometry.get_physical_detector_bounds_mm())
        _, grid_upper_mm = (geometry.get_acoustic_grid_bounds_mm(settings))

        self.assertEqual(detector_upper_mm[0], 5)
        self.assertEqual(grid_upper_mm[0], 4)

        self.assertFalse(geometry.check_detector_elements_inside_acoustic_grid(settings))

    def test_required_volume_dimensions(self):
        geometry = self._single_linear([3,2,0])
        required_dimensions_mm = (geometry.get_required_volume_dimensions_mm(spacing_mm=1))
        np.testing.assert_array_equal(required_dimensions_mm, [6,4,1])

    def test_negative_boundary(self):
        geometry = self._single_linear([1,2,0])

        with self.assertRaisesRegex(ValueError,"Move the device by"):
            geometry.get_required_volume_dimensions_mm(spacing_mm=1)

    def test_plannar_array_surface_axes(self):
        geometry = PlanarArrayDetectionGeometry(pitch_mm=1, number_detector_elements_x=2, number_detector_elements_y=2)

        width_axes, length_axes = (geometry.get_detector_element_surface_axes())

        expected_width_axes = np.tile(np.array([1.0, 0.0, 0.0]), (4,1))
        expected_length_axes = np.tile(np.array([0.0, 1.0, 0.0]), (4,1))

        np.testing.assert_allclose(width_axes, expected_width_axes)
        np.testing.assert_allclose(length_axes, expected_length_axes)
        geometry._validate_detector_surface_axes(width_axes, length_axes)

    def test_curved_array_surface_axes(self):
        geometry = CurvedArrayDetectionGeometry(pitch_mm=np.pi/2, radius_mm=1, number_detector_elements=4, angular_origin_offset=np.pi/4)

        width_axes, length_axes = (geometry.get_detector_element_surface_axes())

        expected_width_axes = np.array([
            [1,0,0],
            [0,0,-1],
            [-1,0,0],
            [0,0,1],
        ], dtype = float)
        expected_length_axes = np.tile(np.array([0.0, 1.0, 0.0]), (4,1))

        np.testing.assert_allclose(width_axes, expected_width_axes, atol=1e-10)
        np.testing.assert_allclose(length_axes, expected_length_axes, atol=1e-10)
        geometry._validate_detector_surface_axes(width_axes, length_axes)

if __name__ == "__main__":
    unittest.main()