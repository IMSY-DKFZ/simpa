# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.core.device_digital_twins.detection_geometries import DetectionGeometryBase
from simpa.log.file_logger import Logger
from simpa.core.device_digital_twins.detection_geometries.linear_array import LinearArrayDetectionGeometry
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import compute_image_dimensions
from simpa.core.device_digital_twins.pa_devices.ithera_msot_acuity import MSOTAcuityEcho


class TestFieldOfView(unittest.TestCase):

    def setUp(self):
        msot = MSOTAcuityEcho()
        self.detection_geometry = msot.get_detection_geometry()
        self.odd_detection_geometry = LinearArrayDetectionGeometry(device_position_mm=[0, 0, 0],
                                                                   number_detector_elements=99)
        self.logger = Logger()

    def _test(self, field_of_view_extent_mm: list, spacing_in_mm: float, detection_geometry: DetectionGeometryBase):
        detection_geometry.field_of_view_extent_mm = field_of_view_extent_mm
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = compute_image_dimensions(
            detection_geometry.field_of_view_extent_mm, spacing_in_mm, self.logger)
        
        assert type(xdim) == int and type(ydim) == int and type(zdim) == int, "dimensions should be integers"
        assert xdim >= 1 and ydim >= 1 and zdim >= 1, "dimensions should be positive"

        return xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end

    def test_symmetric(self):
        image_dimensions = self._test([-25, 25, 0, 0, -12, 8], 0.2, self.detection_geometry)
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

        assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
        assert xdim == 250
        assert ydim == 100
        self.assertAlmostEqual(xdim_start, -125)
        self.assertAlmostEqual(xdim_end, 125)
        self.assertAlmostEqual(ydim_start, -60)
        self.assertAlmostEqual(ydim_end, 40)
        self.assertAlmostEqual(zdim_start, 0)
        self.assertAlmostEqual(zdim_end, 0)

    def test_symmetric_with_small_spacing(self):
        image_dimensions = self._test([-25, 25, 0, 0, -12, 8], 0.1, self.detection_geometry)
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

        assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
        assert xdim == 500
        assert ydim == 200
        self.assertAlmostEqual(xdim_start, -250)
        self.assertAlmostEqual(xdim_end, 250)
        self.assertAlmostEqual(ydim_start, -120)
        self.assertAlmostEqual(ydim_end, 80)
        self.assertAlmostEqual(zdim_start, 0)
        self.assertAlmostEqual(zdim_end, 0)

    def test_unsymmetric_with_small_spacing(self):
        image_dimensions = self._test([-25, 24.9, 0, 0, -12, 8], 0.1, self.detection_geometry)
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

        assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
        assert xdim == 499
        assert ydim == 200
        self.assertAlmostEqual(xdim_start, -250)
        self.assertAlmostEqual(xdim_end, 249)
        self.assertAlmostEqual(ydim_start, -120)
        self.assertAlmostEqual(ydim_end, 80)
        self.assertAlmostEqual(zdim_start, 0)
        self.assertAlmostEqual(zdim_end, 0)

    def test_unsymmetric(self):
        image_dimensions = self._test([-25, 24.9, 0, 0, -12, 8], 0.2, self.detection_geometry)
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

        assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
        assert xdim == 250
        assert ydim == 100
        self.assertAlmostEqual(xdim_start, -125.25)
        self.assertAlmostEqual(xdim_end, 124.75)
        self.assertAlmostEqual(ydim_start, -60)
        self.assertAlmostEqual(ydim_end, 40)
        self.assertAlmostEqual(zdim_start, 0)
        self.assertAlmostEqual(zdim_end, 0)

    def test_symmetric_with_odd_number_of_elements(self):
        """
        The number of sensor elements should not affect the image dimensionality
        """
        image_dimensions = self._test([-25, 25, 0, 0, -12, 8], 0.2, self.odd_detection_geometry)
        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

        assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
        assert xdim == 250
        assert ydim == 100
        self.assertAlmostEqual(xdim_start, -125)
        self.assertAlmostEqual(xdim_end, 125)
        self.assertAlmostEqual(ydim_start, -60)
        self.assertAlmostEqual(ydim_end, 40)
        self.assertAlmostEqual(zdim_start, 0)
        self.assertAlmostEqual(zdim_end, 0)

    def test_with_changing_spacing(self):
        """
        Tests different spacings with the same field of view. Dimensions and their start and end are expected to change
        """
        spacings = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_xdims = [100, 50, 33, 25, 20]
        expected_ydims = [200, 100, 67, 50, 40]
        expected_xdim_starts = [-50, -25, -16.5, -12.5, -10]
        expected_xdim_ends = [50, 25, 16.5, 12.5, 10]
        expected_ydim_starts = [-120, -60, -40.16666666, -30, -24]
        expected_ydim_ends = [80, 40, 26.83333333, 20, 16]

        for i, spacing in enumerate(spacings):
            image_dimensions = self._test([-5, 5, 0, 0, -12, 8], spacing, self.detection_geometry)
            xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = image_dimensions

            assert zdim == 1, "With no FOV extend in z dimension only one slice should be created"
            assert xdim == expected_xdims[i]
            assert ydim == expected_ydims[i]
            self.assertAlmostEqual(xdim_start, expected_xdim_starts[i])
            self.assertAlmostEqual(xdim_end, expected_xdim_ends[i])
            self.assertAlmostEqual(ydim_start, expected_ydim_starts[i])
            self.assertAlmostEqual(ydim_end, expected_ydim_ends[i])
            self.assertAlmostEqual(zdim_start, 0)
            self.assertAlmostEqual(zdim_end, 0)

if __name__ == '__main__':
    test = TestFieldOfView()
    test.setUp()
    test.test_symmetric()
    test.test_symmetric_with_small_spacing()
    test.test_unsymmetric_with_small_spacing()
    test.test_unsymmetric()
    test.test_symmetric_with_odd_number_of_elements()
    test.test_with_changing_spacing()
