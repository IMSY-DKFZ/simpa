# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import simpa as sp
from simpa.utils import Tags


class TestHeterogeneityGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.spacing = 1.0
        self.MIN = -4.0
        self.MAX = 8.0
        self.MEAN = -332.0
        self.STD = 78.0
        self.FULL_IMAGE = np.zeros((4, 8))
        self.FULL_IMAGE[:, 1:2] = 1
        self.PARTIAL_IMAGE = np.zeros((2, 2))
        self.PARTIAL_IMAGE[:, 1] = 1
        self.TEST_SETTINGS = sp.Settings({
            # These parameters set the general properties of the simulated volume
            sp.Tags.SPACING_MM: self.spacing,
            sp.Tags.DIM_VOLUME_Z_MM: 8,
            sp.Tags.DIM_VOLUME_X_MM: 4,
            sp.Tags.DIM_VOLUME_Y_MM: 7
        })
        dimx, dimy, dimz = self.TEST_SETTINGS.get_volume_dimensions_voxels()
        self.HETEROGENEITY_GENERATORS = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.FULL_IMAGE, spacing_mm=self.spacing,
                                  image_pixel_spacing_mm=self.spacing),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_CONSTANT, spacing_mm=self.spacing, constant=0.5),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_STRETCH, spacing_mm=self.spacing),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_SYMMETRIC, spacing_mm=self.spacing),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_WRAP, spacing_mm=self.spacing),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_EDGE, spacing_mm=self.spacing),
        ]
        self.HETEROGENEITY_GENERATORS_MIN_MAX = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX,
                                   gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.FULL_IMAGE, spacing_mm=self.spacing,
                                  target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_CONSTANT, spacing_mm=self.spacing, constant=0.5,
                                  target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_STRETCH, spacing_mm=self.spacing,
                                  target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_SYMMETRIC, spacing_mm=self.spacing,
                                  target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_WRAP, spacing_mm=self.spacing,
                                  target_min=self.MIN, target_max=self.MAX),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_EDGE, spacing_mm=self.spacing,
                                  target_min=self.MIN, target_max=self.MAX),
        ]
        self.HETEROGENEITY_GENERATORS_MEAN_STD = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing,
                                   target_mean=self.MEAN, target_std=self.STD),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_mean=self.MEAN, target_std=self.STD,
                                   gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_mean=self.MEAN, target_std=self.STD),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_CONSTANT, spacing_mm=self.spacing, constant=0.5,
                                  target_mean=self.MEAN, target_std=self.STD),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_STRETCH, spacing_mm=self.spacing,
                                  target_mean=self.MEAN, target_std=self.STD),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_SYMMETRIC, spacing_mm=self.spacing,
                                  target_mean=self.MEAN, target_std=self.STD),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_WRAP, spacing_mm=self.spacing,
                                  target_mean=self.MEAN, target_std=self.STD),
            sp.ImageHeterogeneity(dimx, dimy, dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                  scaling_type=Tags.IMAGE_SCALING_EDGE, spacing_mm=self.spacing,
                                  target_mean=self.MEAN, target_std=self.STD),
        ]

    def tearDown(self) -> None:
        pass

    def assert_dimension_size(self, heterogeneity_generator, dimx, dimy, dimz):
        random_map = heterogeneity_generator.get_map()
        map_shape = np.shape(random_map)
        self.assertAlmostEqual(dimx, map_shape[0])
        self.assertAlmostEqual(dimy, map_shape[1])
        self.assertAlmostEqual(dimz, map_shape[2])

    def test_dimension_sizes(self):
        dimx, dimy, dimz = self.TEST_SETTINGS.get_volume_dimensions_voxels()
        for generator in self.HETEROGENEITY_GENERATORS:
            self.assert_dimension_size(generator, dimx, dimy, dimz)

    def assert_min_max(self, heterogeneity_generator):
        random_map = heterogeneity_generator.get_map()
        self.assertAlmostEqual(np.min(random_map), self.MIN, 5)
        self.assertAlmostEqual(np.max(random_map), self.MAX, 5)

    def test_min_max_bounds(self):
        for generator in self.HETEROGENEITY_GENERATORS_MIN_MAX:
            self.assert_min_max(generator)

    def assert_mean_std(self, heterogeneity_generator):
        random_map = heterogeneity_generator.get_map()
        self.assertAlmostEqual(np.mean(random_map), self.MEAN, 5)
        self.assertAlmostEqual(np.std(random_map), self.STD, 5)

    def test_mean_std_bounds(self):
        for generator in self.HETEROGENEITY_GENERATORS_MEAN_STD:
            self.assert_mean_std(generator)


class TestImageScaling(unittest.TestCase):
    """
    A set of tests for the ImageHeterogeneity class, designed to see if the scaling works.
    """

    def setUp(self):
        self.spacing = 1.0
        self.MIN = -4.0
        self.MAX = 8.0
        self.PARTIAL_IMAGE = np.zeros((2, 2))
        self.PARTIAL_IMAGE[:, 1] = 1
        self.TOO_BIG_IMAGE = np.zeros((8, 8))
        self.TOO_BIG_IMAGE[:, :: 2] = 1
        self.TEST_SETTINGS = sp.Settings({
            # These parameters set the general properties of the simulated volume
            sp.Tags.SPACING_MM: self.spacing,
            sp.Tags.DIM_VOLUME_Z_MM: 8,
            sp.Tags.DIM_VOLUME_X_MM: 4,
            sp.Tags.DIM_VOLUME_Y_MM: 7
        })
        self.dimx, self.dimy, self.dimz = self.TEST_SETTINGS.get_volume_dimensions_voxels()

    def test_stretch(self):
        """
        Test to see if the image can be stretched to fill th area, and then the volume. After stretched to fill the
        area we should see the furthest two columns keep their values
        :return: Assertion for if the image has been stretched
        """
        stretched_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                                scaling_type=Tags.IMAGE_SCALING_STRETCH, spacing_mm=self.spacing).get_map()
        end_equals_1 = np.all(stretched_image[:, :, 6:] == 1)
        beginning_equals_0 = np.all(stretched_image[:, :, :1] == 0)
        assert end_equals_1 and beginning_equals_0

    def test_wrap(self):
        """
        Test to see if the image can be replicated to fill th area, and then the volume. Even and odd columns will keep
        their values
        :return: Assertion for if the image has been wrapped to fill the volume
        """
        wrapped_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                              scaling_type=Tags.IMAGE_SCALING_WRAP, spacing_mm=self.spacing).get_map()
        even_equal_1 = np.all(wrapped_image[:, :, 1::2] == 1)
        odd_equal_zero = np.all(wrapped_image[:, :, ::2] == 0)
        assert even_equal_1 and odd_equal_zero

    def test_edge(self):
        """
        Test to see if the image can fill the area by extending the edges, and then the volume. Should observe a line
        of zeros and the rest ones.
        :return: Assertion for if the image edges have filled the volume
        """
        edge_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                           scaling_type=Tags.IMAGE_SCALING_EDGE, spacing_mm=self.spacing).get_map()
        initially_zero = np.all(edge_image[:, :, 0] == 0)
        rest_ones = np.all(edge_image[:, :, 1:] == 1)
        assert initially_zero and rest_ones

    def test_constant(self):
        """
        Test to see if the image can fill the area with a constant, and then the volume
        :return: Assertion for if the image has been filled by a constant
        """
        constant_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                               scaling_type=Tags.IMAGE_SCALING_CONSTANT, spacing_mm=self.spacing,
                                               constant=0.5).get_map()
        initial_area_same = np.all(constant_image[1:3, :, 0] == 0) and np.all(constant_image[1:3, :, 1] == 1)
        rest_constant = np.all(constant_image[:, :, 2:] == 0.5) and np.all(constant_image[0, :, :] == 0.5) and \
            np.all(constant_image[3, :, :] == 0.5)
        assert initial_area_same and rest_constant

    def test_symmetric(self):
        """
        Test to see if the image can fill the area by symmetric reflections, and then the volume. See stripes following
        two pixel 1 to 0 patterns
        :return: Assertion for if the image has been reflected to fill the volume
        """
        symmetric_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.PARTIAL_IMAGE,
                                                scaling_type=Tags.IMAGE_SCALING_SYMMETRIC,
                                                spacing_mm=self.spacing).get_map()
        ones_stripes_working = np.all(symmetric_image[:, :, 1:3] == 1) and np.all(symmetric_image[:, :, 5:7] == 1)
        zeros_stripes_working = np.all(symmetric_image[:, :, 0] == 0) and np.all(symmetric_image[:, :, 3:5] == 0) and \
            np.all(symmetric_image[:, :, 7:] == 0)
        assert ones_stripes_working and zeros_stripes_working

    def test_crop(self):
        """
        Test to see if the image will crop to the desired area, thus leaving the same pattern but in a smaller shape
        :return: Assertion for if the image has been cropped to the desired area
        """
        crop_image = sp.ImageHeterogeneity(self.dimx, self.dimy, self.dimz, heterogeneity_image=self.TOO_BIG_IMAGE,
                                           spacing_mm=self.spacing, image_pixel_spacing_mm=self.spacing).get_map()
        odd_columns_equal_1 = np.all(crop_image[:, :, ::2] == 1)
        even_columns_equal_0 = np.all(crop_image[:, :, 1::2] == 0)
        size_is_right = np.all(crop_image[:, 1, :].shape == (self.dimx, self.dimz))
        assert odd_columns_equal_1 and even_columns_equal_0 and size_is_right
