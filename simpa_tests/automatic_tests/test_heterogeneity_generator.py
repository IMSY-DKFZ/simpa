import unittest
import numpy as np
import simpa as sp


class TestHeterogeneityGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.spacing = 1.0
        self.MIN = -4.0
        self.MAX = 8.0
        self.MEAN = -332.0
        self.STD = 78.0
        self.TEST_SETTINGS = sp.Settings({
                                            # These parameters set the general properties of the simulated volume
                                            sp.Tags.SPACING_MM: self.spacing,
                                            sp.Tags.DIM_VOLUME_Z_MM: 4,
                                            sp.Tags.DIM_VOLUME_X_MM: 2,
                                            sp.Tags.DIM_VOLUME_Y_MM: 7
                                        })
        dimx, dimy, dimz = self.TEST_SETTINGS.get_volume_dimensions_voxels()
        self.HETEROGENEITY_GENERATORS = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing)
        ]
        self.HETEROGENEITY_GENERATORS_MIN_MAX = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX,
                                   gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_min=self.MIN, target_max=self.MAX)
        ]
        self.HETEROGENEITY_GENERATORS_MEAN_STD = [
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_mean=self.MEAN, target_std=self.STD),
            sp.RandomHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_mean=self.MEAN, target_std=self.STD,
                                   gaussian_blur_size_mm=3),
            sp.BlobHeterogeneity(dimx, dimy, dimz, spacing_mm=self.spacing, target_mean=self.MEAN, target_std=self.STD),
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