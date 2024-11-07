# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags, create_deformation_settings
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import CircularTubularStructure


class TestTubes(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.tube_settings = Settings()
        self.tube_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = 1
        self.tube_settings[Tags.STRUCTURE_END_MM] = [0, 5, 0]
        self.tube_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.tube_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.tube_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.tube_settings
            }
        )

    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + \
            str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + \
            str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + \
            str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + \
            str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + \
            str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + \
            str(values[5]) + " but was " + str(volume[0][0][5])

    def test_tube_structures_partial_volume_within_one_voxel(self):
        self.tube_settings[Tags.STRUCTURE_START_MM] = [0.5, 0, 0.5]
        self.tube_settings[Tags.STRUCTURE_END_MM] = [0.5, 5, 0.5]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ts = CircularTubularStructure(self.global_settings, self.tube_settings)

        target_mask = np.zeros(ts.geometrical_volume.shape, dtype=bool)
        target_mask[0, :, 0] = True
        assert np.all((0 < ts.geometrical_volume[target_mask]) & (ts.geometrical_volume[target_mask] < 1))
        assert np.all(0 == ts.geometrical_volume[~target_mask])

    def test_tube_structure_partial_volume_within_two_voxels(self):
        self.tube_settings[Tags.STRUCTURE_START_MM] = [1, 0, 1]
        self.tube_settings[Tags.STRUCTURE_END_MM] = [1, 5, 1]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ts = CircularTubularStructure(self.global_settings, self.tube_settings)

        target_mask = np.zeros(ts.geometrical_volume.shape, dtype=bool)
        target_mask[0:2, :, 0:2] = True
        assert np.all((0 < ts.geometrical_volume[target_mask]) & (ts.geometrical_volume[target_mask] < 1))
        assert np.all(0 == ts.geometrical_volume[~target_mask])

    def test_tube_structure_partial_volume_with_one_full_voxel(self):
        self.tube_settings[Tags.STRUCTURE_START_MM] = [1.5, 0, 1.5]
        self.tube_settings[Tags.STRUCTURE_END_MM] = [1.5, 5, 1.5]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = np.sqrt(3 * 0.25)  # diagonal of exactly one voxel
        ts = CircularTubularStructure(self.global_settings, self.tube_settings)

        # check values that should be 1
        assert np.all(ts.geometrical_volume[1, :, 1] == 1)

        # check values that should be fuzzy
        target_mask = np.zeros(ts.geometrical_volume.shape, dtype=bool)
        target_mask[(0, 1, 1, 2), :, (1, 0, 2, 1)] = True
        assert np.all((0 < ts.geometrical_volume[target_mask]) & (ts.geometrical_volume[target_mask] < 1))

        # check values that should be 0
        target_mask[1, :, 1] = True
        assert np.all(0 == ts.geometrical_volume[~target_mask])

    def test_tube_structure_partial_volume_across_volume(self):
        self.tube_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.tube_settings[Tags.STRUCTURE_END_MM] = [5, 5, 5]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = 1
        ts = CircularTubularStructure(self.global_settings, self.tube_settings)
        assert ts.geometrical_volume[0, 0, 0] == 1
        assert 0 < ts.geometrical_volume[0, 0, 1] < 1
        assert 0 < ts.geometrical_volume[1, 0, 0] < 1
        assert 0 < ts.geometrical_volume[1, 0, 1] < 1
        assert ts.geometrical_volume[2, 2, 2] == 1
        assert 0 < ts.geometrical_volume[2, 2, 3] < 1
        assert 0 < ts.geometrical_volume[3, 2, 2] < 1
        assert 0 < ts.geometrical_volume[3, 2, 3] < 1
        assert 0 < ts.geometrical_volume[2, 2, 1] < 1
        assert 0 < ts.geometrical_volume[1, 2, 2] < 1
        assert 0 < ts.geometrical_volume[1, 2, 3] < 1
        assert ts.geometrical_volume[4, 4, 4] == 1

    def test_tube_structure_deformation(self):
        self.tube_settings[Tags.STRUCTURE_START_MM] = [2.5, 0, 2.5]
        self.tube_settings[Tags.STRUCTURE_END_MM] = [2.5, 5, 2.5]
        self.tube_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.tube_settings,
                Tags.SIMULATE_DEFORMED_LAYERS: True,
                Tags.DEFORMED_LAYERS_SETTINGS: create_deformation_settings(
                    bounds_mm=[[0, self.global_settings[Tags.DIM_VOLUME_X_MM]],
                               [0, self.global_settings[Tags.DIM_VOLUME_Y_MM]]],
                    maximum_z_elevation_mm=3,
                    filter_sigma=0,
                    cosine_scaling_factor=0)
            }
        )

        ts = CircularTubularStructure(self.global_settings, self.tube_settings)

        # check that vessel was only deformed along z-axis, so there should only be values in 1 slice
        assert np.any(ts.geometrical_volume[2])
        assert np.all(ts.geometrical_volume[(0, 1, 3, 4), :, :] == 0)

        # check that the vessel was deformed (different from non-deformed case)
        center_mask = np.zeros(ts.geometrical_volume.shape, dtype=bool)
        center_mask[2, :, 2] = True
        assert np.any(ts.geometrical_volume[~center_mask])
