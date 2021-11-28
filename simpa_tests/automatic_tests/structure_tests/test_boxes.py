# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import RectangularCuboidStructure


class TestBoxes(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.box_settings = Settings()
        self.box_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.box_settings[Tags.STRUCTURE_X_EXTENT_MM] = 1
        self.box_settings[Tags.STRUCTURE_Y_EXTENT_MM] = 1
        self.box_settings[Tags.STRUCTURE_Z_EXTENT_MM] = 1
        self.box_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.box_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.box_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.box_settings
            }
        )

    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + str(values[5]) + " but was " + str(volume[0][0][5])

    def test_box_structures_partial_volume_within_one_voxel(self):
        self.box_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        edge_length = 0.9
        self.box_settings[Tags.STRUCTURE_X_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Y_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Z_EXTENT_MM] = edge_length
        bs = RectangularCuboidStructure(self.global_settings, self.box_settings)
        assert bs.geometrical_volume[0, 0, 0] == edge_length**3

    def test_box_structure_partial_volume_within_two_voxels(self):
        self.box_settings[Tags.STRUCTURE_START_MM] = [0.5, 0, 0.5]
        edge_length = 1
        self.box_settings[Tags.STRUCTURE_X_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Y_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Z_EXTENT_MM] = edge_length
        bs = RectangularCuboidStructure(self.global_settings, self.box_settings)
        assert bs.geometrical_volume[0, 0, 0] == 0.5**2
        assert bs.geometrical_volume[1, 0, 0] == 0.5**2
        assert bs.geometrical_volume[0, 0, 1] == 0.5**2
        assert bs.geometrical_volume[1, 0, 1] == 0.5**2

    def test_box_structure_partial_volume_with_one_full_voxel(self):
        self.box_settings[Tags.STRUCTURE_START_MM] = [0.75, 0, 0.75]
        edge_length = 1.5
        self.box_settings[Tags.STRUCTURE_X_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Y_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Z_EXTENT_MM] = edge_length
        bs = RectangularCuboidStructure(self.global_settings, self.box_settings)
        assert bs.geometrical_volume[1, 0, 1] == 1
        assert bs.geometrical_volume[0, 0, 0] == 0.25**2
        assert bs.geometrical_volume[0, 0, 2] == 0.25**2
        assert bs.geometrical_volume[2, 0, 0] == 0.25**2
        assert bs.geometrical_volume[2, 0, 2] == 0.25**2
        assert bs.geometrical_volume[0, 0, 1] == 0.5**2
        assert bs.geometrical_volume[1, 0, 0] == 0.5**2
        assert bs.geometrical_volume[2, 0, 1] == 0.5**2
        assert bs.geometrical_volume[1, 0, 2] == 0.5**2

    def test_box_structure_partial_volume_rectangular(self):
        self.box_settings[Tags.STRUCTURE_START_MM] = [0.75, 0, 0.75]
        edge_length = 1.5
        self.box_settings[Tags.STRUCTURE_X_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Y_EXTENT_MM] = edge_length
        self.box_settings[Tags.STRUCTURE_Z_EXTENT_MM] = 2.5
        bs = RectangularCuboidStructure(self.global_settings, self.box_settings)
        self.assertAlmostEqual(bs.geometrical_volume[1, 0, 1], 1)
        self.assertAlmostEqual(bs.geometrical_volume[1, 0, 2], 1)
        self.assertAlmostEqual(bs.geometrical_volume[0, 0, 0], 0.25 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[0, 0, 2], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[2, 0, 0], 0.25 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[2, 0, 2], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[0, 0, 1], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[1, 0, 0], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[2, 0, 1], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[0, 0, 3], 0.25 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[1, 0, 3], 0.5 ** 2)
        self.assertAlmostEqual(bs.geometrical_volume[2, 0, 3], 0.25 ** 2)
