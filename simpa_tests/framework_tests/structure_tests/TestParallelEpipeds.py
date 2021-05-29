# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings_generator import Settings
from simpa.utils.libraries.structure_library import ParallelepipedStructure


class TestParallelEpipeds(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.parallelepiped_settings = Settings()
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [1, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, 1, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 0, 1]
        self.parallelepiped_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.parallelepiped_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.parallelepiped_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + str(values[5]) + " but was " + str(volume[0][0][5])

    def test_parallelepiped_structures_smaller_than_one_voxel(self):
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        edge_length = 0.9
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [edge_length, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, edge_length, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 0, edge_length]
        ps = ParallelepipedStructure(self.global_settings, self.parallelepiped_settings)
        assert ps.geometrical_volume[0, 0, 0] == 0

    def test_parallelepiped_structures_larger_than_one_voxel(self):
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        edge_length = 1.1
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [edge_length, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, edge_length, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 0, edge_length]
        ps = ParallelepipedStructure(self.global_settings, self.parallelepiped_settings)
        assert ps.geometrical_volume[0, 0, 0] == 1

    def test_parallelepiped_structure_within_two_voxels_only_one_full_voxel(self):
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [0.1, 0, 0.1]
        edge_length = 1.9
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [edge_length, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, edge_length, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 0, edge_length]
        ps = ParallelepipedStructure(self.global_settings, self.parallelepiped_settings)
        assert ps.geometrical_volume[0, 0, 0] == 0
        assert ps.geometrical_volume[0, 0, 1] == 0
        assert ps.geometrical_volume[1, 0, 0] == 0
        assert ps.geometrical_volume[1, 0, 1] == 1

    def test_parallelepiped_structure_multiple_full_voxels(self):
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [0.1, 0, 0.1]
        edge_length = 2.9
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [edge_length, 0, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, edge_length, 0]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 0, edge_length]
        ps = ParallelepipedStructure(self.global_settings, self.parallelepiped_settings)
        assert ps.geometrical_volume[0, 0, 0] == 0
        assert ps.geometrical_volume[0, 0, 1] == 0
        assert ps.geometrical_volume[1, 0, 0] == 0
        assert ps.geometrical_volume[1, 0, 1] == 1
        assert ps.geometrical_volume[2, 0, 1] == 1
        assert ps.geometrical_volume[1, 0, 2] == 1
        assert ps.geometrical_volume[2, 0, 2] == 1

    def test_parallelepiped_structure_diagonal_edges(self):
        self.parallelepiped_settings[Tags.STRUCTURE_START_MM] = [5, 0, 5]
        self.global_settings[Tags.DIM_VOLUME_X_MM] = 50
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 50
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 50
        edge_length = 20
        self.parallelepiped_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [edge_length, 0, edge_length/4]
        self.parallelepiped_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [edge_length/4, 0, edge_length]
        self.parallelepiped_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, edge_length, 0]
        ps = ParallelepipedStructure(self.global_settings, self.parallelepiped_settings)
        assert ps.geometrical_volume[5, 0, 5] == 1
        assert ps.geometrical_volume[28, 0, 28] == 1
        assert ps.geometrical_volume[24, 0, 10] == 1
        assert ps.geometrical_volume[10, 0, 24] == 1
