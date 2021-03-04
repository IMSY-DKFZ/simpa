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
import numpy as np
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings_generator import Settings
from simpa.utils.libraries.structure_library import EllipticalTubularStructure


class TestEllipticalTubes(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.elliptical_tube_settings = Settings()
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = 1
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [0, 5, 0]
        self.elliptical_tube_settings[Tags.STRUCTURE_ECCENTRICITY] = 0
        self.elliptical_tube_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.elliptical_tube_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.elliptical_tube_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + str(values[5]) + " but was " + str(volume[0][0][5])

    def test_elliptical_tube_structures_partial_volume_within_one_voxel(self):
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [0.5, 0, 0.5]
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [0.5, 5, 0.5]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ets = EllipticalTubularStructure(self.global_settings, self.elliptical_tube_settings)
        assert 0 < ets.geometrical_volume[0, 0, 0] < 1
        assert 0 < ets.geometrical_volume[0, 4, 0] < 1

    def test_elliptical_tube_structure_partial_volume_within_two_voxels(self):
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [1, 0, 1]
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [1, 5, 1]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ets = EllipticalTubularStructure(self.global_settings, self.elliptical_tube_settings)
        assert 0 < ets.geometrical_volume[0, 0, 0] < 1
        assert 0 < ets.geometrical_volume[1, 0, 0] < 1
        assert 0 < ets.geometrical_volume[0, 1, 0] < 1
        assert 0 < ets.geometrical_volume[0, 0, 1] < 1
        assert 0 < ets.geometrical_volume[1, 1, 0] < 1
        assert 0 < ets.geometrical_volume[1, 0, 1] < 1
        assert 0 < ets.geometrical_volume[0, 1, 1] < 1
        assert 0 < ets.geometrical_volume[1, 1, 1] < 1

    def test_elliptical_tube_structure_partial_volume_with_one_full_voxel(self):
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [1.5, 0, 1.5]
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [1.5, 5, 1.5]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = np.sqrt(3*0.25)     # diagonal of exactly one voxel
        ets = EllipticalTubularStructure(self.global_settings, self.elliptical_tube_settings)
        assert ets.geometrical_volume[1, 1, 1] == 1
        assert ets.geometrical_volume[0, 1, 0] == 0
        assert ets.geometrical_volume[0, 1, 2] == 0
        assert ets.geometrical_volume[2, 1, 0] == 0
        assert ets.geometrical_volume[2, 2, 2] == 0
        assert 0 < ets.geometrical_volume[0, 1, 1] < 1
        assert 0 < ets.geometrical_volume[1, 1, 0] < 1
        assert 0 < ets.geometrical_volume[1, 1, 2] < 1
        assert 0 < ets.geometrical_volume[2, 1, 1] < 1

    def test_elliptical_tube_structure_partial_volume_across_volume(self):
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [5, 5, 5]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = 1
        ets = EllipticalTubularStructure(self.global_settings, self.elliptical_tube_settings)
        assert ets.geometrical_volume[0, 0, 0] == 1
        assert 0 < ets.geometrical_volume[0, 0, 1] < 1
        assert 0 < ets.geometrical_volume[1, 0, 0] < 1
        assert 0 < ets.geometrical_volume[1, 0, 1] < 1
        assert ets.geometrical_volume[2, 2, 2] == 1
        assert 0 < ets.geometrical_volume[2, 2, 3] < 1
        assert 0 < ets.geometrical_volume[3, 2, 2] < 1
        assert 0 < ets.geometrical_volume[3, 2, 3] < 1
        assert 0 < ets.geometrical_volume[2, 2, 1] < 1
        assert 0 < ets.geometrical_volume[1, 2, 2] < 1
        assert 0 < ets.geometrical_volume[1, 2, 3] < 1
        assert ets.geometrical_volume[4, 4, 4] == 1

    def test_elliptical_tube_structure_partial_volume_with_eccentricity(self):
        self.global_settings[Tags.DIM_VOLUME_X_MM] = 100
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 100
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 100
        self.elliptical_tube_settings[Tags.STRUCTURE_START_MM] = [50, 0, 50]
        self.elliptical_tube_settings[Tags.STRUCTURE_END_MM] = [50, 50, 50]
        self.elliptical_tube_settings[Tags.STRUCTURE_RADIUS_MM] = 15
        self.elliptical_tube_settings[Tags.STRUCTURE_ECCENTRICITY] = 0.8
        ets = EllipticalTubularStructure(self.global_settings, self.elliptical_tube_settings)
        assert ets.geometrical_volume[50, 50, 50] == 1
        assert ets.geometrical_volume[50, 50, 40] == 1
        assert ets.geometrical_volume[50, 50, 60] == 1
        assert ets.geometrical_volume[35, 50, 50] == 1
        assert ets.geometrical_volume[65, 50, 50] == 1
        assert 0 < ets.geometrical_volume[50, 50, 38] < 1
        assert 0 < ets.geometrical_volume[50, 50, 61] < 1
        assert 0 < ets.geometrical_volume[30, 50, 50] < 1
        assert 0 < ets.geometrical_volume[69, 50, 50] < 1
