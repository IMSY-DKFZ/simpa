# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import SphericalStructure


class TestSpheres(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.sphere_settings = Settings()
        self.sphere_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.sphere_settings[Tags.STRUCTURE_RADIUS_MM] = 1
        self.sphere_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.sphere_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.sphere_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.sphere_settings
            }
        )


    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + str(values[5]) + " but was " + str(volume[0][0][5])

    def test_spherical_structures_partial_volume_within_one_voxel(self):
        self.sphere_settings[Tags.STRUCTURE_START_MM] = [0.5, 0.5, 0.5]
        self.sphere_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ss = SphericalStructure(self.global_settings, self.sphere_settings)
        assert 0 < ss.geometrical_volume[0, 0, 0] < 1

    def test_spherical_structure_partial_volume_within_two_voxels(self):
        self.sphere_settings[Tags.STRUCTURE_START_MM] = [1, 1, 1]
        self.sphere_settings[Tags.STRUCTURE_RADIUS_MM] = 0.4
        ss = SphericalStructure(self.global_settings, self.sphere_settings)
        assert 0 < ss.geometrical_volume[0, 0, 0] < 1
        assert 0 < ss.geometrical_volume[1, 0, 0] < 1
        assert 0 < ss.geometrical_volume[0, 1, 0] < 1
        assert 0 < ss.geometrical_volume[0, 0, 1] < 1
        assert 0 < ss.geometrical_volume[1, 1, 0] < 1
        assert 0 < ss.geometrical_volume[1, 0, 1] < 1
        assert 0 < ss.geometrical_volume[0, 1, 1] < 1
        assert 0 < ss.geometrical_volume[1, 1, 1] < 1

    def test_spherical_structure_partial_volume_with_one_full_voxel(self):
        self.sphere_settings[Tags.STRUCTURE_START_MM] = [1.5, 1.5, 1.5]
        self.sphere_settings[Tags.STRUCTURE_RADIUS_MM] = np.sqrt(3*0.25)     # diagonal of exactly one voxel
        ss = SphericalStructure(self.global_settings, self.sphere_settings)
        assert ss.geometrical_volume[1, 1, 1] == 1
        assert ss.geometrical_volume[0, 1, 0] == 0
        assert ss.geometrical_volume[0, 1, 2] == 0
        assert ss.geometrical_volume[2, 1, 0] == 0
        assert ss.geometrical_volume[2, 2, 2] == 0
        assert 0 < ss.geometrical_volume[0, 1, 1] < 1
        assert 0 < ss.geometrical_volume[1, 1, 0] < 1
        assert 0 < ss.geometrical_volume[1, 1, 2] < 1
        assert 0 < ss.geometrical_volume[2, 1, 1] < 1

    def test_spherical_structure_partial_volume_at_border(self):
        self.sphere_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.sphere_settings[Tags.STRUCTURE_RADIUS_MM] = np.sqrt(3*1)     # diagonal of exactly one voxel
        ss = SphericalStructure(self.global_settings, self.sphere_settings)
        assert ss.geometrical_volume[0, 0, 0] == 1
        assert 0 < ss.geometrical_volume[0, 0, 1] < 1
        assert 0 < ss.geometrical_volume[1, 0, 0] < 1
        assert 0 < ss.geometrical_volume[1, 0, 1] < 1
        assert 0 < ss.geometrical_volume[0, 1, 0] < 1
        assert 0 < ss.geometrical_volume[0, 1, 1] < 1
        assert 0 < ss.geometrical_volume[1, 1, 0] < 1
        assert ss.geometrical_volume[1, 1, 1] == 0
