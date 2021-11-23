# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import VesselStructure


class TestVesselTree(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1

        self.global_settings[Tags.DIM_VOLUME_X_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 5
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 5

        self.vesseltree_settings = Settings()
        self.vesseltree_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_MM] = 5
        self.vesseltree_settings[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
        self.vesseltree_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 5
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1.0
        self.vesseltree_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 1.0
        self.vesseltree_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.vesseltree_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.vesseltree_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.vesseltree_settings
            }
        )

    def test_vessel_tree_geometrical_volume(self):
        ts = VesselStructure(self.global_settings, self.vesseltree_settings)
        for value in np.nditer(ts.geometrical_volume):
            assert 0 <= value <= 1

        self.assertTrue(np.sum(ts.geometrical_volume) > 0)
