# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import torch
from skimage import measure
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import VesselStructure


class TestVesselTree(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1
        self.global_settings[Tags.RANDOM_SEED] = 42
        self.global_settings[Tags.DIM_VOLUME_X_MM] = 10
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 10
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 10

        self.vesseltree_settings = Settings()
        self.vesseltree_settings[Tags.STRUCTURE_START_MM] = [5, 0, 5]
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_MM] = 2
        self.vesseltree_settings[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
        self.vesseltree_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 20
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 0
        self.vesseltree_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 0
        self.vesseltree_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.vesseltree_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.vesseltree_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.vesseltree_settings
            }
        )

    def test_bifurcation(self):
        """
        Test bifurcation
        Let bifurcation occur once at a large angle and see if the branch does indeed split into two, thus two spots
        at the edges
        WARNING: this method uses a pre-specified random seed which ensures ONE SPLIT for the CURRENT pipeline.
        :return: Assertion for if bifurcation occurs once
        """
        torch.manual_seed(10)
        self.global_settings[Tags.SPACING_MM] = 0.04
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_MM] = 0.5
        self.vesseltree_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 7
        self.vesseltree_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.1
        ts = VesselStructure(self.global_settings, self.vesseltree_settings)

        end_plane = ts.geometrical_volume[:, -1, :]
        top_plane = ts.geometrical_volume[:, :, -1]
        bottom_plane = ts.geometrical_volume[:, :, 0]
        left_plane = ts.geometrical_volume[0, :, :]
        right_plane = ts.geometrical_volume[-1, :, :]

        end_plane_count = measure.label(end_plane, background=0, return_num=True)[1]
        top_plane_count = measure.label(top_plane, background=0, return_num=True)[1]
        bottom_plane_count = measure.label(bottom_plane, background=0, return_num=True)[1]
        left_plane_count = measure.label(left_plane, background=0, return_num=True)[1]
        right_plane_count = measure.label(right_plane, background=0, return_num=True)[1]
        has_split = end_plane_count + top_plane_count + bottom_plane_count + left_plane_count + right_plane_count == 2

        assert has_split

    def test_radius_variation_factor(self):
        """
        Test radius variation factor
        Let there be no bifurcation or curvature, and see if the radius changes
        :return: Assertion for radius variation
        """
        self.vesseltree_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
        ts = VesselStructure(self.global_settings, self.vesseltree_settings)

        vessel_centre = 5
        edge_of_vessel = vessel_centre + self.vesseltree_settings[Tags.STRUCTURE_RADIUS_MM]
        has_reduced = np.min(ts.geometrical_volume[edge_of_vessel-1, :, vessel_centre]) == 0
        has_increased = np.max(ts.geometrical_volume[edge_of_vessel+1, :, vessel_centre]) != 0

        assert has_reduced or has_increased

    def test_curvature_factor(self):
        """
        Test curvature factor
        Let there be no bifurcation or radius change and observe if the vessel leaves its original trajectory,
        therefore breaching the imaginary box put around it
        :return: Assertion of curvature
        """
        curvature_factor = 0.2
        self.vesseltree_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = curvature_factor
        ts = VesselStructure(self.global_settings, self.vesseltree_settings)
        radius = self.vesseltree_settings[Tags.STRUCTURE_RADIUS_MM]
        vessel_centre = 5

        edge_of_vessel = radius + vessel_centre
        has_breached_top = np.max(np.nditer(ts.geometrical_volume[:, :, edge_of_vessel+1])) != 0
        has_breached_bottom = np.max(np.nditer(ts.geometrical_volume[:, :, -edge_of_vessel-1])) != 0
        has_breached_left = np.max(np.nditer(ts.geometrical_volume[-edge_of_vessel-1, :, :])) != 0
        has_breached_right = np.max(np.nditer(ts.geometrical_volume[edge_of_vessel+1, :, :])) != 0

        assert has_breached_top or has_breached_bottom or has_breached_left or has_breached_right

    def test_vessel_tree_geometrical_volume(self):
        ts = VesselStructure(self.global_settings, self.vesseltree_settings)
        for value in np.nditer(ts.geometrical_volume):
            assert 0 <= value <= 1

        self.assertTrue(np.sum(ts.geometrical_volume) > 0)
