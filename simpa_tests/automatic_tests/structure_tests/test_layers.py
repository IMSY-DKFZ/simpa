# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.deformation_manager import create_deformation_settings
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import HorizontalLayerStructure


class TestLayers(unittest.TestCase):

    def setUp(self):
        self.global_settings = Settings()
        self.global_settings[Tags.SPACING_MM] = 1
        self.global_settings[Tags.SIMULATE_DEFORMED_LAYERS] = False
        self.global_settings[Tags.DEFORMED_LAYERS_SETTINGS] = create_deformation_settings(bounds_mm=[[0, 20], [0, 20]],
                                                                                          maximum_z_elevation_mm=3,
                                                                                          filter_sigma=0,
                                                                                          cosine_scaling_factor=4)
        self.global_settings[Tags.DIM_VOLUME_X_MM] = 2
        self.global_settings[Tags.DIM_VOLUME_Y_MM] = 2
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = 6

        self.layer_settings = Settings()
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 0]
        self.layer_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        self.layer_settings[Tags.ADHERE_TO_DEFORMATION] = True
        self.layer_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

        self.global_settings.set_volume_creation_settings(
            {
                Tags.STRUCTURES: self.layer_settings
            }
        )

    def assert_values(self, volume, values):
        assert abs(volume[0][0][0] - values[0]) < 1e-5, "excpected " + str(values[0]) + " but was " + str(volume[0][0][0])
        assert abs(volume[0][0][1] - values[1]) < 1e-5, "excpected " + str(values[1]) + " but was " + str(volume[0][0][1])
        assert abs(volume[0][0][2] - values[2]) < 1e-5, "excpected " + str(values[2]) + " but was " + str(volume[0][0][2])
        assert abs(volume[0][0][3] - values[3]) < 1e-5, "excpected " + str(values[3]) + " but was " + str(volume[0][0][3])
        assert abs(volume[0][0][4] - values[4]) < 1e-5, "excpected " + str(values[4]) + " but was " + str(volume[0][0][4])
        assert abs(volume[0][0][5] - values[5]) < 1e-5, "excpected " + str(values[5]) + " but was " + str(volume[0][0][5])

    def test_layer_structure_partial_volume_within_one_voxel(self):
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.3]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 1.7]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.4, 0, 0, 0, 0])

    def test_layer_structure_partial_volume_within_two_voxels(self):
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.4]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 2.5]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.6, 0.5, 0, 0, 0])

    def test_layer_structure_partial_volume_within_three_voxels(self):
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.3]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 3.5]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.7, 1, 0.5, 0, 0])

    def test_layer_structure_partial_volume_close_to_border(self):
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.2]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 2.1]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.8, 0.1, 0, 0, 0])

    def test_layer_structure_partial_volume_at_border(self):
        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.2]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 2.0]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.8, 0, 0, 0, 0])

        self.layer_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.0]
        self.layer_settings[Tags.STRUCTURE_END_MM] = [0, 0, 1.8]
        ls = HorizontalLayerStructure(self.global_settings, self.layer_settings)
        self.assert_values(ls.geometrical_volume, [0, 0.8, 0, 0, 0, 0])
