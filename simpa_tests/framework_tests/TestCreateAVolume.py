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
import os
import numpy as np
from copy import copy

from simpa.utils import Tags
from simpa.utils.settings_generator import Settings
from simpa.core.simulation import simulate
from simpa_tests.test_utils import create_test_structure_parameters
from simpa.core.volume_creation.volume_creation import check_volumes


class TestCreateVolume(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_create_volume(self):

        random_seed = 4711
        settings = {
            Tags.WAVELENGTHS: [800, 801],
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.RANDOM_SEED: random_seed,
            Tags.VOLUME_NAME: "FlowPhantom_" + str(random_seed).zfill(6),
            Tags.SIMULATION_PATH: ".",
            Tags.RUN_OPTICAL_MODEL: False,
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
            Tags.SPACING_MM: 0.3,
            Tags.DIM_VOLUME_Z_MM: 5,
            Tags.DIM_VOLUME_X_MM: 4,
            Tags.DIM_VOLUME_Y_MM: 3
        }
        print("Simulating ", random_seed)
        settings = Settings(settings)
        settings[Tags.STRUCTURES] = create_test_structure_parameters(settings)
        simulate(settings)

        if (os.path.exists(settings[Tags.SIMPA_OUTPUT_PATH]) and
           os.path.isfile(settings[Tags.SIMPA_OUTPUT_PATH])):
            # Delete the created file
            os.remove(settings[Tags.SIMPA_OUTPUT_PATH])

        print("Simulating ", random_seed, "[Done]")


class TestCheckVolumes(unittest.TestCase):
    def setUp(self) -> None:
        self.required_tags = Tags.MINIMUM_REQUIRED_VOLUMES
        self.volumes = dict()
        self.seed = 1234
        np.random.seed(self.seed)
        self.shape = (100, 50, 200)
        for key in self.required_tags:
            self.volumes[key] = np.random.random_sample(self.shape)

    def test_check_volumes_shapes(self):
        new_shape = tuple([s+1 for s in self.shape])
        new_volumes = copy(self.volumes)
        new_volumes[self.required_tags[0]] = np.random.random_sample(new_shape)
        with self.assertRaises(ValueError, msg="ValueError was not raised when volume shapes differ"):
            check_volumes(new_volumes)

    def test_check_volumes_keys(self):
        self.volumes.pop(self.required_tags[0])
        with self.assertRaises(ValueError, msg="ValueError was not raised when tag is missing in volume"):
            check_volumes(self.volumes)

    def test_check_volumes_values_nan(self):
        new_volumes = copy(self.volumes)
        new_volumes[self.required_tags[0]][0, 0] = np.nan
        with self.assertRaises(ValueError, msg="ValueError was not raised when volume shapes differ"):
            check_volumes(new_volumes)

    def test_check_volumes_values_inf(self):
        new_volumes = copy(self.volumes)
        new_volumes[self.required_tags[0]][0, 0] = np.inf
        with self.assertRaises(ValueError, msg="ValueError was not raised when volume shapes differ"):
            check_volumes(new_volumes)
