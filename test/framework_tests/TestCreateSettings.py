# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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
from simpa.utils import Tags
from simpa.utils.settings_generator import Settings
import numpy as np


class TestCreateSettings(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_create_settings(self):

        settings = Settings()
        settings[Tags.WAVELENGTHS] = np.arange(800, 950, 10)
        settings.add_minimal_meta_information()

    @unittest.expectedFailure
    def test_create_settings_with_irregular_tags(self):

        settings = Settings()
        settings[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.zeros([2, 2, 3])

    @unittest.expectedFailure
    def test_create_settings_with_irregular_data_type(self):

        settings = Settings()
        settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = np.zeros([2, 2, 3])

    # TODO: More tests


