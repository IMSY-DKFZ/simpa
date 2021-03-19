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
from simpa.utils import Tags
from simpa.utils.settings_generator import Settings
import numpy as np
from simpa_tests.test_utils import assert_equals_recursive
import os


class TestCreateSettings(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_create_settings(self):
        settings = Settings()
        settings.add_minimal_meta_information()
        settings.add_minimal_optical_properties()
        settings.add_acoustic_properties()
        settings.add_reconstruction_properties()

    def test_create_settings_with_custom_parameters(self):
        settings = Settings()
        settings[Tags.WAVELENGTHS] = np.arange(800, 950, 10)
        settings.add_minimal_meta_information(volume_name="Name",
                                              simulation_path="Path",
                                              random_seed=1234,
                                              spacing=0.2,
                                              volume_dim_x=123,
                                              volume_dim_y=231,
                                              volume_dim_z=321)
        assert settings[Tags.VOLUME_NAME] == "Name"
        assert settings[Tags.SIMULATION_PATH] == "Path"
        assert settings[Tags.RANDOM_SEED] == 1234
        assert settings[Tags.SPACING_MM] == 0.2
        assert settings[Tags.DIM_VOLUME_X_MM] == 123
        assert settings[Tags.DIM_VOLUME_Y_MM] == 231
        assert settings[Tags.DIM_VOLUME_Z_MM] == 321

        settings.add_minimal_optical_properties(run_optical_model=False,
                                                wavelengths=[123],
                                                optical_model="test",
                                                photon_number=12345,
                                                illumination_type="Test",
                                                illumination_position=[1, 2, 3],
                                                illumination_direction=[3, 2, 1])

        assert settings[Tags.RUN_OPTICAL_MODEL] is False
        assert settings[Tags.WAVELENGTHS] == [123]
        assert settings[Tags.OPTICAL_MODEL] == "test"
        assert settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] == 12345
        assert settings[Tags.ILLUMINATION_TYPE] == "Test"
        assert settings[Tags.ILLUMINATION_POSITION] == [1, 2, 3]
        assert settings[Tags.ILLUMINATION_DIRECTION] == [3, 2, 1]

        settings.add_acoustic_properties(run_acoustic_model=False,
                                         acoustic_model="test",
                                         acoustic_simulation_3D=True,
                                         speed_of_sound=1234,
                                         density=123)

        assert settings[Tags.RUN_ACOUSTIC_MODEL] is False
        assert settings[Tags.ACOUSTIC_MODEL] == "test"
        assert settings[Tags.ACOUSTIC_SIMULATION_3D] is True
        assert settings[Tags.PROPERTY_SPEED_OF_SOUND] == 1234
        assert settings[Tags.PROPERTY_DENSITY] == 123

        settings.add_reconstruction_properties(perform_image_reconstruction=False,
                                               reconstruction_algorithm="Test")

        assert settings[Tags.PERFORM_IMAGE_RECONSTRUCTION] is False
        assert settings[Tags.RECONSTRUCTION_ALGORITHM] == "Test"

    def test_create_settings_with_irregular_tags(self):
        settings = Settings()
        settings[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.zeros([2, 2, 3])

    @unittest.expectedFailure
    def test_create_settings_with_irregular_data_type(self):
        settings = Settings()
        settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = np.zeros([2, 2, 3])

    def test_populate_settings_class_with_dictionary(self):
        pre_dict = dict()
        wavelengths = np.arange(800, 950, 10)
        volume = np.zeros([2, 2, 3])
        pre_dict[Tags.WAVELENGTHS] = wavelengths
        pre_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = volume
        settings = Settings(pre_dict)
        assert (settings[Tags.WAVELENGTHS] == pre_dict[Tags.WAVELENGTHS]).all()
        assert (settings[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] == pre_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE]).all()

    @unittest.expectedFailure
    def test_add_illegal_key_to_settings(self):
        settings = Settings()
        settings[{'a':'b'}] = "Test"

    def test_contains_statement(self):
        settings = Settings()
        settings[('string', str)] = "Test"

        assert ('string', str) in settings
        assert 'string' in settings
        assert ('string2', str) not in settings

        del settings[('string', str)]

        assert ('string', str) not in settings
        assert 'string' not in settings
        assert ('string2', str) not in settings

    def test_save_and_load(self):
        path = "testfile.hdf5"
        # ensure a clean file
        if os.path.exists(path):
            os.remove(path)

        settings = Settings()
        settings.add_minimal_meta_information()
        settings.add_minimal_optical_properties()
        settings.add_acoustic_properties()
        settings.add_reconstruction_properties()

        settings.save(path)

        settings_2 = Settings()
        settings_2.load(path)

        assert_equals_recursive(settings, settings_2)

        # clean up
        if os.path.exists(path):
            os.remove(path)
