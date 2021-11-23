"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
from simpa.io_handling import load_hdf5
from simpa.io_handling import save_hdf5
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.structure_library import Background
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY, SPECTRAL_LIBRARY
from simpa_tests.test_utils import assert_equals_recursive
import os


class TestIOHandling(unittest.TestCase):

    def test_write_and_read_default_dictionary(self):
        save_dictionary = dict()
        settings = Settings()
        settings['Test'] = 'test2'
        save_dictionary[Tags.SETTINGS] = settings
        try:
            save_hdf5(save_dictionary, "test.hdf5")
            read_dictionary = load_hdf5("test.hdf5")
        except Exception as e:
            raise e
        finally:
            # clean up after test
            if os.path.exists("test.hdf5"):
                os.remove("test.hdf5")
        assert_equals_recursive(save_dictionary, read_dictionary)

    def test_write_and_read_structure_dictionary(self):

        save_dictionary = Settings()
        settings = Settings()
        settings[Tags.SPACING_MM] = 0.5
        settings[Tags.DIM_VOLUME_X_MM] = 10
        settings[Tags.DIM_VOLUME_Y_MM] = 10
        settings[Tags.DIM_VOLUME_Z_MM] = 10
        save_dictionary[Tags.SETTINGS] = settings

        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
        structure_settings = Settings()
        structure_settings["background"] = background_dictionary

        save_dictionary[Tags.STRUCTURES] = structure_settings
        save_dictionary["test_dictionary"] = {"test_spectrum": SPECTRAL_LIBRARY.WATER}


        try:
            save_hdf5(save_dictionary, "test.hdf5")
            read_dictionary = load_hdf5("test.hdf5")
        except Exception as e:
            raise e
        finally:
            # clean up after test
            if os.path.exists("test.hdf5"):
                os.remove("test.hdf5")

        assert_equals_recursive(save_dictionary, read_dictionary)
