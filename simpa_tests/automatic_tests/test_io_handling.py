# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.io_handling import load_hdf5
from simpa.io_handling import save_hdf5
from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY, AbsorptionSpectrumLibrary
from simpa_tests.test_utils import assert_equals_recursive
from simpa.core.device_digital_twins import *
import os


class TestIOHandling(unittest.TestCase):

    def assert_save_and_read_dictionaries_equal(self, save_dict, save_string="test.hdf5"):
        try:
            save_hdf5(save_dict, save_string)
            read_dictionary = load_hdf5(save_string)
        except Exception as e:
            raise e
        finally:
            # clean up after test
            if os.path.exists(save_string):
                os.remove(save_string)
        assert_equals_recursive(save_dict, read_dictionary)

    def test_write_and_read_default_dictionary(self):
        save_dictionary = dict()
        settings = Settings()
        settings['Test'] = 'test2'
        save_dictionary[Tags.SETTINGS] = settings
        self.assert_save_and_read_dictionaries_equal(save_dictionary)

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
        save_dictionary["test_dictionary"] = {"test_spectrum": AbsorptionSpectrumLibrary().get_spectrum_by_name("Water")}

        self.assert_save_and_read_dictionaries_equal(save_dictionary)

    def test_write_and_read_devices(self):
        base_devices = [DigitalDeviceTwinBase, PhotoacousticDevice]
        pa_devices = [MSOTAcuityEcho, InVision256TF, RSOMExplorerP50]
        det_geometries = [CurvedArrayDetectionGeometry, LinearArrayDetectionGeometry, PlanarArrayDetectionGeometry]
        ill_geometries = [SlitIlluminationGeometry, GaussianBeamIlluminationGeometry, PencilArrayIlluminationGeometry,
                          PencilBeamIlluminationGeometry, DiskIlluminationGeometry, MSOTAcuityIlluminationGeometry,
                          MSOTInVisionIlluminationGeometry]
        for device in pa_devices:
            save_dictionary = Settings()
            save_dictionary[Tags.DIGITAL_DEVICE] = device
            self.assert_save_and_read_dictionaries_equal(save_dictionary)
