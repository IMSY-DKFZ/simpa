import unittest
from simulate.tissue_properties import TissueProperties
from utils import TISSUE_LIBRARY
from utils import SPECTRAL_LIBRARY
from utils.serialization import IPPAIJSONSerializer
import json
import os
import numpy as np


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_find_absorption_spectra(self):
        tp = TissueProperties(settings=TISSUE_LIBRARY.get_arterial_blood_settings())
        data = tp.get(798)
        print(data)

    def test_absorption_spectra_interpolation(self):
        for absorption_spectrum in SPECTRAL_LIBRARY:
            for i in range(600, 900, 1):
                wl_lower = absorption_spectrum.get_absorption_for_wavelength(i)
                wl_upper = absorption_spectrum.get_absorption_for_wavelength(i+1)
                wl_in_between = absorption_spectrum.get_absorption_for_wavelength(i+0.5)
                if wl_upper > wl_lower:
                    assert (wl_in_between < wl_upper) and (wl_in_between > wl_lower)
                elif wl_lower > wl_upper:
                    assert (wl_in_between > wl_upper) and (wl_in_between < wl_lower)
                elif np.abs(wl_lower - wl_upper) < 1e-5:
                    assert ((np.abs(wl_lower - wl_in_between) < 1e-5) and (np.abs(wl_lower - wl_in_between) < 1e-5))
                else:
                    raise ValueError("Ranges undefined!")

    def test_dictionary_serialization(self):
        settings = TISSUE_LIBRARY.get_arterial_blood_settings()
        tmp_json_filename = "test_settings.json"

        serializer = IPPAIJSONSerializer()

        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings, json_file, indent="\t", default=serializer.default)

        with open(tmp_json_filename, "r+") as json_file:
            new_settings = json.load(json_file)

        print(new_settings)

        os.remove(tmp_json_filename)

