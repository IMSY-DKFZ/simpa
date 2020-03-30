import unittest
from simulate.tissue_properties import TissueProperties
from utils import TISSUE_LIBRARY
from utils import SPECTRAL_LIBRARY
from utils.serialization import IPPAIJSONSerializer
import json
import os


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_find_absorption_spectra(self):

        print(type(SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN))
        tp = TissueProperties(settings=TISSUE_LIBRARY.get_arterial_blood_settings())
        print(tp.get(798))

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

