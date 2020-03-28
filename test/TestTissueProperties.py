import unittest
from simulate.tissue_properties import TissueProperties
from simulate.tissue_properties import get_epidermis_settings as settings_generator
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
        tp = TissueProperties(settings=settings_generator())
        print(tp.get(798))

    def test_dictionary_serialization(self):
        settings = settings_generator()
        tmp_json_filename = "test_settings.json"

        serializer = IPPAIJSONSerializer()

        with open(tmp_json_filename, "w") as json_file:
            json.dump(settings, json_file, indent="\t", default=serializer.default)

        with open(tmp_json_filename, "r+") as json_file:
            new_settings = json.load(json_file)

        print(new_settings)

        os.remove(tmp_json_filename)

