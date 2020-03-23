import unittest
from ippai.simulate.tissue_properties import TissueProperties
from ippai.simulate.tissue_properties import get_epidermis_settings as settings_generator
from utils import SPECTRAL_LIBRARY


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_find_absorption_spectra(self):

        print(type(SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN))

        tp = TissueProperties(settings=settings_generator())

        print(tp.get(798))
