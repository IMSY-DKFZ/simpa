import unittest
from ippai.simulate.tissue_properties import TissueProperties
from utils import SPECTRAL_LIBRARY


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_find_absorption_spectra(self):

        print(type(SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN))

        settings = {
            'hb': TissueProperties.Chromophore(SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN, 0.5, 0.5,
                                                200.0, 1.2, 0.4, 0.9),
            'hbO2': TissueProperties.Chromophore(SPECTRAL_LIBRARY.OXYHEMOGLOBIN, 0.5, 0.5,
                                                 200.0, 1.2, 0.4, 0.9)
        }
        tp = TissueProperties(settings=settings)

        print(tp.get(700))
