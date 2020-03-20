import unittest
from utils import load_absorption_spectra_numpy_array


class TestTissueProperties(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_find_absorption_spectra(self):
        absorption_spectra = load_absorption_spectra_numpy_array()
        print(absorption_spectra)
        assert (absorption_spectra is not None)
