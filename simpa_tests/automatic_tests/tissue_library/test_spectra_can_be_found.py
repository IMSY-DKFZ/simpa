# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import AbsorptionSpectrumLibrary
from simpa.utils import ScatteringSpectrumLibrary
from simpa.utils import AnisotropySpectrumLibrary


class TestSpectraCanBeFound(unittest.TestCase):

    def test_absorption_spectra_valid(self):
        names = ["Copper_Sulphide", "Deoxyhemoglobin", "Fat", "Melanin", "Nickel_Sulphide",
                 "Oxyhemoglobin", "Skin_Baseline"]
        lib = AbsorptionSpectrumLibrary()
        for name in names:
            print(name)
            spectrum = lib.get_spectrum_by_name(name)
            self.assertEqual(name, spectrum.spectrum_name)

    @unittest.expectedFailure
    def test_absorption_spectra_invalid(self):
        AbsorptionSpectrumLibrary().get_spectrum_by_name("This does not exist")

    def test_scattering_spectra_valid(self):
        names = ["background_scattering", "blood_scattering", "bone_scattering",
                 "fat_scattering", "muscle_scattering"]
        lib = ScatteringSpectrumLibrary()
        for name in names:
            print(name)
            spectrum = lib.get_spectrum_by_name(name)
            self.assertEqual(name, spectrum.spectrum_name)

    @unittest.expectedFailure
    def test_scattering_spectra_invalid(self):
        ScatteringSpectrumLibrary().get_spectrum_by_name("This does not exist")

    def test_anisotropy_spectra_valid(self):
        names = ["Epidermis_Anisotropy"]
        lib = AnisotropySpectrumLibrary()
        for name in names:
            print(name)
            spectrum = lib.get_spectrum_by_name(name)
            self.assertEqual(name, spectrum.spectrum_name)

    @unittest.expectedFailure
    def test_anisotropy_spectra_invalid(self):
        AnisotropySpectrumLibrary().get_spectrum_by_name("This does not exist")

