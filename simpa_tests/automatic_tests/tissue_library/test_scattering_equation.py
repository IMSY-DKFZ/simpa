import unittest
from simpa.utils import ScatteringSpectrumLibrary


class TestScatteringEquation(unittest.TestCase):
    """
    This test was written to ensure #412 does not reappear again
    """
    def test_scattering_equation_is_correct(self):
        lib = ScatteringSpectrumLibrary()

        # Testing both mixed
        test_spectrum = lib.scattering_from_rayleigh_and_mie_theory("test",
                                                    mus_at_500_nm=10,
                                                    fraction_rayleigh_scattering=0.5,
                                                    mie_power_law_coefficient=0.1)

        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(500), 10.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(750), 5.788976824948744, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(1000), 4.977664957684037, 5)

        # Testing Rayleigh 0
        test_spectrum = lib.scattering_from_rayleigh_and_mie_theory("test",
                                                                      mus_at_500_nm=100,
                                                                      fraction_rayleigh_scattering=0.0,
                                                                      mie_power_law_coefficient=10)

        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(500), 100.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(750), 1.7341529915832612, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(1000), 0.09765625, 5)

        # testing no decay at all
        test_spectrum = lib.scattering_from_rayleigh_and_mie_theory("test",
                                                                    mus_at_500_nm=100,
                                                                    fraction_rayleigh_scattering=0.0,
                                                                    mie_power_law_coefficient=0)

        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(500), 100.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(750), 100.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(1000), 100.0, 5)

        # testing only Rayleigh
        test_spectrum = lib.scattering_from_rayleigh_and_mie_theory("test",
                                                                    mus_at_500_nm=100,
                                                                    fraction_rayleigh_scattering=0.75,
                                                                    mie_power_law_coefficient=0)

        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(500), 100.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(750), 39.81481481481482, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(1000), 29.6875, 5)

        # testing scaling of zero scattering
        test_spectrum = lib.scattering_from_rayleigh_and_mie_theory("test",
                                                                    mus_at_500_nm=0,
                                                                    fraction_rayleigh_scattering=0.2342345,
                                                                    mie_power_law_coefficient=0.123123)

        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(500), 0.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(750), 0.0, 5)
        self.assertAlmostEqual(test_spectrum.get_value_for_wavelength(1000), 0.0, 5)