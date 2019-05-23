import unittest
from ippai.simulate import Tags
from ippai.simulate.models.optical_model import run_optical_forward_model
import matplotlib.pylab as plt
import numpy as np
import os

class TestInifinitesimalSlabExperiment(unittest.TestCase):

    def setUp(self):
        """
        This is not a completely autonomous test case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """

        self.settings = {
            Tags.WAVELENGTHS: [800], #np.arange(700, 951, 10),
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "Slab",
            Tags.SIMULATION_PATH: "/home/janek/simulation_test/",
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5000000,
            Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/mitk-superbuild/MITK-build/bin/MitkMCxyz",
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 0.5
        }

        self.volume_path = self.settings[Tags.SIMULATION_PATH] + self.settings[Tags.VOLUME_NAME] + "/" \
                           + self.settings[Tags.VOLUME_NAME] + "_" + str(self.settings[Tags.WAVELENGTH]) + ".npz"
        self.xy_dim = 7
        self.z_dim = 100
        self.volume = np.zeros((self.xy_dim, self.xy_dim, self.z_dim, 3))

        self.volume[:, :, :, 0] = 1e-10
        self.volume[:, :, :, 1] = 1e-10
        self.volume[:, :, :, 2] = 0

    def tearDown(self):
        print("tearDown")

    def test_isotropic_scattering(self):
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-10):int((self.z_dim/2)+10), 1] = 1

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio()

    def test_isotropic_scattering_double_width(self):
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-20):int((self.z_dim/2)+20), 1] = 1

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio(expected_decay_ratio=np.e**2)

    def test_isotropic_scattering_triple_width(self):
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-30):int((self.z_dim/2)+30), 1] = 1

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio(expected_decay_ratio=np.e**3)

    def test_anisotropic_scattering(self):
        self.volume[:, :, :, 2] = 0.9
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-10):int((self.z_dim/2)+10), 1] = 10

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio()

    def test_anisotropic_scattering_double_width(self):
        self.volume[:, :, :, 2] = 0.9
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-20):int((self.z_dim/2)+20), 1] = 10

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio(expected_decay_ratio=np.e**2)

    def test_anisotropic_scattering_triple_width(self):
        self.volume[:, :, :, 2] = 0.9
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2), int((self.z_dim/2)-30):int((self.z_dim/2)+30), 1] = 10

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )
        self.assertDecayRatio(expected_decay_ratio=np.e**3)

    def assertDecayRatio(self, expected_decay_ratio=np.e):
        optical_path = run_optical_forward_model(self.settings, self.volume_path)
        fluence = np.load(optical_path)['fluence']
        decay_ratio = fluence[int(self.xy_dim / 2), int(self.xy_dim / 2), 10] / fluence[
            int(self.xy_dim / 2), int(self.xy_dim / 2), 90]
        print(fluence[int(self.xy_dim / 2), int(self.xy_dim / 2), 10])
        print(fluence[int(self.xy_dim / 2), int(self.xy_dim / 2), 90])
        print("measured:", decay_ratio, "expected:", expected_decay_ratio)
        print("ratio:", decay_ratio / expected_decay_ratio)
        self.assertAlmostEqual(decay_ratio / expected_decay_ratio, 1, 1)


if __name__ == "__main__":
    unittest.main()
