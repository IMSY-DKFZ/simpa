import unittest
import matplotlib.pyplot as plt
from ippai.simulate import Tags
from ippai.simulate.models.optical_model import run_optical_forward_model
import numpy as np
import os


class TestInifinitesimalSlabExperiment(unittest.TestCase):

    def setUp(self):
        """
        This is not a completely autonomous test case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """
        print("setUp")
        self.settings = {
            Tags.WAVELENGTHS: [800], #np.arange(700, 951, 10),
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "TestVolume",
            Tags.SIMULATION_PATH: "/media/kris/6TB_Hard_Drive/mcx_test/",
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
            Tags.OPTICAL_MODEL_BINARY_PATH: "/media/kris/6TB_Hard_Drive/mcx_test/mcx",
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 0.5,
            Tags.OPTICAL_MODEL:Tags.MODEL_MCX
        }

        self.volume_path = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/" \
                           + self.settings[Tags.VOLUME_NAME] + "_" + str(self.settings[Tags.WAVELENGTH]) + ".npz"

        folder_name = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.xy_dim = 27
        self.z_dim = 100
        self.volume = np.zeros((self.xy_dim, self.xy_dim, self.z_dim, 3))

        self.volume[:, :, :, 0] = 1e-10
        self.volume[:, :, :, 1] = 1e-10
        self.volume[:, :, :, 2] = 0

    def tearDown(self):
        print("tearDown")
    #
    def test_both(self):
        self.perform_test(10, [0, 1], np.e, 0.5)
    #
    # def test_both_double_width(self):
    #     self.perform_test(20, [0, 1], np.e ** 2, 0.5)

    # def test_isotropic_scattering(self):
    #     self.perform_test(10, 1, np.e, 1)
    #
    # def test_isotropic_scattering_double_width(self):
    #     self.perform_test(20, 1, np.e ** 2, 1)
    #
    # def test_anisotropic_scattering(self):
    #     self.volume[:, :, :, 2] = 0.9
    #     self.perform_test(10, 1, np.e, 1)
    #
    # def test_absorption(self):
    #     self.perform_test(10, 0, np.e, 1)
    #
    # def test_absorption_double_width(self):
    #     self.perform_test(20, 0, np.e ** 2, 1)

    def perform_test(self, distance=10, volume_idx=0, decay_ratio=np.e, volume_value=1.0):
        self.volume[int(self.xy_dim / 2), int(self.xy_dim / 2),
                    int((self.z_dim / 2) - distance):int((self.z_dim / 2) + distance),
                    volume_idx] = volume_value

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )

        self.assertDecayRatio(expected_decay_ratio=decay_ratio)

    def assertDecayRatio(self, expected_decay_ratio=np.e):
        optical_path = run_optical_forward_model(self.settings, self.volume_path)
        fluence = np.load(optical_path)['fluence']
        half_dim = int(self.xy_dim / 2)
        decay_ratio = np.sum(fluence[half_dim, half_dim, 10]) / np.sum(fluence[half_dim, half_dim, 90])
        print(np.sum(fluence[half_dim, half_dim, 10]))
        print(np.sum(fluence[half_dim, half_dim, 90]))
        print("measured:", decay_ratio, "expected:", expected_decay_ratio)
        print("ratio:", decay_ratio / expected_decay_ratio)
        self.assertAlmostEqual(decay_ratio, expected_decay_ratio, 1)


if __name__ == "__main__":
    unittest.main()
