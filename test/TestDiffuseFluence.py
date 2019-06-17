import unittest
from ippai.simulate import Tags
from ippai.simulate.models.optical_model import run_optical_forward_model
import numpy as np
import os
import matplotlib.pyplot as plt


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
            Tags.VOLUME_NAME: "homogeneous_cube",
            Tags.SIMULATION_PATH: "/media/kris/6TB_Hard_Drive/mcx_test/",
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
            Tags.OPTICAL_MODEL_BINARY_PATH: "/media/kris/6TB_Hard_Drive/mcx_test/mcx",
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 1,
            Tags.OPTICAL_MODEL:Tags.MODEL_MCX,
            Tags.PROPERTY_ANISOTROPY: 0.9
        }

        self.volume_path = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/" \
                           + self.settings[Tags.VOLUME_NAME] + "_" + str(self.settings[Tags.WAVELENGTH]) + ".npz"

        folder_name = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.dim = 100
        self.volume = np.zeros((self.dim, self.dim, self.dim, 3))

        self.mua = 0.1
        self.mus = 100
        self.g = 0.9

        self.volume[:, :, :, 0] = self.mua
        self.volume[:, :, :, 1] = self.mus
        self.volume[:, :, :, 2] = self.g

    def tearDown(self):
        print("tearDown")

    def test_fluence(self):
        self.perform_test(distance=self.dim/2)

    def diff_theory_fluence(self, r):
        """
        Calculates the fluence depending on the source-detector distance
        according to the diffusion approximation for the radiative transfer eq.
        right beneath the surface of a semi-finite homogeneous medium.
        :param r: radial distance between source and detector.
        :return: fluence [mm^-2] at a point with source-detector distance r.
        """
        mua = 0.1 * self.mua    # convert mua from cm^-1 to mm^-1
        mus = 0.1 * self.mus    # convert mus from cm^-1 to mm^-1
        mus_red = (1-self.g) * mus
        mu_tot = mus_red + mua
        mu_eff = np.sqrt(3 * mua * mu_tot)
        D = 1 / (3 * (mua + mus_red))
        z0 = 1/mu_tot
        n = 1
        r_d = -1.44 * n ** -2 + 0.71 * n ** -1 + 0.668 + 0.0636 * n
        A = (1 + r_d) / (1 - r_d)
        zb = 2 * A * D
        # distance from point source inside the medium to the detector
        r1 = np.linalg.norm(np.asarray([0, 0, z0]) - np.asarray([r, 0, 0.5]))
        # distance from image point source above the medium to the detector
        r2 = np.linalg.norm(np.asarray([0, 0, -z0 - 2 * zb]) - np.asarray([r, 0, 0.5]))

        # fluence
        phi = 1/(4*np.pi*D) *(np.exp(-mu_eff*r1)/r1 - np.exp(-mu_eff*r2)/r2)

        return phi



    def perform_test(self, distance):

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 )

        self.assertDiffusionTheory(distance)

    def assertDiffusionTheory(self, distance):
        optical_path = run_optical_forward_model(self.settings, self.volume_path)
        fluence = np.load(optical_path)['fluence']
        measurement_distances = np.arange(1, int(distance + 1), 1)
        fluence_measurements = fluence[int(self.dim/2), int(self.dim/2) - 1 + measurement_distances, 0]
        diffusion_approx = self.diff_theory_fluence(measurement_distances)

        # Plot the results:

        # plt.scatter(measurement_distances, fluence_measurements, marker="o", c="r")
        # plt.plot(measurement_distances, diffusion_approx)
        # plt.yscale("log")
        # plt.show()

        for sim, diff in zip(fluence_measurements, diffusion_approx):
            if diff < 1e-8:
                continue
            else:
                self.assertAlmostEqual(sim, diff, delta=0.5*diff)


if __name__ == "__main__":
    unittest.main()
