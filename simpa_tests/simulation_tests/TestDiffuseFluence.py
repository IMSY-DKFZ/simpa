# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
from simpa.utils import Tags
from simpa.core import run_optical_forward_model
import numpy as np
import os
import matplotlib.pyplot as plt


@unittest.skip("skipping local simulation tests")
class TestInifinitesimalSlabExperiment(unittest.TestCase):

    def setUp(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """

        SIMULATION_PATH = "/path/to/save/location"
        MCX_BINARY_PATH = "/path/to/mcx/binary/mcx.exe"     # On Linux systems, the .exe at the end must be omitted.
        VOLUME_NAME = "TestVolume"

        print("setUp")
        self.settings = {
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: VOLUME_NAME,
            Tags.SIMULATION_PATH: SIMULATION_PATH,
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: MCX_BINARY_PATH,
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 1,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.PROPERTY_ANISOTROPY: 0.9,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
            Tags.SIMPA_OUTPUT_PATH: "simpa_tests.hdf5"
        }

        self.volume_path = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/" \
                           + self.settings[Tags.VOLUME_NAME] + "_" + str(self.settings[Tags.WAVELENGTH]) + ".npz"

        folder_name = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.dim = 100
        self.volume = np.zeros((self.dim, self.dim, self.dim, 4))

        self.mua = 0.1
        self.mus = 100
        self.g = 0.9

        self.volume[:, :, :, 0] = self.mua
        self.volume[:, :, :, 1] = self.mus
        self.volume[:, :, :, 2] = self.g
        self.volume[:, :, :, 3] = 1

    def tearDown(self):
        print("tearDown")

    def test_fluence(self):
        self.perform_test(distance=self.dim/2, spacing=1)

    def test_spacing_short(self):
        self.perform_test(distance=5, spacing=0.1)

    def test_spacing_middle(self):
        self.perform_test(distance=self.dim/4, spacing=0.5)

    def test_spacing_long(self):
        self.perform_test(distance=self.dim, spacing=2)

    def diff_theory_fluence(self, r):
        """
        Calculates the fluence depending on the source-detector distance
        according to the diffusion approximation for the radiative transfer eq.
        right beneath the surface of a semi-finite homogeneous medium.
        :param r: radial distance between source and detector.
        :return: fluence at a point with source-detector distance r.
        """
        print(self.settings[Tags.OPTICAL_MODEL])
        if self.settings[Tags.OPTICAL_MODEL] == Tags.OPTICAL_MODEL_MCX:
            print("MCX: transfer to mm")
            mua = 0.1 * self.mua    # convert mua from cm^-1 to mm^-1
            mus = 0.1 * self.mus    # convert mus from cm^-1 to mm^-1
            spacing = self.settings[Tags.SPACING_MM]
        else:
            print("MCXYZ: transfer to cm")
            r = r / 10.0
            mua = self.mua
            mus = self.mus
            spacing = self.settings[Tags.SPACING_MM] / 10.0

        mus_prime = (1-self.g) * mus
        mu_tot = mus_prime + mua
        mu_eff = np.sqrt(3 * mua * mu_tot)
        D = 1 / (3 * (mua + mus_prime))
        z0 = 1 / mu_tot
        n = 1
        r_d = -1.44 * n ** -2 + 0.71 * n ** -1 + 0.668 + 0.0636 * n
        A = (1 + r_d) / (1 - r_d)
        zb = 2 * A * D

        # distance from point source inside the medium to the detector
        r1 = np.linalg.norm(np.asarray([0, 0, z0]) - np.asarray([r, 0, 0.5*spacing]))

        # distance from image point source above the medium to the detector
        r2 = np.linalg.norm(np.asarray([0, 0, -z0 - 2 * zb]) - np.asarray([r, 0, 0.5*spacing]))

        # fluence
        phi = 1 / (4*np.pi*D) * (np.exp(-mu_eff*r1) / r1 - np.exp(-mu_eff*r2) / r2)

        if self.settings[Tags.OPTICAL_MODEL] == Tags.OPTICAL_MODEL_MCXYZ:
            phi = phi / 100

        return phi

    def perform_test(self, distance, spacing):

        self.settings[Tags.SPACING_MM] = spacing

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 gamma=self.volume[:, :, :, 3]
                 )

        self.assertDiffusionTheory(distance)

    def assertDiffusionTheory(self, distance):
        optical_path = run_optical_forward_model(self.settings)
        fluence = np.load(optical_path)['fluence']
        number_of_measurements = np.arange(1, int(distance/self.settings[Tags.SPACING_MM]) + 1, 1)
        measurement_distances = number_of_measurements * self.settings[Tags.SPACING_MM]
        fluence_measurements = fluence[int(self.dim/2), (int(self.dim/2) - 1 + number_of_measurements), 0]

        fluence_measurements = fluence_measurements / 100

        diffusion_approx = self.diff_theory_fluence(measurement_distances)

        # Plot the results:

        fig, ax = plt.subplots()
        ax.scatter(measurement_distances, fluence_measurements, marker="o", c="r", label="Simulation")
        ax.plot(measurement_distances, diffusion_approx, label="Diffusion Approx.")
        ax.fill_between(measurement_distances,
                         diffusion_approx - 0.5*diffusion_approx,
                         diffusion_approx + 0.5*diffusion_approx,
                         alpha=0.2, label="Accepted Error Range")
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[0], handles[2], handles[1]]
        labels = [labels[0], labels[2], labels[1]]
        ax.set_yscale("log")
        plt.legend(handles, labels)
        plt.show()

        # Plot error
        #
        # plt.plot(measurement_distances, (diffusion_approx - fluence_measurements) / diffusion_approx)
        # plt.grid()
        # plt.show()

        # for sim, diff in zip(fluence_measurements, diffusion_approx):
        #     """
        #     if the fluence is smaller than 0.00001% of the source strength,
        #     we assume that the random fluctuation of the photons will cause a
        #     larger error.
        #     """
        #     if diff < 1e-7:
        #         continue
        #     else:
        #         """
        #         we simpa_tests for 50% of the expected value from the diffusion approx.
        #         """
        #         self.assertAlmostEqual(sim, diff, delta=0.5*diff)
