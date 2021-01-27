# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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


"""
This Script uses the Lambert-Beer law to simpa_tests mcx or mcxyz for the correct
attenuation of a photon beam passing through a thin absorbing and/or scattering
slab.

All tests simpa_tests a scenario, where a pencil source in z-dir in the middle 
of the xy-plane emits photons in a 27x27x100 medium.
In the middle of the medium is an absorbing and/or scattering slab of 1 pixel in the xy-plane and some distance in z-dir.
In all tests, the fluence in the middle of the xy-plane and in z-dir the pixels 10 and 90 is measured.
So basically, the fluence after passing the slab with total attenuation (mua+mus) is measured.
For instance, we expect, that the fluence decreases by a factor of e^-1 if mua+mus=0.1 mm^-1 and the slab is 10mm long.

Usage of this script:
The script has to be in the same folder as the mcx executable binary.
If this is met, the script can just be run.

Use the simpa_tests functions to simpa_tests the specific cases that are explained in the respective tests.
Please read the description of every simpa_tests and run them one after the other.
Be aware that by running multiple tests at once, the previous tests are overwritten.
"""


@unittest.skip("skipping local simulation tests")
class TestInifinitesimalSlabExperiment(unittest.TestCase):

    def setUp(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MODEL_BINARY_PATH fields.
        :return:
        """
        print("setUp")
        self.settings = {
            Tags.WAVELENGTHS: [800],
            Tags.WAVELENGTH: 800,
            Tags.VOLUME_NAME: "TestVolume",
            Tags.SIMULATION_PATH: "/home/kris/hard_drive/mcx_test/simpa_tests",
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
            Tags.OPTICAL_MODEL_BINARY_PATH: "/home/kris/hard_drive/cami-experimental/PAI/MCX/mcx-master/bin/mcx",
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 0.5,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL
        }

        self.volume_path = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/" \
                           + self.settings[Tags.VOLUME_NAME] + "_" + str(self.settings[Tags.WAVELENGTH]) + ".npz"

        folder_name = self.settings[Tags.SIMULATION_PATH] + "/"+ self.settings[Tags.VOLUME_NAME] + "/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.xy_dim = 27
        self.z_dim = 100
        self.volume = np.zeros((self.xy_dim, self.xy_dim, self.z_dim, 4))

        self.volume[:, :, :, 0] = 1e-10  # Background values for mua
        self.volume[:, :, :, 1] = 1e-10  # Background values for mua
        self.volume[:, :, :, 2] = 0
        self.volume[:, :, :, 3] = 1  # Pseudo Gruneisen parameter

    def tearDown(self):
        print("tearDown")

    def test_both(self):
        """
        Here, the slab is 10 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        The spacing is 0.5, so that the slab is 20 voxels long.
        We expect a decay ratio of e^1.
        """
        self.perform_test(10, [0, 1], np.e, 0.5)

    def test_both_double_width(self):
        """
        Here, the slab is 20 mm long, mua and mus are both used with values of 0.05 mm^-1, so that mua+mus=0.1 mm^-1.
        The spacing is 0.5, so that the slab is 40 voxels long.
        We expect a decay ratio of e^2.
        """
        self.perform_test(20, [0, 1], np.e ** 2, 0.5)

    def test_isotropic_scattering(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        The spacing is 0.5, so that the slab is 20 voxels long.
        We expect a decay ratio of e^1.
        """
        self.perform_test(10, 1, np.e, 1)

    def test_isotropic_scattering_double_width(self):
        """
        Here, the slab is 20 mm long, only mus is used with a value of 0.1 mm^-1.
        The spacing is 0.5, so that the slab is 40 voxels long.
        We expect a decay ratio of e^2.
        """
        self.perform_test(20, 1, np.e ** 2, 1)

    def test_anisotropic_scattering(self):
        """
        Here, the slab is 10 mm long, only mus is used with a value of 0.1 mm^-1.
        The spacing is 0.5, so that the slab is 20 voxels long.
        The anisotropy of the scattering is 0.9.
        We expect a decay ratio of e^1.
        """
        self.volume[:, :, :, 2] = 0.9
        self.perform_test(10, 1, np.e, 1)

    def test_absorption(self):
        """
        Here, the slab is 10 mm long, only mua is used with a value of 0.1 mm^-1.
        The spacing is 0.5, so that the slab is 20 voxels long.
        We expect a decay ratio of e^1.
        """
        self.perform_test(10, 0, np.e, 1)

    def test_absorption_double_width(self):
        """
        Here, the slab is 20 mm long, only mua is used with a value of 0.1 mm^-1.
        The spacing is 0.5, so that the slab is 40 voxels long.
        We expect a decay ratio of e^2.
        """
        self.perform_test(20, 0, np.e ** 2, 1)

    def perform_test(self, distance=10, volume_idx=0, decay_ratio=np.e, volume_value=1.0):

        # Define the volume of the thin slab

        self.volume[int(self.xy_dim / 2) - 1, int(self.xy_dim / 2) - 1,
                    int((self.z_dim / 2) - distance):int((self.z_dim / 2) + distance),
                    volume_idx] = volume_value
        print(int(self.xy_dim/2))

        # Save the volume

        np.savez(self.volume_path,
                 mua=self.volume[:, :, :, 0],
                 mus=self.volume[:, :, :, 1],
                 g=self.volume[:, :, :, 2],
                 gamma=self.volume[:, :, :, 3]
                 )

        self.assertDecayRatio(expected_decay_ratio=decay_ratio)

    def assertDecayRatio(self, expected_decay_ratio=np.e):
        optical_path = run_optical_forward_model(self.settings, self.volume_path)
        fluence = np.load(optical_path)['fluence']
        absorption = np.load(self.volume_path)['mua']
        half_dim = int(self.xy_dim / 2) - 1
        decay_ratio = np.sum(fluence[half_dim, half_dim, 10]) / np.sum(fluence[half_dim, half_dim, 90])
        print(np.sum(fluence[half_dim, half_dim, 10]))
        print(np.sum(fluence[half_dim, half_dim, 90]))
        print("measured:", decay_ratio, "expected:", expected_decay_ratio)
        print("ratio:", decay_ratio / expected_decay_ratio)
        self.assertAlmostEqual(decay_ratio, expected_decay_ratio, delta=0.15)


if __name__ == "__main__":
    unittest.main()
