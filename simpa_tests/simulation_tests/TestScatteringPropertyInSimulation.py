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
from simpa.utils import Tags, Settings, generate_dict_path
from simpa.core import run_optical_forward_model
from simpa.io_handling import load_data_field, save_hdf5
import numpy as np
import os


"""
This Script uses the Lambert-Beer law to test mcx or mcxyz for the correct
attenuation of a photon beam passing through a thin absorbing and/or scattering
slab.

All tests test a scenario, where a pencil source in z-dir in the middle 
of the xy-plane emits photons in a 27x27x100 medium.
In the middle of the medium is an absorbing and/or scattering slab of 1 pixel in the xy-plane and some distance in z-dir.
In all tests, the fluence in the middle of the xy-plane and in z-dir the pixels 10 and 90 is measured.
So basically, the fluence after passing the slab with total attenuation (mua+mus) is measured.
For instance, we expect, that the fluence decreases by a factor of e^-1 if mua+mus=0.1 mm^-1 and the slab is 10mm long.

Usage of this script:
The script has to be in the same folder as the mcx executable binary.
If this is met, the script can just be run.

Use the test functions to test the specific cases that are explained in the respective tests.
Please read the description of every test and run them one after the other.
Be aware that by running multiple tests at once, the previous tests are overwritten.
"""


@unittest.skip("skipping local simulation tests")
class TestInifinitesimalSlabExperiment(unittest.TestCase):

    def setUp(self):
        """
        This is not a completely autonomous simpa_tests case yet.
        If run on another pc, please adjust the SIMULATION_PATH and MCX_BINARY_PATH.
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
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
            Tags.OPTICAL_MODEL_BINARY_PATH: MCX_BINARY_PATH,
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SPACING_MM: 0.5,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL
        }

        self.settings = Settings(self.settings)

        folder_name = self.settings[Tags.SIMULATION_PATH]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.settings[Tags.SIMPA_OUTPUT_PATH] = folder_name + "/" + self.settings[Tags.VOLUME_NAME] + ".hdf5"

        self.xy_dim = 27
        self.z_dim = 100
        self.volume = np.zeros((self.xy_dim, self.xy_dim, self.z_dim, 4))

        self.volume[:, :, :, 0] = 1e-10  # Background values for mua
        self.volume[:, :, :, 1] = 1e-10  # Background values for mua
        self.volume[:, :, :, 2] = 0
        self.volume[:, :, :, 3] = 1  # Pseudo Gruneisen parameter

    def tearDown(self):
        print("tearDown")
        os.remove(self.settings[Tags.SIMPA_OUTPUT_PATH])

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

    def perform_test(self, distance=10, volume_idx: (int, list) = 0, decay_ratio=np.e, volume_value=1.0):

        # Define the volume of the thin slab

        self.volume[int(self.xy_dim / 2) - 1, int(self.xy_dim / 2) - 1,
                    int((self.z_dim / 2) - distance):int((self.z_dim / 2) + distance),
                    volume_idx] = volume_value
        print(int(self.xy_dim/2))

        wavelength = self.settings[Tags.WAVELENGTH]
        # Save the volume

        optical_properties = {
            Tags.PROPERTY_ABSORPTION_PER_CM: {wavelength: self.volume[:, :, :, 0]},
            Tags.PROPERTY_SCATTERING_PER_CM: {wavelength: self.volume[:, :, :, 1]},
            Tags.PROPERTY_ANISOTROPY: {wavelength: self.volume[:, :, :, 2]},
            Tags.PROPERTY_GRUNEISEN_PARAMETER: {wavelength: self.volume[:, :, :, 3]},
        }

        optical_properties_path = generate_dict_path(Tags.SIMULATION_PROPERTIES, self.settings[Tags.WAVELENGTH])

        save_hdf5(optical_properties, self.settings[Tags.SIMPA_OUTPUT_PATH], optical_properties_path)

        self.assertDecayRatio(expected_decay_ratio=decay_ratio)

    def assertDecayRatio(self, expected_decay_ratio=np.e):
        run_optical_forward_model(self.settings)
        fluence = load_data_field(self.settings[Tags.SIMPA_OUTPUT_PATH], Tags.OPTICAL_MODEL_FLUENCE,
                                  self.settings[Tags.WAVELENGTH])
        half_dim = int(self.xy_dim / 2) - 1
        decay_ratio = np.sum(fluence[half_dim, half_dim, 10]) / np.sum(fluence[half_dim, half_dim, 90])
        print(np.sum(fluence[half_dim, half_dim, 10]))
        print(np.sum(fluence[half_dim, half_dim, 90]))
        print("measured:", decay_ratio, "expected:", expected_decay_ratio)
        print("ratio:", decay_ratio / expected_decay_ratio)
        self.assertAlmostEqual(decay_ratio, expected_decay_ratio, delta=0.2)
