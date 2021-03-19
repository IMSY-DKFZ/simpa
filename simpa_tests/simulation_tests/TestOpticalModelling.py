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
import numpy as np

from simpa.core.optical_simulation.optical_modelling import extract_diffuse_reflectance
from simpa.utils.tags import Tags


class TestOpticalModelling(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = (100, 50, 200)
        self.seed = 1234
        np.random.seed(self.seed)
        self.wavelength = 700
        self.volumes = {Tags.OPTICAL_MODEL_FLUENCE: {self.wavelength: np.random.random_sample(self.dims)},
                        Tags.OPTICAL_MODEL_INITIAL_PRESSURE: {self.wavelength: np.random.random_sample(self.dims)}}
        self.settings = {Tags.WAVELENGTH: self.wavelength}

    def test_extract_diffuse_reflectance_tag(self):
        volumes = extract_diffuse_reflectance(self.settings, self.volumes)
        self.assertTrue(Tags.OPTICAL_MODEL_DIFFUSE_REFLECTANCE in volumes, f"Could not find tag in volumes after "
                                                                           f"extracting diffuse reflectance: "
                                                                           f"{Tags.OPTICAL_MODEL_DIFFUSE_REFLECTANCE}")
        self.assertTrue(Tags.SURFACE_LAYER_POSITION in volumes, f"Could not find tag in volumes after "
                                                                f"extracting diffuse reflectance: "
                                                                f"{Tags.SURFACE_LAYER_POSITION}")

    def test_extract_diffuse_reflectance_values(self):
        volumes = extract_diffuse_reflectance(self.settings, self.volumes)
        surf_pos = volumes[Tags.SURFACE_LAYER_POSITION][self.wavelength]
        volumes_to_check = {Tags.OPTICAL_MODEL_FLUENCE: volumes[Tags.OPTICAL_MODEL_FLUENCE][self.wavelength],
                            Tags.OPTICAL_MODEL_INITIAL_PRESSURE: volumes[Tags.OPTICAL_MODEL_INITIAL_PRESSURE][self.wavelength]}
        for key in volumes_to_check:
            volume = volumes_to_check[key]
            surface = volume[surf_pos]
            all_values_equal = np.all((self.volumes[key][self.wavelength][..., 1:] - volume[..., 1:]) == 0)
            self.assertTrue(np.all(surface) == 0, "Found non-zero value in surface of volume after extracting diffuse "
                                                  f"reflectance: {key}")
            self.assertTrue(all_values_equal, "Not all values in volume are equal to originals after extracting "
                                              f"diffuse reflectance: {key}")

    def test_extract_diffuse_reflectance_shape(self):
        volumes = extract_diffuse_reflectance(self.settings, self.volumes)
        dr_shape = volumes[Tags.OPTICAL_MODEL_DIFFUSE_REFLECTANCE][self.wavelength].shape
        fluence_surf_shape = volumes[Tags.OPTICAL_MODEL_FLUENCE][self.wavelength].shape[:2]
        self.assertTrue(dr_shape == fluence_surf_shape)


if __name__ == '__main__':
    unittest.main()
