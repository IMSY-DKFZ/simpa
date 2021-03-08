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
from simpa.utils import calculate


class TestSurface(unittest.TestCase):
    def setUp(self) -> None:
        self.x_dim = 100
        self.y_dim = 50
        self.z_dim = 200
        self.vol_rand = np.random.rand(self.x_dim, self.y_dim, self.z_dim)
        self.vol_ones = np.ones_like(self.vol_rand)
        self.vol_zeros = np.zeros_like(self.vol_rand)
        self.volumes = [self.vol_rand, self.vol_ones, self.vol_zeros]
        self.axes = [0, 1, 2]
        np.random.seed(1234)

    def test_get_surface_from_volume_dims(self):
        for ax in self.axes:
            for vol in self.volumes:
                pos = calculate.get_surface_from_volume(vol, ax=ax)
                surf = vol[pos]
                vol_shape = list(vol.shape)
                vol_shape.pop(ax)
                expected_shape = tuple(vol_shape)
                surf = surf.reshape(expected_shape)
                self.assertTrue(surf.shape == expected_shape, f"Surface extracted along axis {ax} does not match "
                                                              f"expected shape: {expected_shape}")

    def test_get_surface_from_volume_position(self):
        for ax in self.axes:
            pos = calculate.get_surface_from_volume(self.vol_ones, ax=ax)
            self.assertTrue(np.all(pos[ax] == 0), f"Position of surface expected at 0 along axis {ax} for volume "
                                                  f"filled with ones, but got !=0 at {np.where(pos[ax]!=0)}")
        for ax in self.axes:
            pos = calculate.get_surface_from_volume(self.vol_zeros, ax=ax)
            self.assertTrue(np.all(pos[ax] == self.vol_zeros.shape[ax] - 1), f"Position of surface expected at "
                                                                         f"{self.vol_zeros.shape[ax]} along "
                                                                         f"axis {ax} for volume filled with ones, but "
                                                                         f"got different value at {np.where(pos[ax]!=self.vol_zeros.shape[ax])}")


if __name__ == "__main__":
    unittest.main()
