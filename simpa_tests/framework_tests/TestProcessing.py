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
from simpa.process.signal_processing import apply_b_mode, get_apodization_factor
from simpa.process.preprocess_images import normalize, reconstruction_mode_transformation
from simpa.utils.tags import Tags
import unittest
import numpy as np
import torch


class TestProcessing(unittest.TestCase):

    def setUp(self):
        print("setUp")
        self.test_array = np.array([-1.2, 0, 3.2, 255])
        self.time_series_data = torch.tensor([[1, 3], [8, 12]], device='cpu')
        self.expected_differential = torch.tensor([[2., 0.], [4., 0.]], device='cpu')
        self.test_image = np.array([[-1.2, 0], [3., 255]])

    def tearDown(self):
        print("tearDown")

    def test_normalization(self):
        print("test normalization")
        normalized = normalize(self.test_array)

        # check input and output sizes
        assert normalized.shape == self.test_array.shape, "shapes have changed"

        # check if values are in range [0,1]
        assert ((0 <= normalized) & (1 >= normalized)).all(), "normalized values are not between 0 and 1"

        # check if normalization values are correct
        assert np.equal(normalized, np.array([0, (0 + 1.2)/(255+1.2), (3.2 + 1.2)/(255+1.2), 1])).all(), \
            "normalization values are incorrect"

    def test_reconstruction_mode_transformation(self):
        print("test reconstruction mode transformation")

        pressure = reconstruction_mode_transformation(self.time_series_data)
        pressure = reconstruction_mode_transformation(self.time_series_data, Tags.RECONSTRUCTION_MODE_PRESSURE)
        assert torch.equal(pressure, self.time_series_data), "there should be no change when using pressure mode"

        differential = reconstruction_mode_transformation(self.time_series_data, Tags.RECONSTRUCTION_MODE_DIFFERENTIAL)
        assert torch.equal(differential, self.expected_differential), "computed and expected differential don't match"

    def test_apodization_factors(self):
        print("test apodization factors creation")

        # Hann
        factors = get_apodization_factor(Tags.RECONSTRUCTION_APODIZATION_HANN, (2, 1), 10, device=torch.device('cpu'))
        expected = torch.tensor([[[0.0000000000, 0.0954914987, 0.3454915285, 0.6545085311, 0.9045085311,
                                   1.0000000000, 0.9045084715, 0.6545085311, 0.3454914391, 0.0954913795]],
                                 [[0.0000000000, 0.0954914987, 0.3454915285, 0.6545085311, 0.9045085311,
                                   1.0000000000, 0.9045084715, 0.6545085311, 0.3454914391, 0.0954913795]]])
        assert torch.equal(factors, expected), "computed Hann apodization factors don't match expected ones"

        # Hamming
        factors = get_apodization_factor(Tags.RECONSTRUCTION_APODIZATION_HAMMING,
                                         (2, 1), 10, device=torch.device('cpu'))
        expected = torch.tensor([[[0.0800000131, 0.1678521931, 0.3978522122, 0.6821478605, 0.9121478796,
                                   1.0000000000, 0.9121478200, 0.6821478605, 0.3978521228, 0.1678520739]],
                                 [[0.0800000131, 0.1678521931, 0.3978522122, 0.6821478605, 0.9121478796,
                                   1.0000000000, 0.9121478200, 0.6821478605, 0.3978521228, 0.1678520739]]])
        assert torch.equal(factors, expected), "computed Hamm apodization factors don't match expected ones"

        # Box
        factors = get_apodization_factor(Tags.RECONSTRUCTION_APODIZATION_BOX,
                                         (2, 1), 10, device=torch.device('cpu'))
        expected = torch.ones((2, 1, 10))
        assert torch.equal(factors, expected), "computed Box apodization factors don't match expected ones"

    def test_envelope_detection(self):
        print("test envelope detection")

        # absolute value
        abs = apply_b_mode(self.test_image, method=Tags.RECONSTRUCTION_BMODE_METHOD_ABS)
        expected_abs = np.array([[1.2, 0.], [3., 255.]])
        assert np.equal(abs, expected_abs).all(), "computed absolute array and expected don't match"

        # Hilbert transform
        hilbert = apply_b_mode(self.test_image, method=Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM)
        expected_hilbert = np.array([[1.2, 0.], [3., 255.]])
        assert np.equal(hilbert, expected_hilbert).all(), "computed hilbert transform array and expected don't match"


if __name__ == '__main__':
    test = TestProcessing()
    test.setUp()
    test.test_normalization()
    test.test_reconstruction_mode_transformation()
    test.test_apodization_factors()
    test.test_envelope_detection()
    test.tearDown()
