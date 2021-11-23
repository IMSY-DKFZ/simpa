# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import simpa as sp


class TestProcessing(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_numpy_arrays_same_size(self):
        array_1 = np.random.random((5, 6, 7))
        array_2 = np.random.random((5, 6, 7))
        array_3 = np.random.random((5, 6, 7))

        sp.assert_equal_shapes([array_1, array_2, array_3])

    @unittest.expectedFailure
    def test_numpy_array_not_same_size(self):
        array_1 = np.random.random((5, 6, 7))
        array_2 = np.random.random((5, 6, 7))
        array_3 = np.random.random((5, 6, 6))

        sp.assert_equal_shapes([array_1, array_2, array_3])

    @unittest.expectedFailure
    def test_numpy_arrays_not_same_size(self):
        array_1 = np.random.random((5, 7, 7))
        array_2 = np.random.random((5, 6, 7))
        array_3 = np.random.random((4, 6, 6))

        sp.assert_equal_shapes([array_1, array_2, array_3])

    def test_numpy_2_arrays_same_size(self):
        array_1 = np.random.random((5, 6, 7))
        array_2 = np.random.random((5, 6, 7))

        sp.assert_equal_shapes([array_1, array_2])

    def test_numpy_5_arrays_same_size(self):
        array_1 = np.random.random((5, 6, 7))
        array_2 = np.random.random((5, 6, 7))
        array_3 = np.random.random((5, 6, 7))
        array_4 = np.random.random((5, 6, 7))
        array_5 = np.random.random((5, 6, 7))

        sp.assert_equal_shapes([array_1, array_2, array_3, array_5, array_4])

    def test_numpy_array_same_size(self):
        array_1 = np.random.random((5, 6, 7))
        sp.assert_equal_shapes([array_1])

    def test_normal_array_is_normal(self):
        array = np.random.random((5, 6, 7))
        sp.assert_array_well_defined(array)

    @unittest.expectedFailure
    def test_array_is_negative(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = -3
        sp.assert_array_well_defined(array, assume_non_negativity=True)

    @unittest.expectedFailure
    def test_array_is_not_positive(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = 0
        sp.assert_array_well_defined(array, assume_positivity=True)

    @unittest.expectedFailure
    def test_array_contains_nan(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = np.nan
        sp.assert_array_well_defined(array)

    @unittest.expectedFailure
    def test_array_contains_pinf(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = np.inf
        sp.assert_array_well_defined(array)

    @unittest.expectedFailure
    def test_array_contains_ninf(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = -np.inf
        sp.assert_array_well_defined(array)

    @unittest.expectedFailure
    def test_array_contains_none(self):
        array = np.random.random((5, 6, 7))
        array[3, 3, 2] = None
        sp.assert_array_well_defined(array)
