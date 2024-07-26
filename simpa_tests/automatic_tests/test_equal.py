# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
from simpa.utils.calculate import are_equal


class TestEqual(unittest.TestCase):

    def setUp(self):
        self.list1 = [1, 2, 3]
        self.list2 = [1, 2, 3]
        self.list_unqeual = [1, 2, 4]
        self.array1 = np.array([1, 2, 3])
        self.array2 = np.array([1, 2, 3])
        self.array_unequal = np.array([1, 2, 4])
        self.scalar1 = 1
        self.scalar2 = 1
        self.scalar3 = 2.0
        self.scalar4 = 2.0
        self.scalarunequal1 = 2
        self.scalarunequal2 = 1.0
        self.string1 = 'Test1'
        self.string2 = 'Test1'
        self.string_unequal = 'Test2'
        self.nested_list1 = [self.list1, self.array1]
        self.nested_list2 = [self.list2, self.array2]
        self.nested_list_unequal = [self.list_unqeual, self.array_unequal]

    def test_lists_are_equal(self):
        self.assertTrue(are_equal(self.list1, self.list2))

    @unittest.expectedFailure
    def test_lists_are_unequal(self):
        self.assertTrue(are_equal(self.list1, self.list_unqeual))

    def test_numpy_arrays_are_equal(self):
        self.assertTrue(are_equal(self.array1, self.array2))

    @unittest.expectedFailure
    def test_numpy_arrays_are_unequal(self):
        self.assertTrue(are_equal(self.array1, self.array_unequal))

    def test_scalars_are_equal(self):
        self.assertTrue(are_equal(self.scalar1, self.scalar2))
        self.assertTrue(are_equal(self.scalar3, self.scalar4))

    @unittest.expectedFailure
    def test_scalars_are_unequal(self):
        self.assertTrue(are_equal(self.scalar1, self.scalarunequal1))
        self.assertTrue(are_equal(self.scalar3, self.scalarunequal2))

    def test_strings_are_equal(self):
        self.assertTrue(are_equal(self.string1, self.string2))

    @unittest.expectedFailure
    def test_strings_are_unequal(self):
        self.assertTrue(are_equal(self.string1, self.string_unequal))

    def test_mixed_types_are_equal(self):
        self.assertTrue(are_equal(self.list1, self.array1))
        self.assertTrue(are_equal(self.array1, self.list1))

    @unittest.expectedFailure
    def test_mixed_types_are_unequal(self):
        self.assertTrue(are_equal(self.list1, self.scalar1))
        self.assertTrue(are_equal(self.array1, self.scalar1))
        self.assertTrue(are_equal(self.scalar1, self.scalar3))
        self.assertTrue(are_equal(self.string1, self.scalar1))

    def test_nested_lists_are_equal(self):
        self.assertTrue(are_equal(self.nested_list1, self.nested_list2))

    @unittest.expectedFailure
    def test_nested_lists_are_unequal(self):
        self.assertTrue(are_equal(self.nested_list1, self.nested_list_unequal))
