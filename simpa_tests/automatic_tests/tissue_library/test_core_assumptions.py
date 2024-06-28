# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import TISSUE_LIBRARY
from simpa.utils.libraries.tissue_library import TissueLibrary
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import calculate_oxygenation, calculate_bvf
import inspect
import numpy as np


class TestCoreAssumptions(unittest.TestCase):

    def test_volume_fractions_sum_to_less_or_equal_one(self):
        for (method_name, method) in self.get_all_tissue_library_methods():
            total_volume_fraction = 0
            for molecule in method(TISSUE_LIBRARY):
                total_volume_fraction += molecule.volume_fraction
            self.assertAlmostEqual(total_volume_fraction, 1.0, 3,
                                   f"Volume fraction not 1.0 +/- 0.001 for {method_name}")

    def test_bvf_and_oxygenation_consistency(self):
        # blood_volume_fraction (bvf) and oxygenation of tissue classes defined
        #         # as input have to be the same as the calculated ones

        def compare_input_with_calculations(test_tissue, oxy, bvf):
            calculated_bvf = calculate_bvf(test_tissue)
            calculated_sO2 = calculate_oxygenation(test_tissue)
            if bvf < 1e-10:
                assert calculated_sO2 is None
                assert abs(bvf - calculated_bvf) < 1e-10
            else:
                assert abs(oxy - calculated_sO2) < 1e-10
                assert abs(bvf - calculated_bvf) < 1e-10

        # Test edge cases and a random one
        oxy_values = [0., 0., 1., 1., np.random.random()]
        bvf_values = [0., 1., 0., 1., np.random.random()]
        for oxy in oxy_values:
            # assert blood only with varying oxygenation_values
            test_tissue = TISSUE_LIBRARY.blood(oxygenation=oxy)
            compare_input_with_calculations(test_tissue, oxy, 1.)
            # assert all other tissue classes with varying oxygenation- and bvf_values
            for bvf in bvf_values:
                for (_, method) in self.get_all_tissue_library_methods():
                    args = inspect.getfullargspec(method).args
                    if "background_oxy" in args and "blood_volume_fraction" in args:
                        test_tissue = method(TISSUE_LIBRARY, background_oxy=oxy, blood_volume_fraction=bvf)
                        compare_input_with_calculations(test_tissue, oxy, bvf)

    @staticmethod
    def get_all_tissue_library_methods():
        methods = []
        for method in inspect.getmembers(TissueLibrary, predicate=inspect.isfunction):
            if isinstance(method[1](TISSUE_LIBRARY), MolecularComposition):
                methods.append(method)
        return methods
