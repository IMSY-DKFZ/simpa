# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import Tags, Settings
from simpa.utils.libraries.tissue_library import TissueLibrary
from simpa.utils.libraries.molecule_library import MolecularComposition
import inspect
import numpy as np

TEST_SETTINGS = Settings({
    # These parameters set the general properties of the simulated volume
    Tags.SPACING_MM: 1,
    Tags.DIM_VOLUME_Z_MM: 4,
    Tags.DIM_VOLUME_X_MM: 2,
    Tags.DIM_VOLUME_Y_MM: 7
})

class TestCoreAssumptions(unittest.TestCase):

    def test_volume_fractions_sum_to_less_or_equal_one(self):
        for (method_name, method) in self.get_all_tissue_library_methods():
            molecular_composition = method(TissueLibrary())
            tissue_composition = molecular_composition.get_properties_for_wavelength(TEST_SETTINGS, 800)
            total_volume_fraction = tissue_composition.volume_fraction
            self.assertTrue((np.abs(total_volume_fraction-1.0) < 1e-3).all(),
                                   f"Volume fraction not 1.0 +/- 0.001 for {method_name}")

    @staticmethod
    def get_all_tissue_library_methods():
        methods = []
        for method in inspect.getmembers(TissueLibrary, predicate=inspect.isfunction):
            if isinstance(method[1](TissueLibrary()), MolecularComposition):
                methods.append(method)
        return methods
