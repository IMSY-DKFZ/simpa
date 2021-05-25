"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
from simpa.utils import TISSUE_LIBRARY
from simpa.utils.libraries.tissue_library import TissueLibrary
from simpa.utils.libraries.molecule_library import MolecularComposition
import inspect


class TestNoiseModels(unittest.TestCase):

    def test_noise_model_results(self, noise_model, expected_mean, expected_std,
                                 error_margin):
        for (method_name, method) in self.get_all_tissue_library_methods():
            total_volume_fraction = 0
            for molecule in method(TISSUE_LIBRARY):
                total_volume_fraction += molecule.volume_fraction
            self.assertAlmostEqual(total_volume_fraction, 1.0, 3,
                                   f"Volume fraction not 1.0 +/- 0.001 for {method_name}")

    @staticmethod
    def get_all_tissue_library_methods():
        methods = []
        for method in inspect.getmembers(TissueLibrary, predicate=inspect.isfunction):
            if isinstance(method[1](TISSUE_LIBRARY), MolecularComposition):
                methods.append(method)
        return methods