# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import SegmentationClasses, MolecularCompositionGenerator
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.calculate import calculate_oxygenation
from simpa.utils.calculate import randomize_uniform
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.utils.calculate import positive_gauss
import numpy as np


class TestCalculationUtils(unittest.TestCase):

    def test_oxygenation_calculation(self):

        # Neither oxy nor deoxy:
        mcg = MolecularCompositionGenerator()
        mcg.append(MOLECULE_LIBRARY.fat(1.0))
        oxy_value = calculate_oxygenation(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert oxy_value is None

        mcg = MolecularCompositionGenerator()
        mcg.append(MOLECULE_LIBRARY.fat(1.0))
        mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(0.0))
        oxy_value = calculate_oxygenation(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert oxy_value is None

        # FULLY OXYGENATED CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MOLECULE_LIBRARY.fat(1.0))
        mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(1.0))
        fully_oxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_oxygenated)
        assert abs(oxy_value-1.0) < 1e-5, ("oxy value was not 1.0 but " + str(oxy_value))
        mcg.append(MOLECULE_LIBRARY.deoxyhemoglobin(0.0))
        fully_oxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_oxygenated)
        assert abs(oxy_value-1.0) < 1e-5, ("oxy value was not 1.0 but " + str(oxy_value))

        # FULLY DEOXYGENATED CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MOLECULE_LIBRARY.fat(1.0))
        mcg.append(MOLECULE_LIBRARY.deoxyhemoglobin(1.0))
        fully_deoxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_deoxygenated)
        assert abs(oxy_value) < 1e-5, ("oxy value was not 0.0 but " + str(oxy_value))
        mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(0.0))
        fully_deoxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_deoxygenated)
        assert abs(oxy_value) < 1e-5, ("oxy value was not 0.0 but " + str(oxy_value))

        # RANDOM CASES
        for i in range(100):
            oxy = np.random.random()
            deoxy = np.random.random()
            mcg = MolecularCompositionGenerator()
            mcg.append(MOLECULE_LIBRARY.fat(1.0))
            mcg.append(MOLECULE_LIBRARY.deoxyhemoglobin(deoxy))
            mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(oxy))
            sO2_value = calculate_oxygenation(mcg.get_molecular_composition(
                segmentation_type=SegmentationClasses.GENERIC))

            assert abs(sO2_value - (oxy / (oxy + deoxy))) < 1e-10

    def test_randomize(self):
        for _ in range(1000):
            lower = np.random.randint(0, 10000000)
            upper = lower + np.random.randint(0, 10000000)
            random = randomize_uniform(lower, upper)
            assert lower < random, "randomize_uniform produced a value outside the desired range"
            assert upper >= random, "randomize_uniform produced a value outside the desired range"

    def test_gruneisen_parameter_calculation(self):
        for temperature in range(0, 50, 1):
            gruneisen = calculate_gruneisen_parameter_from_temperature(temperature)
            assert isinstance(gruneisen, float), "Gruneisenparameter was not a float"
            assert gruneisen > 0, "Gruneisenparameter was negative"
            assert gruneisen < 1, "Gruneisenparameter was way too large"
    
    def test_positive_Gauss(self):
        for _ in range(1000):
            mean = np.random.rand(1)[0]
            std = np.random.rand(1)[0]
            random_value = positive_gauss(mean, std)
            assert random_value > float(0), "positive Gauss value outside the desired range and negative"
