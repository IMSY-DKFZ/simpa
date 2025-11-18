# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import SegmentationClasses, MolecularCompositionGenerator
from simpa.utils.libraries.molecule_library import MoleculeLibrary
from simpa.utils.calculate import calculate_oxygenation, calculate_bvf
from simpa.utils.calculate import randomize_uniform
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.utils.calculate import positive_gauss
from simpa.utils.calculate import round_x5_away_from_zero
import numpy as np


class TestCalculationUtils(unittest.TestCase):

    def test_oxygenation_calculation(self):

        # Neither oxy nor deoxy:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        oxy_value = calculate_oxygenation(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert oxy_value is None

        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        mcg.append(MoleculeLibrary.oxyhemoglobin(0.0))
        oxy_value = calculate_oxygenation(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert oxy_value is None

        # FULLY OXYGENATED CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        mcg.append(MoleculeLibrary.oxyhemoglobin(1.0))
        fully_oxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_oxygenated)
        assert abs(oxy_value-1.0) < 1e-5, ("oxy value was not 1.0 but " + str(oxy_value))
        mcg.append(MoleculeLibrary.deoxyhemoglobin(0.0))
        fully_oxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_oxygenated)
        assert abs(oxy_value-1.0) < 1e-5, ("oxy value was not 1.0 but " + str(oxy_value))

        # FULLY DEOXYGENATED CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        mcg.append(MoleculeLibrary.deoxyhemoglobin(1.0))
        fully_deoxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_deoxygenated)
        assert abs(oxy_value) < 1e-5, ("oxy value was not 0.0 but " + str(oxy_value))
        mcg.append(MoleculeLibrary.oxyhemoglobin(0.0))
        fully_deoxygenated = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        oxy_value = calculate_oxygenation(fully_deoxygenated)
        assert abs(oxy_value) < 1e-5, ("oxy value was not 0.0 but " + str(oxy_value))

        # RANDOM CASES
        for _ in range(100):
            oxy = np.random.random()
            deoxy = np.random.random()
            mcg = MolecularCompositionGenerator()
            mcg.append(MoleculeLibrary.fat(1.0))
            mcg.append(MoleculeLibrary.deoxyhemoglobin(deoxy))
            mcg.append(MoleculeLibrary.oxyhemoglobin(oxy))
            sO2_value = calculate_oxygenation(mcg.get_molecular_composition(
                segmentation_type=SegmentationClasses.GENERIC))

            assert abs(sO2_value - (oxy / (oxy + deoxy))) < 1e-10

    def test_bvf_calculation(self):

        # Neither oxy nor deoxy:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        bvf_value = calculate_bvf(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert bvf_value == 0

        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.fat(1.0))
        mcg.append(MoleculeLibrary.oxyhemoglobin(0.0))
        bvf_value = calculate_bvf(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
        assert bvf_value == 0

        # Just oxyhemoglobin CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.oxyhemoglobin(1.0))
        oxy_hemo = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        bvf_value = calculate_bvf(oxy_hemo)
        assert bvf_value == 1.0
        mcg.append(MoleculeLibrary.deoxyhemoglobin(0.0))
        oxy_hemo = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        bvf_value = calculate_bvf(oxy_hemo)
        assert bvf_value == 1.0

        # Just deoxyhemoglobin CASES:
        mcg = MolecularCompositionGenerator()
        mcg.append(MoleculeLibrary.deoxyhemoglobin(1.0))
        deoxy_hemo = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        bvf_value = calculate_bvf(deoxy_hemo)
        assert bvf_value == 1.0
        mcg.append(MoleculeLibrary.oxyhemoglobin(0.0))
        deoxy_hemo = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        bvf_value = calculate_bvf(deoxy_hemo)
        assert bvf_value == 1.0

        # RANDOM CASES
        for _ in range(100):
            oxy = np.random.random()
            deoxy = np.random.random()
            fat = np.random.random()
            sum_oxy_deoxy_fat = oxy + deoxy + fat
            mcg = MolecularCompositionGenerator()
            mcg.append(MoleculeLibrary.fat(fat/sum_oxy_deoxy_fat))
            mcg.append(MoleculeLibrary.deoxyhemoglobin(deoxy/sum_oxy_deoxy_fat))
            mcg.append(MoleculeLibrary.oxyhemoglobin(oxy/sum_oxy_deoxy_fat))
            bvf_value = calculate_bvf(mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC))
            assert abs(bvf_value - (oxy+deoxy)/sum_oxy_deoxy_fat) < 1e-10

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

    def test_rounding_function(self):
        assert round_x5_away_from_zero(0.5) == 1
        assert round_x5_away_from_zero(0.4) == 0
        assert round_x5_away_from_zero(0.6) == 1
        assert round_x5_away_from_zero(0.1) == 0
        assert round_x5_away_from_zero(0.9) == 1
        assert round_x5_away_from_zero(0.0) == 0
        assert round_x5_away_from_zero(1.0) == 1
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.5) == np.arange(0, 15) + 1).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.4) == np.arange(0, 15)).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.6) == np.arange(0, 15) + 1).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.1) == np.arange(0, 15)).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.9) == np.arange(0, 15) + 1).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 0.0) == np.arange(0, 15)).all()
        assert (round_x5_away_from_zero(np.arange(0, 15) + 1.0) == np.arange(0, 15) + 1).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.5) == np.arange(-15, 0) - 1).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.4) == np.arange(-15, 0)).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.6) == np.arange(-15, 0) - 1).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.1) == np.arange(-15, 0)).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.9) == np.arange(-15, 0) - 1).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 0.0) == np.arange(-15, 0)).all()
        assert (round_x5_away_from_zero(np.arange(-15, 0) - 1.0) == np.arange(-15, 0) - 1).all()
