# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from simpa.utils import SegmentationClasses, MolecularCompositionGenerator
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.calculate import calculate_oxygenation
from simpa.utils.calculate import randomize_uniform
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa.utils.calculate import positive_gauss
from simpa.utils.calculate import bilinear_interpolation
from scipy.interpolate import interp2d
import numpy as np
import torch


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

    def test_bilinear_interpolation(self):
        # test 2 dimensional interpolation:
        xdim = np.random.randint(10,200)
        ydim = np.random.randint(10,200)
        n_samples = 1000

        image_2d = torch.rand((xdim, ydim), device="cpu")

        # want also samples outside the image to test nearest neighbors interpolation
        x_samples = torch.FloatTensor(n_samples).uniform_(-1, xdim)
        y_samples = torch.FloatTensor(n_samples).uniform_(-1, ydim)
        
        interpolated_values = bilinear_interpolation(image_2d, x_samples, y_samples)
        
        f = interp2d(np.arange(xdim), np.arange(ydim), image_2d.numpy().T, kind='linear')
        # loop over the samples since the interp2d function restructeres the output randomly for non-1-dim inputs
        for i, (x_samp, y_samp) in enumerate(zip(x_samples.numpy(), y_samples.numpy())):
            ref = f(x_samp, y_samp)
            self.assertAlmostEqual(ref, interpolated_values[i].item()+1e6)

        #TODO: Test 3dimensional interpolation
        #zdim = np.random.randint(10,200)
        #image_3d = torch.rand()
        #z_samples = torch.FloatTensor(n_samples).uniform_(-1, zdim)
        #interpolated_values_3d = bilinear_interpolation(image_3d, x_samples, y_samples, z_samples)