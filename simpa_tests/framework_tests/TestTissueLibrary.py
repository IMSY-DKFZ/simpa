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
from simpa.utils import SegmentationClasses
from simpa.utils.libraries.tissue_library import MolecularComposition, MolecularCompositionGenerator
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils import Tags
from simpa.utils import TISSUE_LIBRARY


def assert_defined_in_common_wavelength_range(molecular_composition: MolecularComposition):
    molecular_composition.update_internal_properties()
    for wavelength in range(600, 1000):
        tissue_properties = molecular_composition.get_properties_for_wavelength(wavelength)
        tissue_properties_2 = molecular_composition.get_properties_for_wavelength(wavelength)
        for property_key in tissue_properties.property_tags:
            assert tissue_properties[property_key] == tissue_properties_2[property_key]
            if property_key == Tags.PROPERTY_SEGMENTATION:
                # Segmentations are not defined to be > 0
                continue
            if property_key == Tags.PROPERTY_OXYGENATION and tissue_properties[property_key] is None:
                # Oxygenations may be None, if no hemoglobin is present in the molecular structure
                continue

            assert tissue_properties[property_key] >= 0, (property_key + ":" +
                                                         str(tissue_properties[property_key]) +
                                                         " was not > 0")


class TestTissueLibrary(unittest.TestCase):

    def setUp(self):
        print("\n[SetUp]")

    def tearDown(self):
        print("\n[TearDown]")

    def test_use_tissue_library_to_get_molecular_compositions(self):
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.muscle())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.constant(mua=1.0, mus=1.0, g=0.9))
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.blood_arterial())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.blood_generic())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.blood_venous())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.bone())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.dermis())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.epidermis())
        assert_defined_in_common_wavelength_range(TISSUE_LIBRARY.subcutaneous_fat())

    def test_molecular_composition_generator(self):
        mcg = MolecularCompositionGenerator()
        mcg.append("fat", MOLECULE_LIBRARY.fat(1.0))
        molecular_composition_1 = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        mcg.append("oxy", MOLECULE_LIBRARY.oxyhemoglobin(1.0))
        molecular_composition_2 = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        mcg.append("deoxy", MOLECULE_LIBRARY.deoxyhemoglobin(0.0))
        molecular_composition_3 = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        mcg = MolecularCompositionGenerator()
        molecular_composition_4 = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)

        assert len(molecular_composition_1) == 1
        assert len(molecular_composition_2) == 2
        assert len(molecular_composition_3) == 3
        assert len(molecular_composition_4) == 0

    def test_molecular_composition_generator_add_same_molecule_twice(self):
        mcg = MolecularCompositionGenerator()
        mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(0.5))
        try:
            mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(0.5))
        except KeyError as e:
            assert e is not None
        molecular_composition = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        assert len(molecular_composition) == 1
        mcg.append(MOLECULE_LIBRARY.oxyhemoglobin(0.5), "different molecule name")
        molecular_composition = mcg.get_molecular_composition(segmentation_type=SegmentationClasses.GENERIC)
        assert len(molecular_composition) == 2
