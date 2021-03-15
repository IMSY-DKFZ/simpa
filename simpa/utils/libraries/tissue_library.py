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

from simpa.utils import OpticalTissueProperties, SegmentationClasses
from simpa.utils import SPECTRAL_LIBRARY
from simpa.utils import Molecule
from simpa.utils import MOLECULE_LIBRARY
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.calculate import randomize_uniform


class MolecularCompositionGenerator(object):
    """
    The MolecularCompositionGenerator is a helper class to facilitate the creation of a
    MolecularComposition instance.
    """
    def __init__(self):
        self.molecular_composition_dictionary = dict()

    def append(self, value: Molecule = None, key: str = None):
        if key is None:
            key = value.name
        if key in self.molecular_composition_dictionary:
            raise KeyError(key + " already in the molecular composition!")
        self.molecular_composition_dictionary[key] = value
        return self

    def get_molecular_composition(self, segmentation_type):
        return MolecularComposition(segmentation_type=segmentation_type,
                                    molecular_composition_settings=self.molecular_composition_dictionary)


class TissueLibrary(object):
    """
    TODO
    """

    def get_blood_volume_fractions(self, total_blood_volume_fraction, oxygenation):
        """
        TODO
        """
        return [total_blood_volume_fraction*oxygenation, total_blood_volume_fraction*(1-oxygenation)]

    def constant(self, mua, mus, g):
        """
        TODO
        """
        return (MolecularCompositionGenerator().append(Molecule(name="constant_absorber",
                                                                spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(mua),
                                                                volume_fraction=1.0,
                                                                mus500=mus, b_mie=0.0, f_ray=0.0,
                                                                anisotropy=g))
                                               .get_molecular_composition(SegmentationClasses.GENERIC))

    def muscle(self, background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
        """

        :return: a settings dictionary containing all min and max parameters fitting for generic background tissue.
        """

        # Determine muscle oxygenation
        oxy = randomize_uniform(background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                                background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(
            OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE, oxy)

        # Get the water volume fraction
        water_volume_fraction = randomize_uniform(0.64, 0.72)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(value=MOLECULE_LIBRARY.soft_tissue_scatterer(
                        volume_fraction=1-fraction_oxy - fraction_deoxy - water_volume_fraction),
                        key="background_scatterers")
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.MUSCLE))

    def epidermis(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for epidermis tissue.
        """
        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get melanin volume fraction
        melanin_volume_fraction = randomize_uniform(OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN -
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN +
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.melanin(melanin_volume_fraction))
                .append(MOLECULE_LIBRARY.epidermal_scatterer(1 - melanin_volume_fraction - water_volume_fraction))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.EPIDERMIS))

    def dermis(self, background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
        """

        :return: a settings dictionary containing all min and max parameters fitting for dermis tissue.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Determine muscle oxygenation
        oxy = randomize_uniform(background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                                background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(
            OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE, oxy)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.dermal_scatterer(
                                                       1-OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.DERMIS))

    def subcutaneous_fat(self, background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
        """

        :return: a settings dictionary containing all min and max parameters fitting for subcutaneous fat tissue.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Determine muscle oxygenation
        oxy = randomize_uniform(background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                                background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(
            OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE, oxy)

        # Determine fat volume fraction
        fat_volume_fraction = randomize_uniform(0.2, 1 - (water_volume_fraction + fraction_oxy + fraction_deoxy))

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.fat(fat_volume_fraction))
                .append(MOLECULE_LIBRARY.soft_tissue_scatterer(
                        1 - (fat_volume_fraction + water_volume_fraction + fraction_oxy + fraction_deoxy)))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.FAT))

    def blood_generic(self, oxygenation=None):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.BLOOD_PLASMA_FRACTION

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if oxygenation is None:
            oxygenation = randomize_uniform(0.0, 1.0)

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1-water_volume_fraction, oxygenation)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.BLOOD))

    def blood_arterial(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.BLOOD_PLASMA_FRACTION

        oxygenation = randomize_uniform(0.8, 1.0)

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1-water_volume_fraction, oxygenation)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.BLOOD))

    def blood_venous(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.BLOOD_PLASMA_FRACTION

        oxygenation = randomize_uniform(0.0, 0.8)

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1.0 - water_volume_fraction, oxygenation)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.BLOOD))

    def bone(self):
        """

            :return: a settings dictionary containing all min and max parameters fitting for full blood.
            """
        # Get water volume fraction
        water_volume_fraction = randomize_uniform(OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN -
                                                  OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD,
                                                  OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN +
                                                  OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD
                                                  )

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.bone(1 - water_volume_fraction))
                .append(MOLECULE_LIBRARY.water(water_volume_fraction))
                .get_molecular_composition(SegmentationClasses.BONE))

    def mediprene(self):
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.mediprene())
                .get_molecular_composition(SegmentationClasses.MEDIPRENE))

    def heavy_water(self):
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.heavy_water())
                .get_molecular_composition(SegmentationClasses.HEAVY_WATER))

    def ultrasound_gel(self):
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.water())
                .get_molecular_composition(SegmentationClasses.ULTRASOUND_GEL))


TISSUE_LIBRARY = TissueLibrary()
