# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

from ippai.utils import OpticalTissueProperties
from ippai.utils import SPECTRAL_LIBRARY
from ippai.utils import Chromophore
from ippai.utils import CHROMOPHORE_LIBRARY
from ippai.utils.calculate import randomize_uniform


class TissueSettingsGenerator(object):
    def __init__(self):
        self.tissue_dictionary = dict()

    def append(self, key: str, value: Chromophore):
        self.tissue_dictionary[key] = value
        return self

    def get_settings(self):
        return self.tissue_dictionary


class TissueLibrary(object):

    def __init__(self):
        print("INIT")

    def get_blood_volume_fractions(self, total_blood_volume_fraction, oxygenation):
        return [total_blood_volume_fraction*oxygenation, total_blood_volume_fraction*(1-oxygenation)]

    def constant(self, mua, mus, g):
        return (TissueSettingsGenerator().append(key="constant_chromophore",
                                                 value=Chromophore(SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ONE,
                                                                   volume_fraction=mua,
                                                                   musp500=mus, b_mie=0.0, f_ray=0.0,
                                                                   anisotropy=g)).get_settings())

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
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="background_scatterers", value=CHROMOPHORE_LIBRARY.soft_tissue_scatterer())
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

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
        return (TissueSettingsGenerator()
                .append(key="melanin", value=CHROMOPHORE_LIBRARY.melanin(melanin_volume_fraction))
                .append(key="epidermal", value=CHROMOPHORE_LIBRARY.epidermal_scatterer(1-melanin_volume_fraction))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

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
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="dermal scatterers", value=CHROMOPHORE_LIBRARY.dermal_scatterer(
                                                       1-OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

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
        fat_volume_fraction = randomize_uniform(0.3, 0.4)

        # generate the tissue dictionary
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="fat", value=CHROMOPHORE_LIBRARY.fat(fat_volume_fraction))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

    def blood_generic(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1.0, randomize_uniform(0.0, 1.0))

        # generate the tissue dictionary
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

    def blood_arterial(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1.0, randomize_uniform(0.8, 1.0))

        # generate the tissue dictionary
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

    def blood_venous(self):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1.0, randomize_uniform(0.0, 0.8))

        # generate the tissue dictionary
        return (TissueSettingsGenerator()
                .append(key="Oxyhemoglobin", value=CHROMOPHORE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(key="Deoxyhemoglobin", value=CHROMOPHORE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())

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

        # Get melanin volume fraction
        melanin_volume_fraction = randomize_uniform(OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN -
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN +
                                                    OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD)

        # generate the tissue dictionary
        return (TissueSettingsGenerator()
                .append(key="bone", value=CHROMOPHORE_LIBRARY.bone_scatterer())
                .append(key="water", value=CHROMOPHORE_LIBRARY.water(water_volume_fraction)).get_settings())


TISSUE_LIBRARY = TissueLibrary()
