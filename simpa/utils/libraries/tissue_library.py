# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import numpy as np
import typing

from simpa.utils import Settings
from simpa.utils import OpticalTissueProperties, SegmentationClasses, StandardProperties, MolecularCompositionGenerator
from simpa.utils import Molecule
from simpa.utils import MOLECULE_LIBRARY
from simpa.utils import Spectrum
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.spectrum_library import AnisotropySpectrumLibrary, ScatteringSpectrumLibrary
from simpa.utils.calculate import randomize_uniform
from simpa.utils.libraries.spectrum_library import AbsorptionSpectrumLibrary


class TissueLibrary(object):
    """
    TODO
    """

    def get_blood_volume_fractions(self, total_blood_volume_fraction=1e-10, oxygenation=1e-10):
        """
        TODO
        """
        return [total_blood_volume_fraction*oxygenation, total_blood_volume_fraction*(1-oxygenation)]

    def constant(self, mua: float = 1e-10, mus: float = 1e-10, g: float = 1e-10) -> MolecularComposition:
        """
        Returns a generic issue defined by the provided constant optical parameters.

        :param mua: The absorption coefficient in cm^{-1}.
        :param mus: The scattering coefficient in cm^{-1}.
        :param g: The anisotropy.

        :returns: The molecular composition of the tissue.
        """
        mua_as_spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(mua)
        mus_as_spectrum = ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(mus)
        g_as_spectrum = AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(g)
        return self.generic_tissue(mua_as_spectrum, mus_as_spectrum, g_as_spectrum, "constant_mua_mus_g")

    def generic_tissue(self,
                       mua: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       mus: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       g: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       molecule_name: typing.Optional[str] = "generic_tissue") -> MolecularComposition:
        """
        Returns a generic issue defined by the provided optical parameters.

        :param mua: The absorption coefficient spectrum in cm^{-1}.
        :param mus: The scattering coefficient spectrum in cm^{-1}.
        :param g: The anisotropy spectrum.
        :param molecule_name: The molecule name.

        :returns: The molecular composition of the tissue.
        """
        assert isinstance(mua, Spectrum), type(mua)
        assert isinstance(mus, Spectrum), type(mus)
        assert isinstance(g, Spectrum), type(g)
        assert isinstance(molecule_name, str) or molecule_name is None, type(molecule_name)

        return (MolecularCompositionGenerator().append(Molecule(name=molecule_name,
                                                                absorption_spectrum=mua,
                                                                volume_fraction=1.0,
                                                                scattering_spectrum=mus,
                                                                anisotropy_spectrum=g))
                .get_molecular_composition(SegmentationClasses.GENERIC))

    def muscle(self, background_oxy=None, blood_volume_fraction=None):
        """

        :return: a settings dictionary containing all min and max parameters fitting for muscle tissue.
        """

        # Determine muscle oxygenation
        if background_oxy is None:
            oxy = 0.175
        else:
            oxy = background_oxy

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if blood_volume_fraction is None:
            bvf = 0.06
        else:
            bvf = blood_volume_fraction

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(bvf, oxy)

        # Get the water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        if isinstance(blood_volume_fraction, np.ndarray):
            if (blood_volume_fraction + water_volume_fraction - 1 > 1e-5).any():
                raise AssertionError(f"Blood volume fraction too large, must be less than {1 - water_volume_fraction}"
                                     f"everywhere to leave space for water")

        else:
            if bvf + water_volume_fraction - 1 > 1e-5:
                raise AssertionError(f"Blood volume fraction too large, must be less than {1 - water_volume_fraction}"
                                     f"everywhere to leave space for water")

        custom_water = MOLECULE_LIBRARY.water(water_volume_fraction)
        custom_water.anisotropy_spectrum = AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
            OpticalTissueProperties.STANDARD_ANISOTROPY - 0.005)
        custom_water.alpha_coefficient = 1.58
        custom_water.speed_of_sound = StandardProperties.SPEED_OF_SOUND_MUSCLE + 16
        custom_water.density = StandardProperties.DENSITY_MUSCLE + 41
        custom_water.mus500 = OpticalTissueProperties.MUS500_MUSCLE_TISSUE
        custom_water.b_mie = OpticalTissueProperties.BMIE_MUSCLE_TISSUE
        custom_water.f_ray = OpticalTissueProperties.FRAY_MUSCLE_TISSUE

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(value=MOLECULE_LIBRARY.muscle_scatterer(
                        volume_fraction=1 - fraction_oxy - fraction_deoxy - water_volume_fraction),
                        key="muscle_scatterers")
                .append(custom_water)
                .get_molecular_composition(SegmentationClasses.MUSCLE))

    def soft_tissue(self, background_oxy=None, blood_volume_fraction=None):
        """
        IMPORTANT! This tissue is not tested and it is not based on a specific real tissue type.
        It is a mixture of muscle (mostly optical properties) and water (mostly acoustic properties).
        This tissue type roughly resembles the generic background tissue that we see in real PA images.
        :return: a settings dictionary containing all min and max parameters fitting for generic soft tissue.
        """

        # Determine muscle oxygenation
        if background_oxy is None:
            oxy = OpticalTissueProperties.BACKGROUND_OXYGENATION
        else:
            oxy = background_oxy

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if blood_volume_fraction is None:
            bvf = OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE
        else:
            bvf = blood_volume_fraction

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(bvf, oxy)

        # Get the water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        if (blood_volume_fraction + water_volume_fraction - 1 > 1e-5).any():
            raise AssertionError(f"Blood volume fraction too large, must be less than {1 - water_volume_fraction}"
                                 f"everywhere to leave space for water")

        custom_water = MOLECULE_LIBRARY.water(water_volume_fraction)
        custom_water.anisotropy_spectrum = AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
            OpticalTissueProperties.STANDARD_ANISOTROPY - 0.005)
        custom_water.alpha_coefficient = 0.08
        custom_water.speed_of_sound = StandardProperties.SPEED_OF_SOUND_WATER
        custom_water.density = StandardProperties.DENSITY_WATER
        custom_water.mus500 = OpticalTissueProperties.MUS500_MUSCLE_TISSUE
        custom_water.b_mie = OpticalTissueProperties.BMIE_MUSCLE_TISSUE
        custom_water.f_ray = OpticalTissueProperties.FRAY_MUSCLE_TISSUE

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(value=MOLECULE_LIBRARY.muscle_scatterer(
                        volume_fraction=1 - fraction_oxy - fraction_deoxy - water_volume_fraction),
                        key="muscle_scatterers")
                .append(custom_water)
                .get_molecular_composition(SegmentationClasses.SOFT_TISSUE))

    def epidermis(self, melanosom_volume_fraction=None):
        """

        :return: a settings dictionary containing all min and max parameters fitting for epidermis tissue.
        """

        # Get melanin volume fraction
        if melanosom_volume_fraction is None:
            melanin_volume_fraction = 0.014
        else:
            melanin_volume_fraction = melanosom_volume_fraction

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.melanin(melanin_volume_fraction))
                .append(MOLECULE_LIBRARY.epidermal_scatterer(1 - melanin_volume_fraction))
                .get_molecular_composition(SegmentationClasses.EPIDERMIS))

    def dermis(self, background_oxy=None, blood_volume_fraction=None):
        """

        :return: a settings dictionary containing all min and max parameters fitting for dermis tissue.
        """

        # Determine muscle oxygenation
        if background_oxy is None:
            oxy = 0.5
        else:
            oxy = background_oxy

        if blood_volume_fraction is None:
            bvf = 0.002
        else:
            bvf = blood_volume_fraction

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(bvf, oxy)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.dermal_scatterer(1.0 - bvf))
                .get_molecular_composition(SegmentationClasses.DERMIS))

    def subcutaneous_fat(self, oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
        """

        :return: a settings dictionary containing all min and max parameters fitting for subcutaneous fat tissue.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
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

    def blood(self, oxygenation=None):
        """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """

        # Get the bloood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if oxygenation is None:
            oxygenation = randomize_uniform(0.0, 1.0)

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(1.0, oxygenation)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
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

    def lymph_node(self, oxy=None, blood_volume_fraction=None):
        """
        IMPORTANT! This tissue is not tested and it is not based on a specific real tissue type.
        It is a mixture of oxyhemoglobin, deoxyhemoglobin, and lymph node customized water.
        :return: a settings dictionary fitting for generic lymph node tissue.
        """

        # Determine muscle oxygenation
        if oxy is None:
            oxy = OpticalTissueProperties.LYMPH_NODE_OXYGENATION

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if blood_volume_fraction is None:
            blood_volume_fraction = OpticalTissueProperties.BLOOD_VOLUME_FRACTION_LYMPH_NODE

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(blood_volume_fraction, oxy)

        # Get the water volume fraction
        # water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        lymphatic_fluid = MOLECULE_LIBRARY.water(1 - fraction_deoxy - fraction_oxy)
        lymphatic_fluid.speed_of_sound = StandardProperties.SPEED_OF_SOUND_LYMPH_NODE + 1.22
        lymphatic_fluid.density = StandardProperties.DENSITY_LYMPH_NODE - 2.30
        lymphatic_fluid.alpha_coefficient = StandardProperties.ALPHA_COEFF_LYMPH_NODE + 0.36

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(lymphatic_fluid)
                .get_molecular_composition(SegmentationClasses.LYMPH_NODE))


TISSUE_LIBRARY = TissueLibrary()
