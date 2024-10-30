# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import typing

from simpa.utils import OpticalTissueProperties, SegmentationClasses, StandardProperties, MolecularCompositionGenerator
from simpa.utils import Molecule
from simpa.utils import MOLECULE_LIBRARY
from simpa.utils.libraries.spectrum_library import (AnisotropySpectrumLibrary, ScatteringSpectrumLibrary,
                                                    RefractiveIndexSpectrumLibrary)
from simpa.utils import Spectrum
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.spectrum_library import AnisotropySpectrumLibrary, ScatteringSpectrumLibrary
from simpa.utils.calculate import randomize_uniform
from simpa.utils.libraries.spectrum_library import AbsorptionSpectrumLibrary

from typing import Union, List
import torch


class TissueLibrary(object):
    """
    A library, returning molecular compositions for various typical tissue segmentations.
    """

    def get_blood_volume_fractions(self, oxygenation: Union[float, int, torch.Tensor] = 1e-10,
                                   blood_volume_fraction: Union[float, int, torch.Tensor] = 1e-10)\
            -> List[Union[int, float, torch.Tensor]]:
        """
        A function that returns the volume fraction of the oxygenated and deoxygenated blood.
        :param oxygenation: The oxygenation level of the blood volume fraction (as a decimal).
        Default: 1e-10
        :param blood_volume_fraction: The total blood volume fraction (including oxygenated and deoxygenated blood).
        Default: 1e-10
        :return: the blood volume fraction of the oxygenated and deoxygenated blood separately.
        """
        return [blood_volume_fraction*oxygenation, blood_volume_fraction*(1-oxygenation)]

    def constant(self, mua: Union[float, int, torch.Tensor] = 1e-10, mus: Union[float, int, torch.Tensor] = 1e-10,
                 g: Union[float, int, torch.Tensor] = 0, n: float = 1.3) -> MolecularComposition:
        """
        A function returning a molecular composition as specified by the user. Typically intended for the use of wanting
        specific mua, mus and g values.
        :param mua: optical absorption coefficient
        Default: 1e-10 cm^⁻1
        :param mus: optical scattering coefficient
        Default: 1e-10 cm^⁻1
        :param g: optical scattering anisotropy
        Default: 0
        :return: the molecular composition as specified by the user
        """
        mua_as_spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(mua)
        mus_as_spectrum = ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(mus)
        g_as_spectrum = AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(g)
        n_as_spectrum = RefractiveIndexSpectrumLibrary.CONSTANT_REFRACTOR_ARBITRARY(n)
        return self.generic_tissue(mua_as_spectrum, mus_as_spectrum, g_as_spectrum, n_as_spectrum, "constant_mua_mus_g_n")

    def generic_tissue(self,
                       mua: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       mus: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       g: Spectrum = AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-10),
                       n: Spectrum = RefractiveIndexSpectrumLibrary.CONSTANT_REFRACTOR_ARBITRARY(1.3),
                       molecule_name: typing.Optional[str] = "generic_tissue") -> MolecularComposition:
        """
        Returns a generic issue defined by the provided optical parameters.

        :param mua: The absorption coefficient spectrum in cm^{-1}.
        :param mus: The scattering coefficient spectrum in cm^{-1}.
        :param g: The anisotropy spectrum.
        :param n: The refractive index spectrum.
        :param molecule_name: The molecule name.

        :returns: The molecular composition of the tissue.
        """
        assert isinstance(mua, Spectrum), type(mua)
        assert isinstance(mus, Spectrum), type(mus)
        assert isinstance(g, Spectrum), type(g)
        assert isinstance(n, Spectrum), type(n)
        assert isinstance(molecule_name, str) or molecule_name is None, type(molecule_name)

        return (MolecularCompositionGenerator().append(Molecule(name=molecule_name,
                                                                absorption_spectrum=mua,
                                                                volume_fraction=1.0,
                                                                scattering_spectrum=mus,
                                                                anisotropy_spectrum=g,
                                                                refractive_index=n))
                .get_molecular_composition(SegmentationClasses.GENERIC))

    def muscle(self, oxygenation: Union[float, int, torch.Tensor] = 0.175,
               blood_volume_fraction: Union[float, int, torch.Tensor] = 0.06) -> MolecularComposition:
        """

        :return: a settings dictionary containing all min and max parameters fitting for muscle tissue.
        """

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(oxygenation, blood_volume_fraction)

        # Get the water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

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

    def soft_tissue(self, oxygenation: Union[float, int, torch.Tensor] = OpticalTissueProperties.BACKGROUND_OXYGENATION,
                    blood_volume_fraction: Union[float, int, torch.Tensor] = OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE) -> MolecularComposition:
        """
        IMPORTANT! This tissue is not tested and it is not based on a specific real tissue type.
        It is a mixture of muscle (mostly optical properties) and water (mostly acoustic properties).
        This tissue type roughly resembles the generic background tissue that we see in real PA images.
        :param oxygenation: The oxygenation level of the blood volume fraction (as a decimal).
        Default: OpticalTissueProperties.BACKGROUND_OXYGENATION
        :param blood_volume_fraction: The total blood volume fraction (including oxygenated and deoxygenated blood).
        Default: OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE
        :return: a settings dictionary containing all min and max parameters fitting for generic soft tissue.
        """

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(oxygenation, blood_volume_fraction)

        # Get the water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

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

    def epidermis(self, melanosom_volume_fraction: Union[float, int, torch.Tensor] = 0.014) -> MolecularComposition:
        """

        :return: a settings dictionary containing all min and max parameters fitting for epidermis tissue.
        """

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.melanin(melanosom_volume_fraction))
                .append(MOLECULE_LIBRARY.epidermal_scatterer(1 - melanosom_volume_fraction))
                .get_molecular_composition(SegmentationClasses.EPIDERMIS))

    def dermis(self, oxygenation: Union[float, int, torch.Tensor] = 0.5,
               blood_volume_fraction: Union[float, int, torch.Tensor] = 0.002) -> MolecularComposition:
        """
        Create a molecular composition mimicking that of dermis
        :param oxygenation: The oxygenation level of the blood volume fraction (as a decimal).
        Default: 0.5
        :param blood_volume_fraction: The total blood volume fraction (including oxygenated and deoxygenated blood).
        Default: 0.002
        :return: a settings dictionary containing all min and max parameters fitting for dermis tissue.
        """

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(oxygenation, blood_volume_fraction)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .append(MOLECULE_LIBRARY.dermal_scatterer(1.0 - blood_volume_fraction))
                .get_molecular_composition(SegmentationClasses.DERMIS))

    def subcutaneous_fat(self,
                         oxygenation: Union[float, int, torch.Tensor] = OpticalTissueProperties.BACKGROUND_OXYGENATION,
                         blood_volume_fraction: Union[float, int, torch.Tensor]
                         = OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE) -> MolecularComposition:
        """
        Create a molecular composition mimicking that of subcutaneous fat
        :param oxygenation: The oxygenation level of the blood volume fraction (as a decimal).
        Default: OpticalTissueProperties.BACKGROUND_OXYGENATION
        :param blood_volume_fraction: The total blood volume fraction (including oxygenated and deoxygenated blood).
        Default: OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE
        :return: a settings dictionary containing all min and max parameters fitting for subcutaneous fat tissue.
        """

        # Get water volume fraction
        water_volume_fraction = OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(
            oxygenation, blood_volume_fraction)

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

    def blood(self, oxygenation: Union[float, int, torch.Tensor, None] = None) -> MolecularComposition:
        """
        Create a molecular composition mimicking that of blood
        :param oxygenation: The oxygenation level of the blood(as a decimal).
        Default: random oxygenation between 0 and 1.
        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if oxygenation is None:
            oxygenation = randomize_uniform(0.0, 1.0)

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(oxygenation, 1.0)

        # generate the tissue dictionary
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.oxyhemoglobin(fraction_oxy))
                .append(MOLECULE_LIBRARY.deoxyhemoglobin(fraction_deoxy))
                .get_molecular_composition(SegmentationClasses.BLOOD))

    def bone(self) -> MolecularComposition:
        """
        Create a molecular composition mimicking that of bone
        :return: a settings dictionary fitting for bone.
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

    def mediprene(self) -> MolecularComposition:
        """
        Create a molecular composition mimicking that of mediprene
        :return: a settings dictionary fitting for mediprene.
        """
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.mediprene())
                .get_molecular_composition(SegmentationClasses.MEDIPRENE))

    def heavy_water(self) -> MolecularComposition:
        """
            Create a molecular composition mimicking that of heavy water
            :return: a settings dictionary containing all min and max parameters fitting for heavy water.
        """
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.heavy_water())
                .get_molecular_composition(SegmentationClasses.HEAVY_WATER))

    def ultrasound_gel(self) -> MolecularComposition:
        """
            Create a molecular composition mimicking that of ultrasound gel
            :return: a settings dictionary fitting for generic ultrasound gel.
        """
        return (MolecularCompositionGenerator()
                .append(MOLECULE_LIBRARY.water())
                .get_molecular_composition(SegmentationClasses.ULTRASOUND_GEL))

    def lymph_node(self, oxygenation: Union[float, int, torch.Tensor, None] = None,
                   blood_volume_fraction: Union[float, int, torch.Tensor, None] = None) -> MolecularComposition:
        """
        IMPORTANT! This tissue is not tested and it is not based on a specific real tissue type.
        It is a mixture of oxyhemoglobin, deoxyhemoglobin, and lymph node customized water.
        :param oxygenation: The oxygenation level of the blood volume fraction (as a decimal).
        Default: 0.175
        :param blood_volume_fraction: The total blood volume fraction (including oxygenated and deoxygenated blood).
        Default: 0.06
        :return: a settings dictionary fitting for generic lymph node tissue.
        """

        # Determine muscle oxygenation
        if oxygenation is None:
            oxygenation = OpticalTissueProperties.LYMPH_NODE_OXYGENATION

        # Get the blood volume fractions for oxyhemoglobin and deoxyhemoglobin
        if blood_volume_fraction is None:
            blood_volume_fraction = OpticalTissueProperties.BLOOD_VOLUME_FRACTION_LYMPH_NODE

        [fraction_oxy, fraction_deoxy] = self.get_blood_volume_fractions(oxygenation, blood_volume_fraction)

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
