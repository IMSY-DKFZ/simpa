# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from simpa.utils import Tags

from simpa.utils.tissue_properties import TissueProperties
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils.libraries.spectrum_library import AnisotropySpectrumLibrary, ScatteringSpectrumLibrary
from simpa.utils import Spectrum
from simpa.utils.calculate import calculate_oxygenation, calculate_gruneisen_parameter_from_temperature
from simpa.utils.serializer import SerializableSIMPAClass
from simpa.utils.libraries.spectrum_library import AbsorptionSpectrumLibrary
from simpa.utils.processing_device import get_processing_device

from typing import Optional, Union


class MolecularComposition(SerializableSIMPAClass, list):
    """
    A class representing a molecular composition which is a list of Molecules.

    Attributes:
        segmentation_type (str): The type of segmentation.
        internal_properties (TissueProperties): The internal tissue properties.
    """

    def __init__(self, segmentation_type: Optional[str] = None, molecular_composition_settings: Optional[dict] = None):
        """
        Initialize the MolecularComposition object.

        :param segmentation_type: The segmentation class associated with this molecular composition.
        :type segmentation_type: str, optional
        :param molecular_composition_settings: A settings dictionary or dict containing the molecules that constitute
        this composition
        :type molecular_composition_settings: dict, optional
        """
        super().__init__()
        self.segmentation_type = segmentation_type
        self.internal_properties: TissueProperties = None

        if molecular_composition_settings is None:
            return

        _keys = molecular_composition_settings.keys()
        for molecule_name in _keys:
            self.append(molecular_composition_settings[molecule_name])

    def update_internal_properties(self, settings):
        """
        Re-defines the internal properties of the molecular composition.
        For each data field and molecule, a linear mixing model is used to arrive at the final parameters.

        Raises:
            AssertionError: If the total volume fraction of all molecules is not exactly 100%.
        """
        self.internal_properties = TissueProperties(settings)
        self.internal_properties[Tags.DATA_FIELD_SEGMENTATION] = self.segmentation_type
        self.internal_properties[Tags.DATA_FIELD_OXYGENATION] = calculate_oxygenation(self)
        search_list = self.copy()

        for molecule in search_list:
            self.internal_properties.volume_fraction += molecule.get_volume_fraction(settings)
            self.internal_properties[Tags.DATA_FIELD_GRUNEISEN_PARAMETER] += \
                molecule.volume_fraction * molecule.gruneisen_parameter
            self.internal_properties[Tags.DATA_FIELD_DENSITY] += molecule.volume_fraction * molecule.density
            self.internal_properties[Tags.DATA_FIELD_SPEED_OF_SOUND] += molecule.volume_fraction * \
                molecule.speed_of_sound
            self.internal_properties[Tags.DATA_FIELD_ALPHA_COEFF] += molecule.volume_fraction * \
                molecule.alpha_coefficient

        if (torch.abs(self.internal_properties.volume_fraction - 1.0) > 1e-5).any():
            raise AssertionError("Invalid Molecular composition! The volume fractions of all molecules must be"
                                 "exactly 100%!")

    def get_properties_for_wavelength(self, settings, wavelength) -> TissueProperties:
        """
        Get the tissue properties for a specific wavelength.

        :param wavelength: The wavelength to get properties for.
        :return: The updated tissue properties.
        """
        self.update_internal_properties(settings)
        self.internal_properties[Tags.DATA_FIELD_ABSORPTION_PER_CM] = 0
        self.internal_properties[Tags.DATA_FIELD_SCATTERING_PER_CM] = 0
        self.internal_properties[Tags.DATA_FIELD_ANISOTROPY] = 0
        search_list = self.copy()
        for molecule in search_list:
            self.internal_properties[Tags.DATA_FIELD_ABSORPTION_PER_CM] += \
                (molecule.volume_fraction * molecule.spectrum.get_value_for_wavelength(wavelength))

            self.internal_properties[Tags.DATA_FIELD_SCATTERING_PER_CM] += \
                (molecule.volume_fraction * (molecule.scattering_spectrum.get_value_for_wavelength(wavelength)))

            self.internal_properties[Tags.DATA_FIELD_ANISOTROPY] += \
                molecule.volume_fraction * molecule.anisotropy_spectrum.get_value_for_wavelength(wavelength)

        return self.internal_properties

    def serialize(self) -> dict:
        """
        Serialize the molecular composition to a dictionary.

        :return: The serialized molecular composition.
        """
        dict_items = self.__dict__
        dict_items["internal_properties"] = None
        list_items = [molecule for molecule in self]
        return {"MolecularComposition": {"dict_items": dict_items, "list_items": list_items}}

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict):
        """
        Deserialize a dictionary into a MolecularComposition object.

        :param dictionary_to_deserialize: The dictionary to deserialize.
        :return: The deserialized MolecularComposition object.
        """
        deserialized_molecular_composition = MolecularCompositionGenerator()
        for molecule in dictionary_to_deserialize["list_items"]:
            deserialized_molecular_composition.append(molecule)
        deserialized_molecular_composition = deserialized_molecular_composition.get_molecular_composition(
            dictionary_to_deserialize["dict_items"]["segmentation_type"]
        )
        return deserialized_molecular_composition


class Molecule(SerializableSIMPAClass, object):
    """
    A class representing a molecule with various properties.

    Attributes:
        name (str): The name of the molecule.
        spectrum (Spectrum): The absorption spectrum of the molecule.
        volume_fraction (float): The volume fraction of the molecule.
        scattering_spectrum (Spectrum): The scattering spectrum of the molecule.
        anisotropy_spectrum (Spectrum): The anisotropy spectrum of the molecule.
        gruneisen_parameter (float): The Grüneisen parameter of the molecule.
        density (float): The density of the molecule.
        speed_of_sound (float): The speed of sound in the molecule.
        alpha_coefficient (float): The alpha coefficient of the molecule.
    """

    def __init__(self, name: str = None,
                 absorption_spectrum: Spectrum = None,
                 volume_fraction: float = None,
                 scattering_spectrum: Spectrum = None,
                 anisotropy_spectrum: Spectrum = None, gruneisen_parameter: float = None,
                 density: float = None, speed_of_sound: float = None,
                 alpha_coefficient: float = None):
        """
        Initialize the Molecule object.

        :param name: The name of the molecule.
        :param absorption_spectrum: The absorption spectrum of the molecule.
        :param volume_fraction: The volume fraction of the molecule.
        :param scattering_spectrum: The scattering spectrum of the molecule.
        :param anisotropy_spectrum: The anisotropy spectrum of the molecule.
        :param gruneisen_parameter: The Grüneisen parameter of the molecule.
        :param density: The density of the molecule.
        :param speed_of_sound: The speed of sound in the molecule.
        :param alpha_coefficient: The alpha coefficient of the molecule.
        """
        if name is None:
            name = "GenericMoleculeName"
        if not isinstance(name, str):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            else:
                raise TypeError("Molecule name must be of type str or bytes instead of {}!".format(type(name)))
        self.name = name

        if absorption_spectrum is None:
            absorption_spectrum = AbsorptionSpectrumLibrary.CONSTANT_ABSORBER_ZERO
        if isinstance(absorption_spectrum, dict):
            absorption_spectrum = absorption_spectrum[list(absorption_spectrum.keys())[0]]
        if not isinstance(absorption_spectrum, Spectrum):
            raise TypeError(f"The given spectrum was not of type AbsorptionSpectrum! Instead: "
                            f"{type(absorption_spectrum)} and reads: {absorption_spectrum}")
        self.spectrum = absorption_spectrum

        if volume_fraction is None:
            volume_fraction = 0.0
        if not isinstance(volume_fraction, (int, float, np.int64, np.ndarray)):
            raise TypeError(f"The given volume_fraction was not of type float or array instead of "
                            f"{type(volume_fraction)}!")
        if isinstance(volume_fraction, np.ndarray):
            volume_fraction = torch.from_numpy(volume_fraction)
        self.volume_fraction = volume_fraction

        if scattering_spectrum is None:
            scattering_spectrum = ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(1e-15)
        if not isinstance(scattering_spectrum, Spectrum):
            raise TypeError(f"The given scattering_spectrum was not of type Spectrum instead of "
                            f"{type(scattering_spectrum)}!")
        self.scattering_spectrum = scattering_spectrum

        if anisotropy_spectrum is None:
            anisotropy_spectrum = 0.0
        if not isinstance(anisotropy_spectrum, Spectrum):
            raise TypeError(f"The given anisotropy was not of type Spectrum instead of {type(anisotropy_spectrum)}!")
        self.anisotropy_spectrum = anisotropy_spectrum

        if gruneisen_parameter is None:
            gruneisen_parameter = calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)
        if not isinstance(gruneisen_parameter, (int, float)):
            raise TypeError(f"The given gruneisen_parameter was not of type int or float instead "
                            f"of {type(gruneisen_parameter)}!")
        self.gruneisen_parameter = gruneisen_parameter

        if density is None:
            density = StandardProperties.DENSITY_GENERIC
        if not isinstance(density, (np.int32, np.int64, int, float)):
            raise TypeError(f"The given density was not of type int or float instead of {type(density)}!")
        self.density = density

        if speed_of_sound is None:
            speed_of_sound = StandardProperties.SPEED_OF_SOUND_GENERIC
        if not isinstance(speed_of_sound, (np.int32, np.int64, int, float)):
            raise TypeError("The given speed_of_sound was not of type int or float instead of {}!"
                            .format(type(speed_of_sound)))
        self.speed_of_sound = speed_of_sound

        if alpha_coefficient is None:
            alpha_coefficient = StandardProperties.ALPHA_COEFF_GENERIC
        if not isinstance(alpha_coefficient, (int, float)):
            raise TypeError("The given alpha_coefficient was not of type int or float instead of {}!"
                            .format(type(alpha_coefficient)))
        self.alpha_coefficient = alpha_coefficient

    def __eq__(self, other) -> bool:
        """
        Check equality between two Molecule objects.

        :param other: The other Molecule object to compare.
        :return: True if the Molecule objects are equal, False otherwise.
        """
        if isinstance(other, Molecule):
            return (self.name == other.name and
                    self.spectrum == other.spectrum and
                    self.volume_fraction == other.volume_fraction and
                    self.scattering_spectrum == other.scattering_spectrum and
                    self.alpha_coefficient == other.alpha_coefficient and
                    self.speed_of_sound == other.speed_of_sound and
                    self.gruneisen_parameter == other.gruneisen_parameter and
                    self.anisotropy_spectrum == other.anisotropy_spectrum and
                    self.density == other.density
                    )
        else:
            return super().__eq__(other)

    def get_volume_fraction(self):
        return self.volume_fraction

    def serialize(self):
        """
        Serialize the molecule to a dictionary.

        :return: The serialized molecule.
        :rtype: dict
        """
        serialized_molecule = self.__dict__
        return {"Molecule": serialized_molecule}

    @staticmethod
    def deserialize(dictionary_to_deserialize: dict):
        """
        Deserialize a dictionary into a Molecule object.

        :param dictionary_to_deserialize: The dictionary to deserialize.
        :return: The deserialized Molecule object.
        """
        deserialized_molecule = Molecule(name=dictionary_to_deserialize["name"],
                                         absorption_spectrum=dictionary_to_deserialize["spectrum"],
                                         volume_fraction=dictionary_to_deserialize["volume_fraction"],
                                         scattering_spectrum=dictionary_to_deserialize["scattering_spectrum"],
                                         alpha_coefficient=dictionary_to_deserialize["alpha_coefficient"],
                                         speed_of_sound=dictionary_to_deserialize["speed_of_sound"],
                                         gruneisen_parameter=dictionary_to_deserialize["gruneisen_parameter"],
                                         anisotropy_spectrum=dictionary_to_deserialize["anisotropy_spectrum"],
                                         density=dictionary_to_deserialize["density"])
        return deserialized_molecule


class MoleculeLibrary(object):
    """
    A class to create predefined molecules with specific properties.

    Each method in this class returns a Molecule object with predefined settings
    representing different types of tissues or substances.
    """
    # Main absorbers
    @staticmethod
    def water(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a water molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing water
        """
        return Molecule(name="water",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Water"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                            StandardProperties.WATER_MUS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            StandardProperties.WATER_G),
                        density=StandardProperties.DENSITY_WATER,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_WATER,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_WATER
                        )

    @staticmethod
    def oxyhemoglobin(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create an oxyhemoglobin molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing oxyhemoglobin
        """
        return Molecule(name="oxyhemoglobin",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Oxyhemoglobin"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("blood_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.BLOOD_ANISOTROPY),
                        density=StandardProperties.DENSITY_BLOOD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BLOOD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BLOOD
                        )

    @staticmethod
    def deoxyhemoglobin(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a deoxyhemoglobin molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing deoxyhemoglobin
        """
        return Molecule(name="deoxyhemoglobin",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Deoxyhemoglobin"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("blood_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.BLOOD_ANISOTROPY),
                        density=StandardProperties.DENSITY_BLOOD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BLOOD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BLOOD
                        )

    @staticmethod
    def melanin(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a melanin molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing melanin
        """
        return Molecule(name="melanin",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Melanin"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.scattering_from_rayleigh_and_mie_theory(
                            "epidermis", OpticalTissueProperties.MUS500_EPIDERMIS, OpticalTissueProperties.FRAY_EPIDERMIS,
                            OpticalTissueProperties.BMIE_EPIDERMIS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary().get_spectrum_by_name("Epidermis_Anisotropy"),
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def fat(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a fat molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing fat
        """
        return Molecule(name="fat",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Fat"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("fat_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.STANDARD_ANISOTROPY),
                        density=StandardProperties.DENSITY_FAT,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_FAT,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_FAT
                        )

    # Scatterers
    @staticmethod
    def constant_scatterer(scattering_coefficient: float = 100.0, anisotropy: float = 0.9,
                           volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a constant scatterer molecule with predefined properties.

        :param scattering_coefficient: The scattering coefficient, defaults to 100.0
        :param anisotropy: The anisotropy factor, defaults to 0.9
        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing a constant scatterer
        """
        return Molecule(name="constant_scatterer",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-20),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                            scattering_coefficient),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(anisotropy),
                        density=StandardProperties.DENSITY_GENERIC,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GENERIC,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GENERIC
                        )

    @staticmethod
    def soft_tissue_scatterer(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a soft tissue scatterer molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing a soft tissue scatterer
        """
        return Molecule(name="soft_tissue_scatterer",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-20),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("background_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.STANDARD_ANISOTROPY),
                        density=StandardProperties.DENSITY_GENERIC,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GENERIC,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GENERIC
                        )

    @staticmethod
    def muscle_scatterer(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a muscle scatterer molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing a muscle scatterer
        """
        return Molecule(name="muscle_scatterer",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-20),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("muscle_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.STANDARD_ANISOTROPY),
                        density=StandardProperties.DENSITY_GENERIC,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GENERIC,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GENERIC
                        )

    @staticmethod
    def epidermal_scatterer(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create an epidermal scatterer molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing an epidermal scatterer
        """
        return Molecule(name="epidermal_scatterer",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(1e-20),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.scattering_from_rayleigh_and_mie_theory(
                            "epidermis", OpticalTissueProperties.MUS500_EPIDERMIS, OpticalTissueProperties.FRAY_EPIDERMIS,
                            OpticalTissueProperties.BMIE_EPIDERMIS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary().get_spectrum_by_name("Epidermis_Anisotropy"),
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def dermal_scatterer(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a dermal scatterer molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing a dermal scatterer
        """
        return Molecule(name="dermal_scatterer",
                        absorption_spectrum=AbsorptionSpectrumLibrary().get_spectrum_by_name("Skin_Baseline"),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.scattering_from_rayleigh_and_mie_theory(
                            "dermis", OpticalTissueProperties.MUS500_DERMIS, OpticalTissueProperties.FRAY_DERMIS,
                            OpticalTissueProperties.BMIE_DERMIS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.DERMIS_ANISOTROPY),
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def bone(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a bone molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing bone
        """
        return Molecule(name="bone",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(
                            OpticalTissueProperties.BONE_ABSORPTION),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary().get_spectrum_by_name("bone_scattering"),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            OpticalTissueProperties.STANDARD_ANISOTROPY),
                        density=StandardProperties.DENSITY_BONE,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BONE,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BONE
                        )

    @staticmethod
    def mediprene(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a mediprene molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing mediprene
        """
        return Molecule(name="mediprene",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(-np.log(0.85) / 10),  # FIXME
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY((-np.log(0.85)) -
                                                                                                    (-np.log(0.85) / 10)),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(0.9),
                        density=StandardProperties.DENSITY_GEL_PAD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GEL_PAD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GEL_PAD
                        )

    @staticmethod
    def heavy_water(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create a heavy water molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing heavy water
        """
        return Molecule(name="heavy_water",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(
                            StandardProperties.HEAVY_WATER_MUA),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                            StandardProperties.WATER_MUS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            StandardProperties.WATER_G),
                        density=StandardProperties.DENSITY_HEAVY_WATER,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_HEAVY_WATER,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_WATER
                        )

    @staticmethod
    def air(volume_fraction: (float, torch.Tensor) = 1.0) -> Molecule:
        """
        Create an air molecule with predefined properties.

        :param volume_fraction: The volume fraction of the molecule, defaults to 1.0
        :return: A Molecule object representing air
        """
        return Molecule(name="air",
                        absorption_spectrum=AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(
                            StandardProperties.AIR_MUA),
                        volume_fraction=volume_fraction,
                        scattering_spectrum=ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(
                            StandardProperties.AIR_MUS),
                        anisotropy_spectrum=AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(
                            StandardProperties.AIR_G),
                        density=StandardProperties.DENSITY_AIR,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_AIR,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_AIR
                        )


MOLECULE_LIBRARY = MoleculeLibrary()


class MolecularCompositionGenerator(object):
    """
    The MolecularCompositionGenerator is a helper class to facilitate the creation of a
    MolecularComposition instance.

    This class provides methods to build and retrieve a molecular composition by
    appending Molecule objects to a dictionary, which is then used to create a
    MolecularComposition instance.
    """

    def __init__(self):
        """
        Initialize a new MolecularCompositionGenerator.

        The constructor initializes an empty dictionary to store the molecular composition.
        """
        self.molecular_composition_dictionary = dict()

    def append(self, value: Molecule = None, key: str = None):
        """
        Append a Molecule to the molecular composition.

        Adds the given Molecule object to the molecular composition dictionary with the
        specified key. If no key is provided, the name attribute of the Molecule is used as the key.

        :param value: The Molecule object to add to the molecular composition.
        :param key: The key under which the Molecule should be stored. If None, the Molecule's name is used.
        :raises KeyError: If the specified key already exists in the molecular composition.
        :return: The current instance of MolecularCompositionGenerator.
        """
        if key is None:
            key = value.name
        if key in self.molecular_composition_dictionary:
            raise KeyError(key + " already in the molecular composition!")
        self.molecular_composition_dictionary[key] = value
        return self

    def get_molecular_composition(self, segmentation_type):
        """
        Retrieve the molecular composition as a MolecularComposition instance.

        Creates a MolecularComposition instance using the current state of the
        molecular composition dictionary and the specified segmentation type.

        :param segmentation_type: The type of segmentation to be used in the MolecularComposition.
        :return: A MolecularComposition instance.
        """
        return MolecularComposition(segmentation_type=segmentation_type,
                                    molecular_composition_settings=self.molecular_composition_dictionary)
