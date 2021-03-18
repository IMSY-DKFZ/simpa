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

import numpy as np
from simpa.utils import Tags
from simpa.utils.tissue_properties import TissueProperties
from simpa.utils.libraries.literature_values import OpticalTissueProperties, StandardProperties
from simpa.utils import AbsorptionSpectrum
from simpa.utils import SPECTRAL_LIBRARY
from simpa.utils.calculate import calculate_oxygenation, calculate_gruneisen_parameter_from_temperature


class MolecularComposition(list):

    def __init__(self, segmentation_type=None, molecular_composition_settings=None):
        super().__init__()
        self.segmentation_type = segmentation_type
        self.internal_properties = TissueProperties()
        self.cached_absorption = np.ones((5000, )) * -1
        self.cached_scattering = np.ones((5000,)) * -1

        if molecular_composition_settings is None:
            return

        _keys = molecular_composition_settings.keys()
        for molecule_name in _keys:
            self.append(molecular_composition_settings[molecule_name])

    def update_internal_properties(self):
        """
        FIXME
        """
        self.internal_properties = TissueProperties()
        self.internal_properties[Tags.PROPERTY_SEGMENTATION] = self.segmentation_type
        self.internal_properties[Tags.PROPERTY_OXYGENATION] = calculate_oxygenation(self)
        for molecule in self:
            self.internal_properties.volume_fraction += molecule.volume_fraction
            self.internal_properties[Tags.PROPERTY_ANISOTROPY] += molecule.volume_fraction * molecule.anisotropy
            self.internal_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER] += \
                molecule.volume_fraction * molecule.gruneisen_parameter
            self.internal_properties[Tags.PROPERTY_DENSITY] += molecule.volume_fraction * molecule.density
            self.internal_properties[Tags.PROPERTY_SPEED_OF_SOUND] += molecule.volume_fraction * molecule.speed_of_sound
            self.internal_properties[Tags.PROPERTY_ALPHA_COEFF] += molecule.volume_fraction * molecule.alpha_coefficient

        if self.internal_properties.volume_fraction > 1:
            self.internal_properties[Tags.PROPERTY_ANISOTROPY] /= self.internal_properties.volume_fraction
            # The maximum volume fraction any given molecular composition can have is 1!
            self.internal_properties.volume_fraction = 1

    def get_properties_for_wavelength(self, wavelength) -> TissueProperties:

        self.update_internal_properties()
        if self.cached_absorption[wavelength] != -1:
            self.internal_properties[Tags.PROPERTY_ABSORPTION_PER_CM] = self.cached_absorption[wavelength]
            self.internal_properties[Tags.PROPERTY_SCATTERING_PER_CM] = self.cached_scattering[wavelength]
        else:
            self.internal_properties[Tags.PROPERTY_ABSORPTION_PER_CM] = 0
            self.internal_properties[Tags.PROPERTY_SCATTERING_PER_CM] = 0
            for molecule in self:
                self.internal_properties[Tags.PROPERTY_ABSORPTION_PER_CM] += \
                    (molecule.volume_fraction * molecule.spectrum.get_absorption_for_wavelength(wavelength))
                self.internal_properties[Tags.PROPERTY_SCATTERING_PER_CM] += \
                    (molecule.volume_fraction * (molecule.mus500 * (molecule.f_ray *
                                                                    (wavelength / 500) ** 1e-4 + (1 - molecule.f_ray) *
                                                                    (wavelength / 500) ** -molecule.b_mie)))
            self.cached_absorption[wavelength] = self.internal_properties[Tags.PROPERTY_ABSORPTION_PER_CM]
            self.cached_scattering[wavelength] = self.internal_properties[Tags.PROPERTY_SCATTERING_PER_CM]
        return self.internal_properties


class Molecule(object):

    def __init__(self, name: str = None,
                 spectrum: AbsorptionSpectrum = None,
                 volume_fraction: float = None,
                 mus500: float = None, f_ray: float = None, b_mie: float = None,
                 anisotropy: float = None, gruneisen_parameter: float = None,
                 density: float = None, speed_of_sound: float = None,
                 alpha_coefficient: float = None):
        """
        :param name: str
        :param spectrum: AbsorptionSpectrum
        :param volume_fraction: float
        :param mus500: float The scattering coefficient at 500 nanometers
        :param f_ray: float
        :param b_mie: float
        :param anisotropy: float
        :param gruneisen_parameter: float
        :param density: float
        :param speed_of_sound: float
        :param alpha_coefficient: float
        """
        if name is None:
            name = "GenericMoleculeName"
        if not isinstance(name, str):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            else:
                raise TypeError("Molecule name must be of type str or bytes instead of {}!".format(type(name)))
        self.name = name

        if spectrum is None:
            spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
        if isinstance(spectrum, dict):
            spectrum = spectrum[list(spectrum.keys())[0]]
        if not isinstance(spectrum, AbsorptionSpectrum):
            raise TypeError("The given spectrum was not of type AbsorptionSpectrum! Instead: " + str(type(spectrum)) +
                            "and reads: " + str(spectrum))
        self.spectrum = spectrum

        if volume_fraction is None:
            volume_fraction = 0.0
        if not isinstance(volume_fraction, (int, float, np.int64)):
            raise TypeError("The given volume_fraction was not of type float instead of {}!"
                            .format(type(volume_fraction)))
        self.volume_fraction = volume_fraction

        if mus500 is None:
            mus500 = 1e-20
        if not isinstance(mus500, (int, float)):
            raise TypeError("The given mus500 was not of type float instead of {}!".format(type(mus500)))
        self.mus500 = mus500

        if f_ray is None:
            f_ray = 0.0
        if not isinstance(f_ray, float):
            raise TypeError("The given f_ray was not of type float instead of {}!".format(type(f_ray)))
        self.f_ray = f_ray

        if b_mie is None:
            b_mie = 0.0
        if not isinstance(b_mie, float):
            raise TypeError("The given b_mie was not of type float instead of {}!".format(type(b_mie)))
        self.b_mie = b_mie

        if anisotropy is None:
            anisotropy = 0.0
        if not isinstance(anisotropy, float):
            raise TypeError("The given anisotropy was not of type float instead of {}!".format(type(anisotropy)))
        self.anisotropy = anisotropy

        if gruneisen_parameter is None:
            gruneisen_parameter = calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)
        if not isinstance(gruneisen_parameter, (int, float)):
            raise TypeError("The given gruneisen_parameter was not of type int or float instead of {}!"
                            .format(type(gruneisen_parameter)))
        self.gruneisen_parameter = gruneisen_parameter

        if density is None:
            density = StandardProperties.DENSITY_GENERIC
        if not isinstance(density, (np.int32, np.int64, int, float)):
            raise TypeError("The given density was not of type int or float instead of {}!".format(type(density)))
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

    def __eq__(self, other):
        if isinstance(other, Molecule):
            return (self.name == other.name and
                    self.spectrum == other.spectrum and
                    self.volume_fraction == other.volume_fraction and
                    self.mus500 == other.mus500 and
                    self.alpha_coefficient == other.alpha_coefficient and
                    self.f_ray == other.f_ray and
                    self.b_mie == other.b_mie and
                    self.speed_of_sound == other.speed_of_sound and
                    self.gruneisen_parameter == other.gruneisen_parameter and
                    self.anisotropy == other.anisotropy and
                    self.density == other.density
                    )
        else:
            return super().__eq__(other)

    @staticmethod
    def from_settings(settings):
        return Molecule(name=settings["name"],
                        spectrum=settings["spectrum"],
                        volume_fraction=settings["volume_fraction"],
                        mus500=settings["mus500"],
                        alpha_coefficient=settings["alpha_coefficient"],
                        f_ray=settings["f_ray"],
                        b_mie=settings["b_mie"],
                        speed_of_sound=settings["speed_of_sound"],
                        gruneisen_parameter=settings["gruneisen_parameter"],
                        anisotropy=settings["anisotropy"],
                        density=settings["density"])


class MoleculeLibrary(object):

    # Main absorbers
    @staticmethod
    def water(volume_fraction: float = 1.0):
        return Molecule(name="water",
                        spectrum=SPECTRAL_LIBRARY.WATER,
                        volume_fraction=volume_fraction, mus500=StandardProperties.WATER_MUS,
                        b_mie=0.0, f_ray=0.0, anisotropy=StandardProperties.WATER_G,
                        density=StandardProperties.DENSITY_WATER,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_WATER,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_WATER
                        )

    @staticmethod
    def oxyhemoglobin(volume_fraction: float = 1.0):
        return Molecule(name="oxyhemoglobin",
                        spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_BLOOD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BLOOD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BLOOD
                        )

    @staticmethod
    def deoxyhemoglobin(volume_fraction: float = 1.0):
        return Molecule(name="deoxyhemoglobin",
                        spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_BLOOD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BLOOD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BLOOD
                        )

    @staticmethod
    def melanin(volume_fraction: float = 1.0):
        return Molecule(name="melanin",
                        spectrum=SPECTRAL_LIBRARY.MELANIN,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_EPIDERMIS,
                        b_mie=OpticalTissueProperties.BMIE_EPIDERMIS,
                        f_ray=OpticalTissueProperties.FRAY_EPIDERMIS,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def fat(volume_fraction: float = 1.0):
        return Molecule(name="fat",
                        spectrum=SPECTRAL_LIBRARY.FAT,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_FAT,
                        b_mie=OpticalTissueProperties.BMIE_FAT,
                        f_ray=OpticalTissueProperties.FRAY_FAT,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_FAT,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_FAT,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_FAT
                        )

    # Scatterers
    @staticmethod
    def constant_scatterer(scattering_coefficient: float = 100.0, anisotropy: float = 0.9,
                           volume_fraction: float = 1.0):
        return Molecule(name="constant_scatterer",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                        volume_fraction=volume_fraction, mus500=scattering_coefficient,
                        b_mie=0.0, f_ray=0.0, anisotropy=anisotropy,
                        density=StandardProperties.DENSITY_GENERIC,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GENERIC,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GENERIC
                        )

    @staticmethod
    def soft_tissue_scatterer(volume_fraction: float = 1.0):
        return Molecule(name="soft_tissue_scatterer",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_BACKGROUND_TISSUE,
                        b_mie=OpticalTissueProperties.BMIE_BACKGROUND_TISSUE,
                        f_ray=OpticalTissueProperties.FRAY_BACKGROUND_TISSUE,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_GENERIC,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GENERIC,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GENERIC
                        )

    @staticmethod
    def epidermal_scatterer(volume_fraction: float = 1.0):
        return Molecule(name="epidermal_scatterer",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_EPIDERMIS,
                        b_mie=OpticalTissueProperties.BMIE_EPIDERMIS,
                        f_ray=OpticalTissueProperties.FRAY_EPIDERMIS,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def dermal_scatterer(volume_fraction: float = 1.0):
        return Molecule(name="dermal_scatterer",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_DERMIS,
                        b_mie=OpticalTissueProperties.BMIE_DERMIS,
                        f_ray=OpticalTissueProperties.FRAY_DERMIS,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_SKIN,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_SKIN,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_SKIN
                        )

    @staticmethod
    def bone(volume_fraction: float = 1.0):
        return Molecule(name="bone",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(OpticalTissueProperties.BONE_ABSORPTION),
                        volume_fraction=volume_fraction,
                        mus500=OpticalTissueProperties.MUS500_BONE,
                        b_mie=OpticalTissueProperties.BMIE_BONE,
                        f_ray=OpticalTissueProperties.FRAY_BONE,
                        anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY,
                        density=StandardProperties.DENSITY_BONE,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_BONE,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_BONE
                        )

    @staticmethod
    def mediprene(volume_fraction: float = 1.0):
        return Molecule(name="mediprene",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(-np.log(0.85) / 10),  # FIXME
                        volume_fraction=volume_fraction,
                        mus500=(-np.log(0.85)) - (-np.log(0.85) / 10),  # FIXME
                        b_mie=0.0,
                        f_ray=0.0,
                        anisotropy=0.9,
                        density=StandardProperties.DENSITY_GEL_PAD,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_GEL_PAD,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_GEL_PAD
                        )

    @staticmethod
    def heavy_water(volume_fraction: float = 1.0):
        return Molecule(name="heavy_water",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.HEAVY_WATER_MUA),
                        volume_fraction=volume_fraction,
                        mus500=StandardProperties.WATER_MUS,
                        b_mie=0.0,
                        f_ray=0.0,
                        anisotropy=StandardProperties.WATER_G,
                        density=StandardProperties.DENSITY_HEAVY_WATER,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_HEAVY_WATER,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_WATER
                        )
    @staticmethod
    def air(volume_fraction: float = 1.0):
        return Molecule(name="air",
                        spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA),
                        volume_fraction=volume_fraction,
                        mus500=StandardProperties.AIR_MUS,
                        b_mie=0.0,
                        f_ray=0.0,
                        anisotropy=StandardProperties.AIR_G,
                        density=StandardProperties.DENSITY_AIR,
                        speed_of_sound=StandardProperties.SPEED_OF_SOUND_AIR,
                        alpha_coefficient=StandardProperties.ALPHA_COEFF_AIR
                        )


MOLECULE_LIBRARY = MoleculeLibrary()
