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

from simpa.utils import OpticalTissueProperties
from simpa.utils import AbsorptionSpectrum
from simpa.utils import SPECTRAL_LIBRARY

class Chromophore(object):

    def __init__(self, spectrum: AbsorptionSpectrum,
                 volume_fraction: float,
                 mus500: float, f_ray: float, b_mie: float,
                 anisotropy: float):
        """
        :param spectrum: AbsorptionSpectrum
        :param volume_fraction: float
        :param mus500: float The scattering coefficient at 500 nanometers
        :param f_ray: float
        :param b_mie: float
        :param anisotropy: float
        """
        if not isinstance(spectrum, AbsorptionSpectrum):
            raise TypeError("The given spectrum was not of type AbsorptionSpectrum!")
        self.spectrum = spectrum

        if not isinstance(volume_fraction, float):
            raise TypeError("The given volume_fraction was not of type float!")
        self.volume_fraction = volume_fraction

        if not isinstance(mus500, float):
            raise TypeError("The given mus500 was not of type float!")
        self.mus500 = mus500

        if not isinstance(f_ray, float):
            raise TypeError("The given f_ray was not of type float!")
        self.f_ray = f_ray

        if not isinstance(b_mie, float):
            raise TypeError("The given b_mie was not of type float!")
        self.b_mie = b_mie

        if not isinstance(anisotropy, float):
            raise TypeError("The given anisotropy was not of type float!")
        self.anisotropy = anisotropy


class ChromophoreLibrary(object):

    # Main absorbers
    def water(self, volume_fraction: float = 1.0):
        return Chromophore(spectrum=SPECTRAL_LIBRARY.WATER,
                           volume_fraction=volume_fraction, mus500=0.0,
                           b_mie=0.0, f_ray=0.0, anisotropy=1.0)

    def oxyhemoglobin(self, volume_fraction: float = 1.0):
        return Chromophore(
            spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
            volume_fraction=volume_fraction,
            mus500=OpticalTissueProperties.MUS500_BLOOD,
            b_mie=OpticalTissueProperties.BMIE_BLOOD,
            f_ray=OpticalTissueProperties.FRAY_BLOOD,
            anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY
        )

    def deoxyhemoglobin(self, volume_fraction: float = 1.0):
        return Chromophore(
            spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
            volume_fraction=volume_fraction,
            mus500=OpticalTissueProperties.MUS500_BLOOD,
            b_mie=OpticalTissueProperties.BMIE_BLOOD,
            f_ray=OpticalTissueProperties.FRAY_BLOOD,
            anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY
        )

    def melanin(self, volume_fraction: float = 1.0):
        return Chromophore(
            spectrum=SPECTRAL_LIBRARY.MELANIN,
            volume_fraction=volume_fraction,
            mus500=OpticalTissueProperties.MUS500_EPIDERMIS,
            b_mie=OpticalTissueProperties.BMIE_EPIDERMIS,
            f_ray=OpticalTissueProperties.FRAY_EPIDERMIS,
            anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY
        )

    def fat(self, volume_fraction: float = 1.0):
        return Chromophore(
            spectrum=SPECTRAL_LIBRARY.FAT,
            volume_fraction=volume_fraction,
            mus500=OpticalTissueProperties.MUS500_FAT,
            b_mie=OpticalTissueProperties.BMIE_FAT,
            f_ray=OpticalTissueProperties.FRAY_FAT,
            anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY
        )

    # Scatterers
    def constant_scatterer(self, scattering_coefficient: float = 100.0, anisotropy: float = 0.9,
                           volume_fraction: float = 1.0):
        return Chromophore(spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                           volume_fraction=volume_fraction, mus500=scattering_coefficient,
                           b_mie=0.0, f_ray=0.0, anisotropy=anisotropy)

    def soft_tissue_scatterer(self, volume_fraction: float = 1.0):
        return Chromophore(spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                           volume_fraction=volume_fraction,
                           mus500=OpticalTissueProperties.MUS500_BACKGROUND_TISSUE,
                           b_mie=OpticalTissueProperties.BMIE_BACKGROUND_TISSUE,
                           f_ray=OpticalTissueProperties.FRAY_BACKGROUND_TISSUE,
                           anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY)

    def epidermal_scatterer(self, volume_fraction: float = 1.0):
        return Chromophore(spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                           volume_fraction=volume_fraction,
                           mus500=OpticalTissueProperties.MUS500_EPIDERMIS,
                           b_mie=OpticalTissueProperties.BMIE_EPIDERMIS,
                           f_ray=OpticalTissueProperties.FRAY_EPIDERMIS,
                           anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY)

    def dermal_scatterer(self, volume_fraction: float = 1.0):
        return Chromophore(spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
                           volume_fraction=volume_fraction,
                           mus500=OpticalTissueProperties.MUS500_DERMIS,
                           b_mie=OpticalTissueProperties.BMIE_DERMIS,
                           f_ray=OpticalTissueProperties.FRAY_DERMIS,
                           anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY)

    def bone_scatterer(self, volume_fraction: float = 1.0):
        return Chromophore(
            spectrum=SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
            volume_fraction=volume_fraction,
            mus500=OpticalTissueProperties.MUS500_BONE,
            b_mie=OpticalTissueProperties.BMIE_BONE,
            f_ray=OpticalTissueProperties.FRAY_BONE,
            anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY
        )


CHROMOPHORE_LIBRARY = ChromophoreLibrary()
