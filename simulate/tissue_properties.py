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

import numpy as np
from simulate import Tags, OpticalTissueProperties
from utils import SPECTRAL_LIBRARY, randomize_uniform, Chromophore


class TissueProperties(object):
    """

    """

    def __init__(self, settings: dict):
        """
        :param settings:
        :Param legacy_normalize_scattering:
        """

        self.ensure_valid_settings(settings)

        self.chromophores = []
        self.constant_properties = None

        _keys = settings.keys()

        if Tags.KEY_CONSTANT_PROPERTIES in _keys and settings[Tags.KEY_CONSTANT_PROPERTIES] == True:
            self.constant_properties = dict()
            self.constant_properties[Tags.KEY_MUA] = settings[Tags.KEY_MUA]
            self.constant_properties[Tags.KEY_MUS] = settings[Tags.KEY_MUS]
            self.constant_properties[Tags.KEY_G] = settings[Tags.KEY_G]
        else:
            for chromophore_name in _keys:
                self.chromophores.append(settings[chromophore_name])
                print(chromophore_name)
                print(settings[chromophore_name].spectrum)

    def ensure_valid_settings(self, settings: dict):
        """
        :param settings: a dictionary containing at least one key - value pair of a string and a
                         corresponding Chromophore instance.
        """

        if settings is None:
            raise ValueError("The given settings sub-dictionary for the tissue properties must not be None.")

        if not isinstance(settings, dict):
            raise TypeError("The given settings sub-dictionary must be of type dict.")

        keys = settings.keys()

        if len(keys) < 1:
            raise ValueError("The settings dictionary must at least contain one entry.")

        if Tags.KEY_CONSTANT_PROPERTIES in keys and settings[Tags.KEY_CONSTANT_PROPERTIES] == True:
            if Tags.KEY_MUA not in keys:
                raise ValueError("Tags.KEY_MUA (" + Tags.KEY_MUA + ") was not set in the settings file.")
            else:
                if not isinstance(settings[Tags.KEY_MUA], float):
                    raise TypeError("Absorption coefficient was not of type float!")
            if Tags.KEY_MUS not in keys:
                raise ValueError("Tags.KEY_MUS (" + Tags.KEY_MUS + ") was not set in the settings file.")
            else:
                if not isinstance(settings[Tags.KEY_MUS], float):
                    raise TypeError("Scattering coefficient was not of type float!")
            if Tags.KEY_G not in keys:
                raise ValueError("Tags.KEY_G (" + Tags.KEY_G + ") was not set in the settings file.")
            else:
                if not isinstance(settings[Tags.KEY_G], float):
                    raise TypeError("Anisotropy was not of type float!")
        else:
            for key in keys:
                if not isinstance(settings[key], Chromophore):
                    raise TypeError("All value items in the dictionary must be of type Chromophore.")

    def get(self, wavelength):
        """
        TODO
        """

        if self.constant_properties is not None:
            return[self.constant_properties[Tags.KEY_MUA], self.constant_properties[Tags.KEY_MUS],
                   self.constant_properties[Tags.KEY_G]]

        _mua_per_centimeter = 0
        _mus_per_centimeter = 0
        _g = 0
        _sum_of_fractions = 0

        for chromophore in self.chromophores:
            _sum_of_fractions += chromophore.volume_fraction
            _mua_per_centimeter += chromophore.volume_fraction * \
                                   chromophore.spectrum.get_absorption_for_wavelength(wavelength)
            _g += chromophore.volume_fraction * chromophore.anisotropy
            _mus_per_centimeter += (chromophore.musp500 * (chromophore.f_ray *
                                    (wavelength / 500) ** 1e-4 + (1 - chromophore.f_ray) *
                                    (wavelength / 500) ** -chromophore.b_mie))

        # If _sum_of_fraction does not add up to one, pretend that it did
        # (we just want the weighted average for the anisotropy)
        _g = _g / _sum_of_fractions

        return [_mua_per_centimeter, _mus_per_centimeter, _g]


def get_muscle_settings(background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
    """

    :return: a settings dictionary containing all min and max parameters fitting for generic background tissue.
    """
    return get_settings(b_min=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        b_max=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        w_min=0.64, w_max=0.72,
                        musp500=OpticalTissueProperties.MUSP500_BACKGROUND_TISSUE,
                        f_ray=OpticalTissueProperties.FRAY_BACKGROUND_TISSUE,
                        b_mie=OpticalTissueProperties.BMIE_BACKGROUND_TISSUE,
                        oxy_min=background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                        oxy_max=background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)


def get_epidermis_settings(background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
    """

    :return: a settings dictionary containing all min and max parameters fitting for epidermis tissue.
    """
    return get_settings(b_min=1e-4, b_max=1e-4,
                        w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        m_max=OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN +
                              OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
                        m_min=OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN -
                              OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
                        musp500=OpticalTissueProperties.MUSP500_EPIDERMIS,
                        f_ray=OpticalTissueProperties.FRAY_EPIDERMIS,
                        b_mie=OpticalTissueProperties.BMIE_EPIDERMIS,
                        oxy_min=background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                        oxy_max=background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)


def get_dermis_settings(background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
    """

    :return: a settings dictionary containing all min and max parameters fitting for dermis tissue.
    """
    return get_settings(b_min=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        b_max=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        musp500=OpticalTissueProperties.MUSP500_DERMIS,
                        f_ray=OpticalTissueProperties.FRAY_DERMIS,
                        b_mie=OpticalTissueProperties.BMIE_DERMIS,
                        oxy_min=background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                        oxy_max=background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)


def get_subcutaneous_fat_settings(background_oxy=OpticalTissueProperties.BACKGROUND_OXYGENATION):
    """

    :return: a settings dictionary containing all min and max parameters fitting for subcutaneous fat tissue.
    """
    return get_settings(b_min=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        b_max=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
                        w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        f_min=0.3, f_max=0.4,  #TODO: Add "correct" fat volume fraction of human adipose tissue
                        musp500=OpticalTissueProperties.MUSP500_FAT,
                        f_ray=OpticalTissueProperties.FRAY_FAT,
                        b_mie=OpticalTissueProperties.BMIE_FAT,
                        oxy_min=background_oxy - OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION,
                        oxy_max=background_oxy + OpticalTissueProperties.BACKGROUND_OXYGENATION_VARIATION)


def get_blood_settings():
    """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
    return get_settings(b_min=1, b_max=1, w_min=1, w_max=1,
                        musp500=OpticalTissueProperties.MUSP500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD)


def get_arterial_blood_settings():
    """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
    return get_settings(b_min=1, b_max=1, w_min=1, w_max=1,
                        oxy_min=0.8,
                        oxy_max=1,
                        musp500=OpticalTissueProperties.MUSP500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD)


def get_venous_blood_settings():
    """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
    return get_settings(b_min=1, b_max=1, w_min=1, w_max=1,
                        oxy_min=0,
                        oxy_max=0.8,
                        musp500=OpticalTissueProperties.MUSP500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD)


def get_bone_settings():
    """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
    return get_settings(b_min=1e-4, b_max=1e-4, w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN -
                                                      OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN +
                              OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD,
                        oxy_min=0, oxy_max=1,
                        musp500=OpticalTissueProperties.MUSP500_BONE,
                        b_mie=OpticalTissueProperties.BMIE_BONE,
                        f_ray=OpticalTissueProperties.FRAY_BONE)


def get_random_tube_settings():
    """
    :return: a settings dictionary containing random min and max parameters.
    """
    return get_settings(b_min=1, b_max=1, w_min=1, w_max=1,
                        oxy_min=0,
                        oxy_max=1,
                        musp500=OpticalTissueProperties.MUSP500_BLOOD,
                        b_mie=OpticalTissueProperties.BMIE_BLOOD,
                        f_ray=OpticalTissueProperties.FRAY_BLOOD)


def get_random_background_settings():
    """

        :return: a settings dictionary containing random min and max parameters.
        """
    random_musp = np.random.randint(8, 18)
    return get_settings(b_min=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE / 2,
                        b_max=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE * 2,
                        w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_HUMAN_BODY,
                        musp500=random_musp,
                        b_mie=0,
                        f_ray=0)


def get_constant_settings(mua, mus, g):
    return_dict = dict()
    return_dict[Tags.KEY_CONSTANT_PROPERTIES] = True
    return_dict[Tags.KEY_MUA] = mua
    return_dict[Tags.KEY_MUS] = mus
    return_dict[Tags.KEY_G] = g
    return return_dict


class TissueSettingsGenerator(object):
    def __init__(self):
        self.tissue_dictionary = dict()

    def append(self, key: str, value: Chromophore):
        self.tissue_dictionary[key] = value

    def get_settings(self):
        return self.tissue_dictionary


def get_settings(b_min=0.0, b_max=0.0,
                 w_min=0.0, w_max=0.0,
                 f_min=0.0, f_max=0.0,
                 m_min=0.0, m_max=0.0,
                 oxy_min=0.0, oxy_max=1.0,
                 musp500=10.0, f_ray=0.0, b_mie=0.0,
                 anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY):

    tissue_generator = TissueSettingsGenerator()

    bvf = randomize_uniform(b_min, b_max)  # Blood volume fraction (bvf)
    wvf = randomize_uniform(w_min, w_max)  # Water volume fraction (wvf)
    fvf = randomize_uniform(f_min, f_max)  # Fat volume fraction (fvf)
    mvf = randomize_uniform(m_min, m_max)  # Melanin volume fraction (mvf)
    oxy = randomize_uniform(oxy_min, oxy_max)  # Blood oxygenation (oxy)

    if bvf > 0:
        tissue_generator.append("hb", Chromophore(spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
                                                                   volume_fraction=bvf * (1-oxy),
                                                                   musp500=musp500, f_ray=f_ray,
                                                                   b_mie=b_mie, anisotropy=anisotropy))
        tissue_generator.append("hbO2", Chromophore(spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
                                                                     volume_fraction=bvf * oxy,
                                                                     musp500=musp500, f_ray=f_ray,
                                                                     b_mie=b_mie, anisotropy=anisotropy))

    if wvf > 0:
        tissue_generator.append("water", Chromophore(spectrum=SPECTRAL_LIBRARY.WATER,
                                                                      volume_fraction=wvf,
                                                                      musp500=musp500, f_ray=f_ray,
                                                                      b_mie=b_mie, anisotropy=anisotropy))

    if fvf > 0:
        tissue_generator.append("fat", Chromophore(spectrum=SPECTRAL_LIBRARY.FAT,
                                                                    volume_fraction=fvf,
                                                                    musp500=musp500, f_ray=f_ray,
                                                                    b_mie=b_mie, anisotropy=anisotropy))

    if mvf > 0:
        tissue_generator.append("melanin", Chromophore(spectrum=SPECTRAL_LIBRARY.MELANIN,
                                                                        volume_fraction=mvf,
                                                                        musp500=musp500, f_ray=f_ray,
                                                                        b_mie=b_mie, anisotropy=anisotropy))

    return tissue_generator.get_settings()
