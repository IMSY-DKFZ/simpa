import numpy as np
from ippai.simulate.utils import randomize
from ippai.simulate import Tags, OpticalTissueProperties
from utils import SPECTRAL_LIBRARY, AbsorptionSpectrum


class TissueProperties(object):
    """

    """

    class Chromophore(object):

        def __init__(self, spectrum: AbsorptionSpectrum,
                     max_fraction: float, min_fraction: float,
                     musp500: float, f_ray: float, b_mie: float,
                     anisotropy: float):
            """
            :param spectrum: AbsorptionSpectrum
            :param max_fraction: float
            :param min_fraction: float
            :param musp500: float
            :param f_ray: float
            :param b_mie: float
            :param anisotropy: float
            """

            self.spectrum = spectrum
            if not isinstance(spectrum, AbsorptionSpectrum):
                raise TypeError("The given spectrum was not of type AbsorptionSpectrum!")

            self.fraction = (np.random.random() * (max_fraction - min_fraction)) + min_fraction
            self.musp500 = musp500,
            self.f_ray = f_ray
            self.b_mie = b_mie
            self.anisotropy = anisotropy

    def __init__(self, settings):
        """

        """

        self.chromophores = []

        for chromophore_name in settings.keys():
            self.chromophores.append(settings[chromophore_name])
            print(chromophore_name)
            print(settings[chromophore_name].spectrum)

    def get(self, wavelength):

        _mua_per_centimeter = 0
        _mus_per_centimeter = 0
        _g = 0
        _sum_of_fractions = 0

        for chromophore in self.chromophores:
            _sum_of_fractions += chromophore.fraction
            _mua_per_centimeter += chromophore.fraction * chromophore.spectrum.get_absorption_for_wavelength(wavelength)
            _g += chromophore.fraction * chromophore.anisotropy
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
    return get_settings(b_min=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE, b_max=OpticalTissueProperties.BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE,
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
                        m_max=OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN + OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
                        m_min=OpticalTissueProperties.MELANIN_VOLUME_FRACTION_MEAN- OpticalTissueProperties.MELANIN_VOLUME_FRACTION_STD,
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
    return get_settings(b_min=1e-4, b_max=1e-4, w_min=OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN - OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD,
                        w_max=OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_MEAN + OpticalTissueProperties.WATER_VOLUME_FRACTION_BONE_STD,
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


def get_settings(b_min=0.0, b_max=0.0,
                 w_min=0.0, w_max=0.0,
                 f_min=0.0, f_max=0.0,
                 m_min=0.0, m_max=0.0,
                 oxy_min=0.0, oxy_max=1.0,
                 musp500=10.0, f_ray=0.0, b_mie=0.0,
                 anisotropy=OpticalTissueProperties.STANDARD_ANISOTROPY):
    return_dict = dict()
    return_dict[Tags.KEY_B_MIN] = b_min
    return_dict[Tags.KEY_B_MAX] = b_max
    return_dict[Tags.KEY_W_MIN] = w_min
    return_dict[Tags.KEY_W_MAX] = w_max
    return_dict[Tags.KEY_F_MIN] = f_min
    return_dict[Tags.KEY_F_MAX] = f_max
    return_dict[Tags.KEY_M_MIN] = m_min
    return_dict[Tags.KEY_M_MAX] = m_max
    return_dict[Tags.KEY_OXY_MIN] = oxy_min
    return_dict[Tags.KEY_OXY_MAX] = oxy_max
    return_dict[Tags.KEY_MUSP500] = musp500
    return_dict[Tags.KEY_F_RAY] = f_ray
    return_dict[Tags.KEY_B_MIE] = b_mie
    return_dict[Tags.KEY_ANISOTROPY] = anisotropy
    return return_dict
