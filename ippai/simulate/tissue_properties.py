import numpy as np
from ippai.simulate.utils import randomize
from ippai.simulate import Tags, OpticalTissueProperties
from utils import SPECTRAL_LIBRARY, AbsorptionSpectrum, randomize_uniform


class TissueProperties(object):
    """

    """

    class Chromophore(object):

        def __init__(self, spectrum: AbsorptionSpectrum,
                     volume_fraction: float,
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
            if not isinstance(spectrum, AbsorptionSpectrum):
                raise TypeError("The given spectrum was not of type AbsorptionSpectrum!")

            self.spectrum = spectrum

            if not isinstance(volume_fraction, float):
                raise TypeError("The given volume_fraction was not of type float!")
            self.volume_fraction = volume_fraction

            if not isinstance(musp500, float):
                raise TypeError("The given musp500 was not of type float!")
            self.musp500 = musp500

            if not isinstance(f_ray, float):
                raise TypeError("The given f_ray was not of type float!")
            self.f_ray = f_ray

            if not isinstance(b_mie, float):
                raise TypeError("The given b_mie was not of type float!")
            self.b_mie = b_mie

            if not isinstance(anisotropy, float):
                raise TypeError("The given anisotropy was not of type float!")
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
            _sum_of_fractions += chromophore.volume_fraction
            _mua_per_centimeter += chromophore.volume_fraction * chromophore.spectrum.get_absorption_for_wavelength(wavelength)
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


class TissueSettingsGenerator(object):
    def __init__(self):
        self.tissue_dictionary = dict()

    def append(self, key: str, value: TissueProperties.Chromophore):
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
        tissue_generator.append("hb", TissueProperties.Chromophore(spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
                                                                   volume_fraction=bvf * (1-oxy),
                                                                   musp500=musp500, f_ray=f_ray,
                                                                   b_mie=b_mie, anisotropy=anisotropy))
        tissue_generator.append("hbO2", TissueProperties.Chromophore(spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
                                                                     volume_fraction=bvf * oxy,
                                                                     musp500=musp500, f_ray=f_ray,
                                                                     b_mie=b_mie, anisotropy=anisotropy))

    if wvf > 0:
        tissue_generator.append("water", TissueProperties.Chromophore(spectrum=SPECTRAL_LIBRARY.WATER,
                                                                      volume_fraction=wvf,
                                                                      musp500=musp500, f_ray=f_ray,
                                                                      b_mie=b_mie, anisotropy=anisotropy))

    if fvf > 0:
        tissue_generator.append("fat", TissueProperties.Chromophore(spectrum=SPECTRAL_LIBRARY.FAT,
                                                                    volume_fraction=fvf,
                                                                    musp500=musp500, f_ray=f_ray,
                                                                    b_mie=b_mie, anisotropy=anisotropy))

    if mvf > 0:
        tissue_generator.append("melanin", TissueProperties.Chromophore(spectrum=SPECTRAL_LIBRARY.MELANIN,
                                                                        volume_fraction=mvf,
                                                                        musp500=musp500, f_ray=f_ray,
                                                                        b_mie=b_mie, anisotropy=anisotropy))

    return tissue_generator.get_settings()
