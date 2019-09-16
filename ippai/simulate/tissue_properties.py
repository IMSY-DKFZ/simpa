import numpy as np
from ippai.simulate.utils import randomize
from ippai.simulate import Tags, OpticalTissueProperties
import os


class TissueProperties(object):

    def __init__(self, settings, tissue_type, shape, spacing):
        """

        The TissueProperties class encapsules the generation of physical properties from tissue parameters.
        The tissue parameters that are of relevance are blood volume fraction (B), water volume fraction (W),
        fat volume fraction (F), melanin volume fraction (M), and blood oxygenation (OXY).

        From these parameters, depending on the chosen wavelength l, the parameters will be calculated:

            [mu_a, mu_s, g] = f(B, W, F, M, OXY, l)

        To enable randomized instantiation, the input settings contain a upper and lower limit of the parameters,
        between which the parameter p will be drawn from a uniform distribution:

            p = U(p_min, p_max)

        Usage example:

            s = get_settings()
            tissue_type = "MyTissueType"
            wavelength = 800
            my_tissue_type_properties = TissueProperties(s, tissue_type)
            [mua, mus, g] = my_tissue_type_properties.get(wavelength)

        :param settings: The simulation settings dictionary
        :param tissue_type: The tissue type to check in the settings file
        """
        self.KEYS_IN_ORDER = [Tags.KEY_B, Tags.KEY_W, Tags.KEY_F, Tags.KEY_M, Tags.KEY_OXY]
        self.B_min = None
        self.B_max = None
        self.W_min = None
        self.W_max = None
        self.F_min = None
        self.F_max = None
        self.M_min = None
        self.M_max = None
        self.OXY_min = None
        self.OXY_max = None
        self.bvf = None
        self.wvf = None
        self.fvf = None
        self.mvf = None
        self.oxy = None
        self.musp500 = None
        self.f_ray = None
        self.b_mie = None
        self.anisotropy = None
        self.constant_properties = False
        self.constant_mua = None
        self.constant_mus = None
        self.constant_g = None

        self.KEYWORDS = [Tags.KEY_B_MIN, Tags.KEY_B_MAX, Tags.KEY_W_MAX, Tags.KEY_W_MIN, Tags.KEY_F_MAX, Tags.KEY_F_MIN,
                         Tags.KEY_M_MAX, Tags.KEY_M_MIN, Tags.KEY_OXY_MAX, Tags.KEY_OXY_MIN]
        if settings is not None:
            if Tags.KEY_CONSTANT_PROPERTIES in settings[Tags.STRUCTURE_TISSUE_PROPERTIES]:
                if settings[Tags.STRUCTURE_TISSUE_PROPERTIES][Tags.KEY_CONSTANT_PROPERTIES] is True:
                    self.constant_properties = True
                    self.constant_mua = settings[Tags.STRUCTURE_TISSUE_PROPERTIES][Tags.KEY_MUA]
                    self.constant_mus = settings[Tags.STRUCTURE_TISSUE_PROPERTIES][Tags.KEY_MUS]
                    self.constant_g = settings[Tags.STRUCTURE_TISSUE_PROPERTIES][Tags.KEY_G]
                    return
            self.ensure_valid_settings_file(settings, tissue_type)
            self.B_min = settings[tissue_type][Tags.KEY_B_MIN]
            self.B_max = settings[tissue_type][Tags.KEY_B_MAX]
            self.W_min = settings[tissue_type][Tags.KEY_W_MIN]
            self.W_max = settings[tissue_type][Tags.KEY_W_MAX]
            self.F_min = settings[tissue_type][Tags.KEY_F_MIN]
            self.F_max = settings[tissue_type][Tags.KEY_F_MAX]
            self.M_min = settings[tissue_type][Tags.KEY_M_MIN]
            self.M_max = settings[tissue_type][Tags.KEY_M_MAX]
            self.OXY_min = settings[tissue_type][Tags.KEY_OXY_MIN]
            self.OXY_max = settings[tissue_type][Tags.KEY_OXY_MAX]
            self.musp500 = settings[tissue_type][Tags.KEY_MUSP500]
            self.f_ray = settings[tissue_type][Tags.KEY_F_RAY]
            self.b_mie = settings[tissue_type][Tags.KEY_B_MIE]
            self.anisotropy = settings[tissue_type][Tags.KEY_ANISOTROPY]

            distributions = dict()
            sizes = dict()
            gauss_size = dict()
            for key in self.KEYS_IN_ORDER:
                distributions[key] = 'uniform'
                sizes[key] = (1,)
                gauss_size[key] = 0

            if Tags.STRUCTURE_USE_DISTORTION in settings:
                if settings[Tags.STRUCTURE_USE_DISTORTION]:
                    if Tags.STRUCTURE_DISTORTED_PARAM_LIST in settings:
                        for key in settings[Tags.STRUCTURE_DISTORTED_PARAM_LIST]:
                            distributions[key] = 'normal'
                            sizes[key] = shape
                            if (Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM in settings and
                                    settings[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] is not None):
                                gauss_size[key] = settings[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] / spacing

            self.randomize(distributions, sizes, gauss_size)

        try:
            self.absorption_data = np.load("/home/kris/hard_drive/ippai/data/absorption_450_1000.npz")
        except FileNotFoundError:
            try:
                self.absorption_data = np.load("../../data/absorption_450_1000.npz")
            except FileNotFoundError:
                try:
                    self.absorption_data = np.load("../../ippai/data/absorption_450_1000.npz")
                except FileNotFoundError:
                    raise Exception("The absorption data could not be found in the ippai/data folder.")

    def ensure_valid_settings_file(self, settings, tissue_type):
        """
        Method to ensure that all necessary parameter limits (min & max) are set for the tissue
        parameters B, W, F, M, and OXY.

        :param settings: The simulation settings dictionary
        :param tissue_type: The tissue type to check in the settings file

        :raises FileNotFoundError: if the settings are None
        :raises LookupError: if the tissue type settings are not given or any of the min max tags is missing

        :return: None
        """

        if settings is None:
            raise FileNotFoundError("settings file not given")

        if settings[tissue_type] is None:
            raise LookupError("Tissue settings for " + str(tissue_type) + " are not contained in settings file")

        for _keyword in self.KEYWORDS:
            if settings[tissue_type][_keyword] is None:
                raise LookupError("Tissue settings for " + str(_keyword) +
                                  " are not contained in  " + str(tissue_type) + " settings")

    def get(self, wavelength):

        if self.constant_properties:
            return [self.constant_mua, self.constant_mus, self.constant_g, -1.0]

        min_wavelength = self.absorption_data['min_wavelength']
        max_wavelength = self.absorption_data['max_wavelength']
        wavelength_idx = wavelength - min_wavelength
        if wavelength_idx < 0 or wavelength_idx > (max_wavelength-min_wavelength+1):
            raise AssertionError("Wavelengths only supported between " + str(min_wavelength) +
                                 " nm and " + str(max_wavelength) + " nm")

        absorption = (self.wvf * self.absorption_data['water'][wavelength_idx] +
                      self.fvf * self.absorption_data['fat'][wavelength_idx] +
                      self.mvf * self.absorption_data['melanin'][wavelength_idx] +
                      self.bvf * self.oxy * self.absorption_data['hbo2'][wavelength_idx] +
                      self.bvf * (1-self.oxy) * self.absorption_data['hb'][wavelength_idx])

        scattering_p = self.musp500 * (self.f_ray * (wavelength / 500) ** 1e-4 +
                                       (1 - self.f_ray) * (wavelength / 500) ** -self.b_mie)
        anisotropy = self.anisotropy
        scattering = scattering_p / (1-anisotropy)

        return [absorption, scattering, anisotropy, self.oxy]

    def randomize(self, dist, size, gauss_size):
        """
        Randomizes the tissue parameters within the given bounds.

        :param dist: TODO
        :param size: TODO
        :param gauss_size: TODO
        :return: None
        """

        self.bvf = randomize(self.B_min, self.B_max, distribution=dist[Tags.KEY_B], size=size[Tags.KEY_B],
                             gauss_kernel_size=gauss_size[Tags.KEY_B])
        self.wvf = randomize(self.W_min, self.W_max, distribution=dist[Tags.KEY_W], size=size[Tags.KEY_W],
                             gauss_kernel_size=gauss_size[Tags.KEY_W])
        self.fvf = randomize(self.F_min, self.F_max, distribution=dist[Tags.KEY_F], size=size[Tags.KEY_F],
                             gauss_kernel_size=gauss_size[Tags.KEY_F])
        self.mvf = randomize(self.M_min, self.M_max, distribution=dist[Tags.KEY_M], size=size[Tags.KEY_M],
                             gauss_kernel_size=gauss_size[Tags.KEY_M])
        self.oxy = randomize(self.OXY_min, self.OXY_max, distribution=dist[Tags.KEY_OXY], size=size[Tags.KEY_OXY],
                             gauss_kernel_size=gauss_size[Tags.KEY_OXY])


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
