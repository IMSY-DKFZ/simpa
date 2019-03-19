import numpy as np
from ippai.simulate.utils import randomize
from ippai.simulate import Tags


class TissueProperties(object):

    def __init__(self, settings, tissue_type):
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

        self.KEYWORDS = [Tags.KEY_B_MIN, Tags.KEY_B_MAX, Tags.KEY_W_MAX, Tags.KEY_W_MIN, Tags.KEY_F_MAX, Tags.KEY_F_MIN,
                         Tags.KEY_M_MAX, Tags.KEY_M_MIN, Tags.KEY_OXY_MAX, Tags.KEY_OXY_MIN]
        if settings is not None:
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

            self.randomize()

        self.absorption_data = np.load("../data/absorption.npz")

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
        wavelength = wavelength-700
        if wavelength < 0 or wavelength > 251:
            raise AssertionError("Wavelengths only supported between 700 nm and 950 nm")

        absorption = self.wvf * self.absorption_data['water'][wavelength] +\
            self.fvf * self.absorption_data['fat'][wavelength] +\
            self.mvf * self.absorption_data['melanin'][wavelength] +\
            self.bvf * self.oxy * self.absorption_data['hbo2'][wavelength] +\
            self.bvf * (1-self.oxy) * self.absorption_data['hb'][wavelength]
        scattering = 100  # FIXME: Include scattering term
        anisotropy = 0.9  # FIXME: Include anisotropy term

        return [absorption, scattering, anisotropy]

    def randomize(self):
        """
        Randomizes the tissue parameters within the given bounds.
        :return: None
        """
        self.bvf = randomize(self.B_min, self.B_max)
        self.wvf = randomize(self.W_min, self.W_max)
        self.fvf = randomize(self.F_min, self.F_max)
        self.mvf = randomize(self.M_min, self.M_max)
        self.oxy = randomize(self.OXY_min, self.OXY_max)


def get_background_settings():
    """

    :return: a settings dictionary containing all min and max parameters fitting for generic background tissue.
    """
    return get_settings(b_min=0.005, b_max=0.005, w_min=0.68, w_max=0.68)


def get_epidermis_settings():
    """

    :return: a settings dictionary containing all min and max parameters fitting for epidermis tissue.
    """
    return get_settings(b_min=0.01, b_max=0.01, w_min=0.68, w_max=0.68, m_max=0.5, m_min=0.1)


def get_dermis_settings():
    """

    :return: a settings dictionary containing all min and max parameters fitting for dermis tissue.
    """
    return get_settings(b_min=0.005, b_max=0.02, w_min=0.68, w_max=0.68)


def get_subcutaneous_fat_settings():
    """

    :return: a settings dictionary containing all min and max parameters fitting for subcutaneous fat tissue.
    """
    return get_settings(b_min=0.005, b_max=0.01, w_min=0.68, w_max=0.68, f_min=0.3, f_max=0.6)


def get_blood_settings():
    """

        :return: a settings dictionary containing all min and max parameters fitting for full blood.
        """
    return get_settings(b_min=1, b_max=1, w_min=1, w_max=1)


def get_settings(b_min=0.0, b_max=0.0, w_min=0.0, w_max=0.0, f_min=0.0, f_max=0.0,
                 m_min=0.0, m_max=0.0, oxy_min=0.0, oxy_max=1.0):
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
    return return_dict
