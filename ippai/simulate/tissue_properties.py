import numpy as np
from ippai.simulate.utils import randomize


class TissueProperties(object):

    def __init__(self, settings, keyword):
        self.KEY_B_MIN = "B_min"
        self.KEY_B_MAX = "B_max"
        self.KEY_W_MAX = "w_max"
        self.KEY_W_MIN = "w_min"
        self.KEY_F_MAX = "f_max"
        self.KEY_F_MIN = "f_min"
        self.KEY_M_MAX = "m_max"
        self.KEY_M_MIN = "m_min"
        self.KEY_OXY_MAX = "oxy_max"
        self.KEY_OXY_MIN = "oxy_min"

        self.KEYWORDS = [self.KEY_B_MIN, self.KEY_B_MAX, self.KEY_W_MAX, self.KEY_W_MIN, self.KEY_F_MAX, self.KEY_F_MIN,
                         self.KEY_M_MAX, self.KEY_M_MIN, self.KEY_OXY_MAX, self.KEY_OXY_MIN]
        if settings is not None:
            self.ensure_valid_settings_file(settings, keyword)
            self.B_min = settings[keyword][self.KEY_B_MIN]
            self.B_max = settings[keyword][self.KEY_B_MAX]
            self.W_min = settings[keyword][self.KEY_W_MIN]
            self.W_max = settings[keyword][self.KEY_W_MAX]
            self.F_min = settings[keyword][self.KEY_F_MIN]
            self.F_max = settings[keyword][self.KEY_F_MAX]
            self.M_min = settings[keyword][self.KEY_M_MIN]
            self.M_max = settings[keyword][self.KEY_M_MAX]
            self.OXY_min = settings[keyword][self.KEY_OXY_MIN]
            self.OXY_max = settings[keyword][self.KEY_OXY_MAX]

        self.absorption_data = np.load("../data/absorption.npz")

    def ensure_valid_settings_file(self, settings, keyword):
        if settings[keyword] is None:
            raise LookupError("Tissue settings for "+str(keyword)+" are not contained in settings file")

        for _keyword in self.KEYWORDS:
            if settings[keyword][_keyword] is None:
                raise LookupError("Tissue settings for " + str(_keyword) +
                                  " are not contained in  " + str(keyword) + " settings")

    def instantiate_for_wavelength(self, wavelength):
        wavelength = wavelength-700
        if wavelength < 0 or wavelength > 251:
            raise AssertionError("Wavelengths only supported between 700 nm and 950 nm")

        bvf = randomize(self.B_min, self.B_max)
        wvf = randomize(self.W_min, self.W_max)
        fvf = randomize(self.F_min, self.F_max)
        mvf = randomize(self.M_min, self.M_max)
        oxy = randomize(self.OXY_min, self.OXY_max)

        absorption = wvf * self.absorption_data['water'][wavelength] + \
                     fvf * self.absorption_data['fat'][wavelength] + \
                     mvf * self.absorption_data['melanin'][wavelength] + \
                     bvf * oxy * self.absorption_data['hbo2'][wavelength] + \
                     bvf * (1-oxy) * self.absorption_data['hb'][wavelength]
        scattering = 100  # FIXME: Include scattering term
        anisotropy = 0.9  # FIXME: Include anisotropy term

        return [absorption, scattering, anisotropy]

    def get_background_settings(self):
        return_dict = dict()
        return_dict[self.KEY_B_MIN] = 0.01
        return_dict[self.KEY_B_MAX] = 0.01
        return_dict[self.KEY_W_MIN] = 0.68
        return_dict[self.KEY_W_MAX] = 0.68
        return_dict[self.KEY_F_MIN] = 0
        return_dict[self.KEY_F_MAX] = 0
        return_dict[self.KEY_M_MIN] = 0
        return_dict[self.KEY_M_MAX] = 0
        return_dict[self.KEY_OXY_MIN] = 0
        return_dict[self.KEY_OXY_MAX] = 1
        return return_dict
