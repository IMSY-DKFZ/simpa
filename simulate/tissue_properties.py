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

from simulate import Tags
from utils import Chromophore


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
