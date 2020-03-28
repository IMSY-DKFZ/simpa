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

from utils import AbsorptionSpectrum


class Chromophore(object):

    def __init__(self, spectrum: AbsorptionSpectrum,
                 volume_fraction: float,
                 musp500: float, f_ray: float, b_mie: float,
                 anisotropy: float):
        """
        :param spectrum: AbsorptionSpectrum
        :param volume_fraction: float
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
