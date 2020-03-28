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
from utils import SPECTRAL_LIBRARY

def calculate_oxygenation(tissue_properties):
    """
    :return: an oxygenation value between 0 and 1 if possible, or None, if not computable.
    """

    hb = None
    hbO2 = None

    for chromophore in tissue_properties.chromophores:
        if chromophore.spectrum == SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN:
            hb = chromophore.volume_fraction
        if chromophore.spectrum == SPECTRAL_LIBRARY.OXYHEMOGLOBIN:
            hbO2 = chromophore.volume_fraction

    if hb is None and hbO2 is None:
        return None

    if hb is None:
        hb = 0
    elif hbO2 is None:
        hbO2 = 0

    if hb + hbO2 < 1e-10:  # negative values are not allowed and division by (approx) zero
        return None        # will lead to negative side effects.

    return hbO2 / (hb + hbO2)
