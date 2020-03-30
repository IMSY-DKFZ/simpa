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
import numpy as np
from scipy.ndimage import gaussian_filter

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


def randomize(lower_bound, upper_bound, distribution='uniform', size=(1,), gauss_kernel_size=0):
    """
    This function creates a randomly distributed variable between lower_bound and upper_bound.

    The number is drawn from obe of the following distributions:
        - 'uniform'
        - 'normal'

    The default behaviour is to draw from a uniform distribution.

    If a gauss_kernel_size > 0 is given, then a gaussian filter is applied to the random distribution.

    :param lower_bound: the lower bound of the random distribution
    :param upper_bound: the upper bound of the random distribution
    :param distribution:  string argument: 'uniform', 'normal'
    :param size: target shape of the output array
    :param gauss_kernel_size: if >0 then a filter is applied to the distribution with filter size gauss_kernel_size.

    :return: a size shaped array of random numbers
    """
    mean = (upper_bound + lower_bound) / 2
    spread = (upper_bound - lower_bound)

    if distribution == 'normal':
        result = np.random.normal(mean, (spread / 2), size=size)
    elif distribution == 'uniform':
        result = (np.random.random(size=size) * spread) + lower_bound
    else:
        result = (np.random.random(size=size) * spread) + lower_bound

    if gauss_kernel_size > 0:
        result = gaussian_filter(result, gauss_kernel_size)
        # After filtering, the noise needs to be rescaled to be in the correct range again (esp. wrt the spread)
        result = ((result - np.mean(result)) / np.std(result)) * \
                 (spread / 2) + mean

    return result


def calculate_gruneisen_parameter_from_temperature(temperature_in_celcius):
    """
    This function returns the dimensionless gruneisen parameter based on a heuristic formula that
    was determined experimentally:

    @book{wang2012biomedical,
        title={Biomedical optics: principles and imaging},
        author={Wang, Lihong V and Wu, Hsin-i},
        year={2012},
        publisher={John Wiley \& Sons}
    }

    :param temperature_in_celcius: the temperature in degrees celcius
    :return: a floating point number, if temperature_in_celcius is a number or a
    float array, if temperature_in_celcius is an array
    """
    return 0.0043 + 0.0053 * temperature_in_celcius


def randomize_uniform(min_value: float, max_value: float):
    """
    returns a uniformly drawn random number in [min_value, max_value[

    :param min_value: minimum value
    :param max_value: maximum value
    :return: random number in [min_value, max_value[
    """
    return (np.random.random() * (max_value-min_value)) + min_value