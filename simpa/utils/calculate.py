# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
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

from simpa.utils.libraries.spectra_library import SPECTRAL_LIBRARY
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def calculate_oxygenation(molecule_list):
    """
    :return: an oxygenation value between 0 and 1 if possible, or None, if not computable.
    """
    hb = None
    hbO2 = None

    for molecule in molecule_list:
        if molecule.spectrum.spectrum_name == SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN.spectrum_name:
            hb = molecule.volume_fraction
        if molecule.spectrum.spectrum_name == SPECTRAL_LIBRARY.OXYHEMOGLOBIN.spectrum_name:
            hbO2 = molecule.volume_fraction

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


def create_spline_for_range(xmin_mm=0, xmax_mm=10, maximum_y_elevation_mm=1, spacing=0.1):
    """
    Creates a functional that simulates distortion along the y position
    between the minimum and maximum x positions. The elevation can never be
    smaller than 0 or bigger than maximum_y_elevation_mm.

    @param xmin_mm: the minimum x axis value the return functional is defined in
    @param xmax_mm: the maximum x axis value the return functional is defined in
    @param maximum_y_elevation_mm: the maximum y axis value the return functional will yield

    @return: a functional that describes a distortion field along the y axis

    """
    # Convert units from mm spacing to voxel spacing.
    xmax_voxels = xmax_mm / spacing
    maximum_y_elevation_mm = -maximum_y_elevation_mm

    # Create initial guesses left and right position
    left_boundary = np.random.random() * maximum_y_elevation_mm
    right_boundary = np.random.random() * maximum_y_elevation_mm

    # Define the number of division knots
    divisions = np.random.randint(1, 5)
    order = divisions
    if order > 3:
        order = 3

    # Create x and y value pairs that should be fit by the spline (needs to be division knots + 2)
    locations = np.linspace(xmin_mm, xmax_mm, divisions + 1)
    constraints = np.linspace(left_boundary, right_boundary, divisions + 1)

    # Add random permutations to the y-axis of the division knots
    for i in range(0, divisions + 1):
        scaling_value = np.sqrt(2 - ((i - (divisions / 2)) / (divisions / 2)) ** 2)

        constraints[i] = np.random.normal(scaling_value, 0.2) * constraints[i]
        if constraints[i] < maximum_y_elevation_mm:
            constraints[i] = maximum_y_elevation_mm
        if constraints[i] > 0:
            constraints[i] = 0

    constraints = constraints - np.max(constraints)

    spline = interp1d(locations, constraints, order)

    max_el = np.min(spline(np.arange(0, int(round(xmax_voxels)), 1) * spacing))

    return spline, max_el


def spline_evaluator2d_voxel(x, y, spline, offset_voxel, thickness_voxel):
    elevation = spline[x]
    y_value = np.round(elevation + offset_voxel)
    if y_value <= y < thickness_voxel + y_value:
        return True
    else:
        return False


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