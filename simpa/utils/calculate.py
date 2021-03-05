# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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


def create_spline_for_range(xmin_mm=0, xmax_mm=10, maximum_y_elevation_mm=1, spacing=0.1):
    """
    Creates a functional that simulates distortion along the y position
    between the minimum and maximum x positions. The elevation can never be
    smaller than 0 or bigger than maximum_y_elevation_mm.

    :param xmin_mm: the minimum x axis value the return functional is defined in
    :param xmax_mm: the maximum x axis value the return functional is defined in
    :param maximum_y_elevation_mm: the maximum y axis value the return functional will yield
    :return: a functional that describes a distortion field along the y axis

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
    was determined experimentally::

        @book{wang2012biomedical,
            title={Biomedical optics: principles and imaging},
            author={Wang, Lihong V and Wu, Hsin-i},
            year={2012},
            publisher={John Wiley \& Sons}
        }

    :param temperature_in_celcius: the temperature in degrees celcius
    :return: a floating point number, if temperature_in_celcius is a number or a float array, if temperature_in_celcius is an array

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


def rotation_x(theta):
    """
    Rotation matrix around the x-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])


def rotation_y(theta):
    """
    Rotation matrix around the y-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])


def rotation_z(theta):
    """
    Rotation matrix around the z-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])


def rotation(angles):
    """
    Rotation matrix around the x-, y-, and z-axis with angles [theta_x, theta_y, theta_z].

    :param angles: Angles through which the matrix is supposed to rotate in the form of [theta_x, theta_y, theta_z].
    :return: rotation matrix
    """
    return rotation_x(angles[0]) * rotation_y(angles[1]) * rotation_z(angles[2])
