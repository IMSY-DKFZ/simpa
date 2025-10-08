# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT


from typing import Union, List, Dict, Optional, Sized
import numpy as np
import torch
from scipy.interpolate import interp1d


def extract_hemoglobin_fractions(molecule_list: List) -> Dict[str, float]:
    """
    Extract hemoglobin volume fractions from a list of molecules.

    :param molecule_list: List of molecules with their spectrum information and volume fractions.
    :return: A dictionary with hemoglobin types as keys and their volume fractions as values.
    """

    # Put 0.0 as default value for both hemoglobin types in case they are not present in the molecule list.
    hemoglobin = {
        "Deoxyhemoglobin": 0.0,
        "Oxyhemoglobin": 0.0
    }

    for molecule in molecule_list:
        spectrum_name = molecule.absorption_spectrum.spectrum_name
        if spectrum_name in hemoglobin:
            hemoglobin[spectrum_name] = molecule.volume_fraction

    return hemoglobin


def calculate_oxygenation(molecule_list: List) -> Optional[float]:
    """
    Calculate the oxygenation level based on the volume fractions of deoxyhemoglobin and oxyhemoglobin.

    :param molecule_list: List of molecules with their spectrum information and volume fractions.
    :return: An oxygenation value between 0 and 1 if possible, or None if not computable.
    """
    hemoglobin = extract_hemoglobin_fractions(molecule_list)
    hb, hbO2 = hemoglobin["Deoxyhemoglobin"], hemoglobin["Oxyhemoglobin"]

    total = hb + hbO2

    # Avoid division by zero. If none of the hemoglobin types are present, the oxygenation level is not computable.
    if isinstance(hb, torch.Tensor) or isinstance(hbO2, torch.Tensor):
        return torch.where(total < 1e-10, 0, hbO2 / total)

    else:
        if total < 1e-10:
            return None
        else:
            return hbO2 / total


def calculate_bvf(molecule_list: List) -> Union[float, int]:
    """
    Calculate the blood volume fraction based on the volume fractions of deoxyhemoglobin and oxyhemoglobin.

    :param molecule_list: List of molecules with their spectrum information and volume fractions.
    :return: The blood volume fraction value between 0 and 1, or 0, if oxy and deoxy not present.
    """
    hemoglobin = extract_hemoglobin_fractions(molecule_list)
    hb, hbO2 = hemoglobin["Deoxyhemoglobin"], hemoglobin["Oxyhemoglobin"]
    # We can use the sum of hb and hb02 to compute blood volume fraction as the volume fraction of all molecules is 1.
    return hb + hbO2


def create_spline_for_range(xmin_mm: Union[float, int] = 0, xmax_mm: Union[float, int] = 10,
                            maximum_y_elevation_mm: Union[float, int] = 1, spacing: Union[float, int] = 0.1) -> tuple:
    """
    Creates a functional that simulates distortion along the y position
    between the minimum and maximum x positions. The elevation can never be
    smaller than 0 or bigger than maximum_y_elevation_mm.

    :param xmin_mm: the minimum x axis value the return functional is defined in
    :param xmax_mm: the maximum x axis value the return functional is defined in
    :param maximum_y_elevation_mm: the maximum y axis value the return functional will yield
    :param spacing: the voxel spacing in the simulation
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


def spline_evaluator2d_voxel(x: int, y: int, spline: Union[list, np.ndarray], offset_voxel: Union[float, int],
                             thickness_voxel: int) -> bool:
    """
    Evaluate whether a given point (x, y) lies within the thickness bounds around a spline curve.

    This function checks if the y-coordinate of a point lies within a vertical range defined
    around a spline curve at a specific x-coordinate. The range is determined by the spline elevation,
    an offset, and a thickness.

    :param x: The x-coordinate of the point to evaluate.
    :param y: The y-coordinate of the point to evaluate.
    :param spline: A 1D array or list representing the spline curve elevations at each x-coordinate.
    :param offset_voxel: The offset to be added to the spline elevation to define the starting y-coordinate of the range.
    :param thickness_voxel: The vertical thickness of the range around the spline.
    :return: True if the point (x, y) lies within the range around the spline, False otherwise.
    """
    elevation = spline[x]
    y_value = round_x5_away_from_zero(elevation + offset_voxel)
    if y_value <= y < thickness_voxel + y_value:
        return True
    else:
        return False


def calculate_gruneisen_parameter_from_temperature(temperature_in_celcius: Union[float, int]) -> Union[float, int]:
    """
    This function returns the dimensionless gruneisen parameter based on a heuristic formula that
    was determined experimentally::

        @book{wang2012biomedical,
            title={Biomedical optics: principles and imaging},
            author={Wang, Lihong V and Wu, Hsin-i},
            year={2012},
            publisher={John Wiley & Sons}
        }

    :param temperature_in_celcius: the temperature in degrees celcius
    :return: a floating point number, if temperature_in_celcius is a number or a float array, if temperature_in_celcius
        is an array

    """
    return 0.0043 + 0.0053 * temperature_in_celcius


def randomize_uniform(min_value: float, max_value: float) -> Union[float, int]:
    """
    returns a uniformly drawn random number in [min_value, max_value[

    :param min_value: minimum value
    :param max_value: maximum value
    :return: random number in [min_value, max_value[

    """
    return (np.random.random() * (max_value-min_value)) + min_value


def rotation_x(theta: Union[float, int]) -> torch.Tensor:
    """
    Rotation matrix around the x-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta)],
                         [0, torch.sin(theta), torch.cos(theta)]])


def rotation_y(theta: Union[float, int]) -> torch.Tensor:
    """
    Rotation matrix around the y-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]])


def rotation_z(theta: Union[float, int]) -> torch.Tensor:
    """
    Rotation matrix around the z-axis with angle theta.

    :param theta: Angle through which the matrix is supposed to rotate.
    :return: rotation matrix
    """
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 1]])


def rotation(angles: Union[list, np.ndarray]) -> torch.Tensor:
    """
    Rotation matrix around the x-, y-, and z-axis with angles [theta_x, theta_y, theta_z].

    :param angles: Angles through which the matrix is supposed to rotate in the form of [theta_x, theta_y, theta_z].
    :return: rotation matrix
    """
    return rotation_x(angles[0]) * rotation_y(angles[1]) * rotation_z(angles[2])


def rotation_matrix_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns the rotation matrix from a to b

    :param a: 3D vector to rotate
    :param b: 3D target vector
    :return: rotation matrix
    """
    a_norm, b_norm = (a / np.linalg.norm(a)).reshape(3), (b / np.linalg.norm(b)).reshape(3)
    cross_product = np.cross(a_norm, b_norm)
    if np.abs(cross_product.all()) < 1e-10:
        return np.zeros([3, 3])
    dot_product = np.dot(a_norm, b_norm)
    s = np.linalg.norm(cross_product)
    mat = np.array([[0, -cross_product[2], cross_product[1]],
                    [cross_product[2], 0, -cross_product[0]],
                    [-cross_product[1], cross_product[0], 0]])
    rotation_matrix = np.eye(3) + mat + mat.dot(mat) * ((1 - dot_product) / (s ** 2))
    return rotation_matrix


def min_max_normalization(data: np.ndarray = None) -> np.ndarray:
    """
    Normalizes the given data by applying min max normalization.
    The resulting array has values between 0 and 1 inclusive.

    :param data: (numpy array) data to be normalized
    :return: (numpy array) normalized array
    """

    if data is None:
        raise AttributeError("Data must not be none in order to normalize it.")

    _min = np.min(data)
    _max = np.max(data)
    output = (data - _min) / (_max - _min)

    return output


def positive_gauss(mean, std) -> float:
    """
    Generates a non-negative random sample (scalar) from a normal (Gaussian) distribution.

    :param mean : float defining the mean ("centre") of the distribution. 
    :param std: float defining the standard deviation (spread or "width") of the distribution. Must be non-negative.
    :return: non-negative random sample from a normal (Gaussian) distribution.
    """
    random_value = np.random.normal(mean, std)
    if random_value <= 0:
        return positive_gauss(mean, std)
    else:
        return random_value


def are_equal(obj1: Union[list, tuple, np.ndarray, object], obj2: Union[list, tuple, np.ndarray, object]) -> bool:
    """Compare if two objects are equal. For lists, tuples and arrays, all entries need to be equal to return True.

    :param obj1: The first object to compare. Can be of any type, but typically a list, numpy array, or scalar.
    :type obj1: Union[list, tuple, np.ndarray, object]
    :param obj2: The second object to compare. Can be of any type, but typically a list, numpy array, or scalar.
    :type obj2: Union[list, tuple, np.ndarray, object]
    :return: True if the objects are equal, False otherwise. For lists and numpy arrays, returns True only if all
        corresponding elements are equal.
    :rtype: bool
    """
    # Check if one object is numpy array or list
    if isinstance(obj1, (list, np.ndarray, tuple)) or isinstance(obj2, (list, np.ndarray, tuple)):
        return np.array_equal(obj1, obj2)
    # For other types, use standard equality check which also works for lists
    else:
        return obj1 == obj2


def round_x5_away_from_zero(x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Round a number away from zero. The np.round function rounds x.5 to the nearest even number, which is not always the
    desired behavior. This function always rounds x.5 away from zero. For example, x.5 will be rounded to 1, and -x.5
    will be rounded to -1. All other numbers are rounded to the nearest integer.
    :param x: input number or array of numbers
    :return: rounded number or array of numbers
    :rtype: int or np.ndarray of int
    """

    def round_single_value(value):
        # If the value is positive, add 0.5 and use floor to round away from zero
        # If the value is negative, subtract 0.5 and use ceil to round away from zero
        return int(np.floor(value + 0.5)) if value > 0 else int(np.ceil(value - 0.5))

    if isinstance(x, (np.ndarray, list, tuple)):
        # Apply rounding function to each element in the array
        return np.array([round_x5_away_from_zero(val) for val in x], dtype=int)
    else:
        # Apply rounding to a single value
        return round_single_value(x)
