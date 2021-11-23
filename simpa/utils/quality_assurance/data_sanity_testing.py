# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import inspect


def assert_equal_shapes(numpy_arrays: list):
    """
    This method takes a list of n-dimensional numpy arrays and raises an AssertionError if the sizes of all arrays do
    not match.

    :param numpy_arrays: a list of np.ndarray
    :raises AssertionError: if there is a mismatch between any of the volume dimensions.
    """

    if len(numpy_arrays) < 2:
        return

    shapes = np.asarray([np.shape(_arr) for _arr in numpy_arrays]).astype(float)
    mean = np.mean(shapes, axis=0)
    for i in range(len(shapes)):
        shapes[i, :] = shapes[i, :] - mean

    if not np.sum(np.abs(shapes)) <= 1e-5:
        raise AssertionError("The given volumes did not all have the same"
                             " dimensions. Please double check the simulation"
                             f" parameters. Called from {inspect.stack()[1].function}")


def assert_array_well_defined(array: np.ndarray, assume_non_negativity:bool = False,
                              assume_positivity=False, array_name:str=None):
    """
    This method tests if all entries of the given array are well-defined (i.e. not np.inf, np.nan, or None).
    The method can be parametrised to be more strict.

    :param array: The input np.ndarray
    :param assume_non_negativity: bool (default: False). If true, all values must be greater than or equal to 0.
    :param assume_positivity: bool (default: False). If true, all values must be greater than 0.
    :param array_name: a string that gives more information in case of an error.
    :raises AssertionError: if there are any unexpected values in the given array.
    """

    if array_name is None:
        array_name = "'Not specified'"
    caller = inspect.stack()[1]
    stack_string = f" \n\tArray Name: {array_name} \n\tCaller: {caller.filename}" \
                   f" \n\tline: {caller.lineno} \n\tcode: {caller.code_context}"

    if np.isinf(array).any() or np.isneginf(array).any():
        raise AssertionError("The given array contained values that were inf or -inf."
                             f" Info: {stack_string}.")

    if np.isnan(array).any():
        raise AssertionError("The given array contained values that were nan."
                             f" Info: {stack_string}.")

    if assume_positivity and (array <= 0).any():
        raise AssertionError("The given array contained values that were not positive."
                             f" Info: {stack_string}.")

    if assume_non_negativity and (array < 0).any():
        raise AssertionError("The given array contained values that were negative."
                             f" Info: {stack_string}.")
