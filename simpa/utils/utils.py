# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from typing import Union

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
