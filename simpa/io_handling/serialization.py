"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from json import JSONEncoder
from simpa.utils import Spectrum, Molecule
import numpy as np


class SIMPASerializer(object):
    """
    TODO
    """

    def serialize(self, _object: object):
        """

        """
        if isinstance(_object, Molecule):
            return _object.__dict__

        if isinstance(_object, Spectrum):
            return _object.__dict__

        return _object


class SIMPAJSONSerializer(JSONEncoder):
    """
    TODO
    """

    def __init__(self):
        """
        TODO
        """
        super().__init__()
        self._serializer = SIMPASerializer()

    def default(self, _object: object):
        """
        TODO
        """

        serialized_object = self._serializer.serialize(_object)

        if serialized_object is not None:
            return serialized_object

        if isinstance(_object, np.ndarray):
            return list(_object)

        if isinstance(_object, (np.int, np.int16, np.int32, np.int64)):
            return int(_object)

        if isinstance(_object, (np.float, np.float16, np.float32, np.float64)):
            return float(_object)

        return super().default(_object)
