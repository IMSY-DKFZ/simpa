"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Spectrum, Molecule, Settings


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


SERIALIZATION_MAP = {
    "Settings": Settings.deserialize
}
