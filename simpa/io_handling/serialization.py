# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
# SPDX-License-Identifier: MIT

from simpa.utils import Spectrum, Molecule, Settings
from simpa.utils.libraries.molecule_library import MolecularComposition

import inspect
import sys

members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
SERIALIZATION_MAP = dict()
for member in members:
    SERIALIZATION_MAP[member[0]] = member[1]
