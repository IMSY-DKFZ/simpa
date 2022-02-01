# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.log.file_logger import Logger
from simpa.utils import Spectrum, Molecule, Settings
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.core.device_digital_twins import *

import inspect
import sys

members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
SERIALIZATION_MAP = dict()
for member in members:
    SERIALIZATION_MAP[member[0]] = member[1]
