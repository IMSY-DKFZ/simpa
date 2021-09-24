"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

# First load everything without internal dependencies
from .tags import Tags
from .libraries.literature_values import MorphologicalTissueProperties
from .libraries.literature_values import StandardProperties
from .libraries.literature_values import OpticalTissueProperties
from .constants import SaveFilePaths, SegmentationClasses

# Then load classes and methods with an <b>increasing</b> amount of internal dependencies.
# If there are import errors in the tests, it is probably due to an incorrect
# initialization order
from .libraries.spectra_library import AbsorptionSpectrumLibrary
from .libraries.spectra_library import Spectrum
from .libraries.spectra_library import SPECTRAL_LIBRARY
from .libraries.spectra_library import view_absorption_spectra

from .libraries.molecule_library import Molecule, MolecularCompositionGenerator
from .libraries.molecule_library import MoleculeLibrary
from .libraries.molecule_library import MOLECULE_LIBRARY

from .libraries.tissue_library import TissueLibrary
from .libraries.tissue_library import TISSUE_LIBRARY

from .calculate import calculate_oxygenation
from .calculate import calculate_gruneisen_parameter_from_temperature
from .calculate import randomize_uniform

from .deformation_manager import create_deformation_settings
from .deformation_manager import get_functional_from_deformation_settings

from .settings import Settings

from .dict_path_manager import generate_dict_path
from .dict_path_manager import get_data_field_from_simpa_output

from .constants import EPS

from .path_manager import PathManager

if __name__ == "__main__":
    view_absorption_spectra()
