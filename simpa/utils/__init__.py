# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

# First load everything without internal dependencies
from .tags import Tags
from .libraries.literature_values import MorphologicalTissueProperties
from .libraries.literature_values import StandardProperties
from .libraries.literature_values import OpticalTissueProperties
from .constants import SegmentationClasses

# Then load classes and methods with an <b>increasing</b> amount of internal dependencies.
# If there are import errors in the tests, it is probably due to an incorrect
# initialization order
from .libraries.spectrum_library import AbsorptionSpectrumLibrary
from .libraries.spectrum_library import Spectrum
from .libraries.spectrum_library import view_saved_spectra
from .libraries.spectrum_library import AnisotropySpectrumLibrary
from .libraries.spectrum_library import ScatteringSpectrumLibrary
from .libraries.spectrum_library import get_simpa_internal_absorption_spectra_by_names

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

from .libraries.structure_library.BackgroundStructure import Background, define_background_structure_settings
from .libraries.structure_library.CircularTubularStructure import CircularTubularStructure, \
    define_circular_tubular_structure_settings
from .libraries.structure_library.EllipticalTubularStructure import EllipticalTubularStructure, \
    define_elliptical_tubular_structure_settings
from .libraries.structure_library.HorizontalLayerStructure import HorizontalLayerStructure, \
    define_horizontal_layer_structure_settings
from .libraries.structure_library.ParallelepipedStructure import ParallelepipedStructure, \
    define_parallelepiped_structure_settings
from .libraries.structure_library.RectangularCuboidStructure import RectangularCuboidStructure, \
    define_rectangular_cuboid_structure_settings
from .libraries.structure_library.SphericalStructure import SphericalStructure, define_spherical_structure_settings
from .libraries.structure_library.VesselStructure import VesselStructure, define_vessel_structure_settings

if __name__ == "__main__":
    view_saved_spectra()
