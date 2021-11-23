# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import operator
import traceback

from simpa.log import Logger
from simpa.utils import Settings, Tags

from simpa.utils.libraries.structure_library.BackgroundStructure import Background, \
    define_background_structure_settings
from simpa.utils.libraries.structure_library.CircularTubularStructure import CircularTubularStructure, \
    define_circular_tubular_structure_settings
from simpa.utils.libraries.structure_library.EllipticalTubularStructure import EllipticalTubularStructure, \
    define_elliptical_tubular_structure_settings
from simpa.utils.libraries.structure_library.HorizontalLayerStructure import HorizontalLayerStructure, \
    define_horizontal_layer_structure_settings
from simpa.utils.libraries.structure_library.ParallelepipedStructure import ParallelepipedStructure, \
    define_parallelepiped_structure_settings
from simpa.utils.libraries.structure_library.RectangularCuboidStructure import RectangularCuboidStructure, \
    define_rectangular_cuboid_structure_settings
from simpa.utils.libraries.structure_library.SphericalStructure import SphericalStructure, \
    define_spherical_structure_settings
from simpa.utils.libraries.structure_library.VesselStructure import VesselStructure, \
    define_vessel_structure_settings


class Structures:
    """
    TODO
    """
    def __init__(self, settings: Settings, volume_creator_settings: dict):
        """
        TODO
        """
        self.logger = Logger()
        self.structures = self.from_settings(settings, volume_creator_settings)
        self.sorted_structures = sorted(self.structures, key=operator.attrgetter('priority'), reverse=True)

    def from_settings(self, global_settings, volume_creator_settings):
        structures = list()
        if not Tags.STRUCTURES in volume_creator_settings:
            self.logger.warning("Did not find any structure definitions in the settings file!")
            return structures
        _structure_settings = volume_creator_settings[Tags.STRUCTURES]
        for struc_tag_name in _structure_settings:
            single_structure_settings = _structure_settings[struc_tag_name]
            try:
                structure_class = globals()[single_structure_settings[Tags.STRUCTURE_TYPE]]
                structure = structure_class(global_settings, single_structure_settings)
                structures.append(structure)
            except Exception as e:
                self.logger.critical("An exception has occurred while trying to parse " +
                                     str(single_structure_settings[Tags.STRUCTURE_TYPE]) +
                                     " from the dictionary.")
                self.logger.critical("The structure type was " + str(single_structure_settings[Tags.STRUCTURE_TYPE]))
                self.logger.critical(traceback.format_exc())
                self.logger.critical("trying to continue as normal...")
                raise e

        return structures