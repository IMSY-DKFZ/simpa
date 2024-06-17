# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import traceback
import torch
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


def priority_sorted_structures(settings: Settings, volume_creator_settings: dict):
    """
    A generator function to lazily construct structures in descending order of priority
    """
    logger = Logger()
    if not Tags.STRUCTURES in volume_creator_settings:
        logger.warning("Did not find any structure definitions in the settings file!")
        return
    sorted_structure_settings = sorted(
        [structure_setting for structure_setting in volume_creator_settings[Tags.STRUCTURES].values()],
        key=lambda s: s[Tags.PRIORITY] if Tags.PRIORITY in s else 0, reverse=True)
    for structure_setting in sorted_structure_settings:
        try:
            structure_class = globals()[structure_setting[Tags.STRUCTURE_TYPE]]
            yield structure_class(settings, structure_setting)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.critical("An exception has occurred while trying to parse " +
                            str(structure_setting[Tags.STRUCTURE_TYPE]) +
                            " from the dictionary.")
            logger.critical("The structure type was " + str(structure_setting[Tags.STRUCTURE_TYPE]))
            logger.critical(traceback.format_exc())
            logger.critical("trying to continue as normal...")
            raise e
