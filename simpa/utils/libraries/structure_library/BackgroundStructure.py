# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.utils import Settings, Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class Background(GeometricalStructure):
    """
    Defines a background that fills the whole simulation volume. It is always given the priority of 0 so that other
    structures can overwrite it when necessary.

    Example usage::

        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.1, 100.0, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    """

    def get_enclosed_indices(self):
        array = np.ones((self.volume_dimensions_voxels[0],
                         self.volume_dimensions_voxels[1],
                         self.volume_dimensions_voxels[2]))
        return array == 1, 1

    def get_params_from_settings(self, single_structure_settings):
        return None

    def __init__(self, global_settings: Settings, background_settings: Settings = None):

        if background_settings is not None:
            background_settings[Tags.PRIORITY] = 0
            background_settings[Tags.CONSIDER_PARTIAL_VOLUME] = False
            super().__init__(global_settings, background_settings)
        else:
            super().__init__(global_settings)
            self.priority = 0
            self.partial_volume = True

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        return settings_dict


def define_background_structure_settings(molecular_composition: MolecularComposition):
    """
    TODO
    """
    return {
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.STRUCTURE_TYPE: Tags.BACKGROUND
    }