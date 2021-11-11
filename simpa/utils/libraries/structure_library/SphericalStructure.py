# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class SphericalStructure(GeometricalStructure):
    """
    Defines a sphere which is defined by a start point and a radius. This structure implements
    partial volume effects. The sphere can be set to adhere to a deformation defined by the
    simpa.utils.deformation_manager. The start point of the sphere will then be shifted along the z-axis
    accordingly.
    Example usage:

        # single_structure_settings initialization
        structure = Settings()

        structure[Tags.PRIORITY] = 9
        structure[Tags.STRUCTURE_START_MM] = [50, 50, 50]
        structure[Tags.STRUCTURE_RADIUS_MM] = 10
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.SPHERICAL_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_RADIUS_MM]),
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_RADIUS_MM] = self.params[1]
        return settings

    def get_enclosed_indices(self):
        start_mm, radius_mm, partial_volume = self.params
        start_voxels = start_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing
        x, y, z = np.meshgrid(np.arange(self.volume_dimensions_voxels[0]),
                              np.arange(self.volume_dimensions_voxels[1]),
                              np.arange(self.volume_dimensions_voxels[2]),
                              indexing='ij')

        x = x + 0.5
        y = y + 0.5
        z = z + 0.5

        if partial_volume:
            radius_margin = 0.5
        else:
            radius_margin = 0.7071

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start_voxels)
        target_radius = np.linalg.norm(target_vector, axis=-1)

        volume_fractions = np.zeros(self.volume_dimensions_voxels)
        filled_mask = target_radius <= radius_voxels - 1 + radius_margin
        border_mask = (target_radius > radius_voxels - 1 + radius_margin) & \
                      (target_radius < radius_voxels + 2 * radius_margin)

        volume_fractions[filled_mask] = 1
        volume_fractions[border_mask] = 1 - (target_radius - (radius_voxels-radius_margin))[border_mask]
        volume_fractions[volume_fractions < 0] = 0

        if partial_volume:
            mask = filled_mask | border_mask
        else:
            mask = filled_mask

        return mask, volume_fractions[mask]


def define_spherical_structure_settings(start_mm: list, molecular_composition: MolecularComposition,
                                        radius_mm: float = 1, priority: int = 10,
                                        consider_partial_volume: bool = False,
                                        adhere_to_deformation: bool = False):
    """
    TODO
    """
    return {
        Tags.STRUCTURE_START_MM: start_mm,
        Tags.STRUCTURE_RADIUS_MM: radius_mm,
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.SPHERICAL_STRUCTURE
    }