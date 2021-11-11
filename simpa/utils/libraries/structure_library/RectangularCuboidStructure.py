# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class RectangularCuboidStructure(GeometricalStructure):
    """
    Defines a rectangular cuboid (box) which is defined by a start point its extent along the x-, y-, and z-axis.
    This structure implements partial volume effects. The box can be set to adhere to a deformation defined by the
    simpa.utils.deformation_manager. The start point of the box will then be shifted along the z-axis
    accordingly.
    Example usage:

        # single_structure_settings initialization
        structure = Settings()

        structure[Tags.PRIORITY] = 9
        structure[Tags.STRUCTURE_START_MM] = [25, 25, 25]
        structure[Tags.STRUCTURE_X_EXTENT_MM] = 40
        structure[Tags.STRUCTURE_Y_EXTENT_MM] = 50
        structure[Tags.STRUCTURE_Z_EXTENT_MM] = 60
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_X_EXTENT_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_Y_EXTENT_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_Z_EXTENT_MM]),
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_X_EXTENT_MM] = self.params[1]
        settings[Tags.STRUCTURE_Y_EXTENT_MM] = self.params[2]
        settings[Tags.STRUCTURE_Z_EXTENT_MM] = self.params[3]
        settings[Tags.CONSIDER_PARTIAL_VOLUME] = self.params[4]
        return settings

    def get_enclosed_indices(self):
        start_mm, x_edge_mm, y_edge_mm, z_edge_mm, partial_volume = self.params
        start_voxels = start_mm / self.voxel_spacing
        x_edge_voxels = np.array([x_edge_mm / self.voxel_spacing, 0, 0])
        y_edge_voxels = np.array([0, y_edge_mm / self.voxel_spacing, 0])
        z_edge_voxels = np.array([0, 0, z_edge_mm / self.voxel_spacing])

        x, y, z = np.meshgrid(np.arange(self.volume_dimensions_voxels[0]),
                              np.arange(self.volume_dimensions_voxels[1]),
                              np.arange(self.volume_dimensions_voxels[2]),
                              indexing='ij')

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start_voxels)

        matrix = np.array([x_edge_voxels, y_edge_voxels, z_edge_voxels])

        inverse_matrix = np.linalg.inv(matrix)

        result = np.matmul(target_vector, inverse_matrix)

        norm_vector = np.array([1/np.linalg.norm(x_edge_voxels),
                                1/np.linalg.norm(y_edge_voxels),
                                1/np.linalg.norm(z_edge_voxels)])

        filled_mask_bool = (0 <= result) & (result <= 1 - norm_vector)
        border_bool = (0 - norm_vector < result) & (result <= 1)

        volume_fractions = np.zeros(self.volume_dimensions_voxels)
        filled_mask = np.all(filled_mask_bool, axis=-1)

        border_mask = np.all(border_bool, axis=-1)

        border_mask = np.logical_xor(border_mask, filled_mask)

        edge_values = result[border_mask]

        fraction_values = np.matmul(edge_values, matrix)

        larger_fraction_values = (x_edge_voxels + y_edge_voxels + z_edge_voxels) - fraction_values

        small_bool = fraction_values > 0
        large_bool = larger_fraction_values >= 1

        fraction_values[small_bool & large_bool] = 0
        fraction_values[fraction_values <= 0] = 1 + fraction_values[fraction_values <= 0]
        fraction_values[larger_fraction_values < 1] = larger_fraction_values[larger_fraction_values < 1]

        fraction_values = np.abs(np.prod(fraction_values, axis=-1))

        volume_fractions[filled_mask] = 1
        volume_fractions[border_mask] = fraction_values

        if partial_volume:
            mask = filled_mask | border_mask
        else:
            mask = filled_mask

        return mask, volume_fractions[mask]


def define_rectangular_cuboid_structure_settings(start_mm: list, extent_mm: (int, list),
                                                 molecular_composition: MolecularComposition, priority: int = 10,
                                                 consider_partial_volume: bool = False,
                                                 adhere_to_deformation: bool = False):
    """
    TODO
    """
    if isinstance(extent_mm, int):
        extent_mm = [extent_mm, extent_mm, extent_mm]

    return {
        Tags.STRUCTURE_START_MM: start_mm,
        Tags.STRUCTURE_X_EXTENT_MM: extent_mm[0],
        Tags.STRUCTURE_Y_EXTENT_MM: extent_mm[1],
        Tags.STRUCTURE_Z_EXTENT_MM: extent_mm[2],
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.RECTANGULAR_CUBOID_STRUCTURE
    }