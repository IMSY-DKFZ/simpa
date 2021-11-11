# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class ParallelepipedStructure(GeometricalStructure):
    """
    Defines a parallelepiped which is defined by a start point and three edge vectors which originate from the start
    point. This structure currently does not implement partial volume effects.
    Example usage:

        # single_structure_settings initialization
        structure = Settings()

        structure[Tags.PRIORITY] = 9
        structure[Tags.STRUCTURE_START_MM] = [25, 25, 25]
        structure[Tags.STRUCTURE_FIRST_EDGE_MM] = [5, 1, 1]
        structure[Tags.STRUCTURE_SECOND_EDGE_MM] = [1, 5, 1]
        structure[Tags.STRUCTURE_THIRD_EDGE_MM] = [1, 1, 5]
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        structure[Tags.STRUCTURE_TYPE] = Tags.PARALLELEPIPED_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_FIRST_EDGE_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_SECOND_EDGE_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_THIRD_EDGE_MM]))
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_FIRST_EDGE_MM] = self.params[1]
        settings[Tags.STRUCTURE_SECOND_EDGE_MM] = self.params[2]
        settings[Tags.STRUCTURE_THIRD_EDGE_MM] = self.params[3]
        return settings

    def get_enclosed_indices(self):
        start_mm, x_edge_mm, y_edge_mm, z_edge_mm = self.params
        start_voxels = start_mm / self.voxel_spacing
        x_edge_voxels = np.array(x_edge_mm / self.voxel_spacing)
        y_edge_voxels = np.array(y_edge_mm / self.voxel_spacing)
        z_edge_voxels = np.array(z_edge_mm / self.voxel_spacing)

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

        volume_fractions = np.zeros(self.volume_dimensions_voxels)
        filled_mask = np.all(filled_mask_bool, axis=-1)

        volume_fractions[filled_mask] = 1

        return filled_mask, volume_fractions[filled_mask]


def define_parallelepiped_structure_settings(start_mm: list, edge_a_mm: list, edge_b_mm: list, edge_c_mm: list,
                                             molecular_composition: MolecularComposition, priority: int = 10,
                                             consider_partial_volume: bool = False,
                                             adhere_to_deformation: bool = False):
    """
    TODO
    """

    return {
        Tags.STRUCTURE_START_MM: start_mm,
        Tags.STRUCTURE_FIRST_EDGE_MM: edge_a_mm,
        Tags.STRUCTURE_SECOND_EDGE_MM: edge_b_mm,
        Tags.STRUCTURE_THIRD_EDGE_MM: edge_c_mm,
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.PARALLELEPIPED_STRUCTURE
    }