# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from typing import Union
import torch

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
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_X_EXTENT_MM],
                  single_structure_settings[Tags.STRUCTURE_Y_EXTENT_MM],
                  single_structure_settings[Tags.STRUCTURE_Z_EXTENT_MM],
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
        start_mm = torch.tensor(start_mm, dtype=torch.float).to(self.torch_device)
        x_edge_mm = torch.tensor(x_edge_mm, dtype=torch.float).to(self.torch_device)
        y_edge_mm = torch.tensor(y_edge_mm, dtype=torch.float).to(self.torch_device)
        z_edge_mm = torch.tensor(z_edge_mm, dtype=torch.float).to(self.torch_device)

        start_voxels = start_mm / self.voxel_spacing
        x_edge_voxels = torch.tensor([x_edge_mm / self.voxel_spacing, 0, 0]).to(self.torch_device)
        y_edge_voxels = torch.tensor([0, y_edge_mm / self.voxel_spacing, 0]).to(self.torch_device)
        z_edge_voxels = torch.tensor([0, 0, z_edge_mm / self.voxel_spacing]).to(self.torch_device)

        x, y, z = torch.meshgrid(torch.arange(self.volume_dimensions_voxels[0]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[1]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[2]).to(self.torch_device),
                                 indexing='ij')

        target_vector = torch.subtract(torch.stack([x, y, z], axis=-1), start_voxels)

        matrix = torch.stack([x_edge_voxels, y_edge_voxels, z_edge_voxels])

        inverse_matrix = torch.linalg.inv(matrix)

        result = torch.matmul(target_vector, inverse_matrix)

        norm_vector = torch.tensor([1/torch.linalg.norm(x_edge_voxels),
                                    1/torch.linalg.norm(y_edge_voxels),
                                    1/torch.linalg.norm(z_edge_voxels)]).to(self.torch_device)

        filled_mask_bool = (0 <= result) & (result <= 1 - norm_vector)
        border_bool = (0 - norm_vector < result) & (result <= 1)

        volume_fractions = torch.zeros(tuple(self.volume_dimensions_voxels), dtype=torch.float).to(self.torch_device)
        filled_mask = torch.all(filled_mask_bool, axis=-1)

        border_mask = torch.all(border_bool, axis=-1)

        border_mask = torch.logical_xor(border_mask, filled_mask)

        edge_values = result[border_mask]

        fraction_values = torch.matmul(edge_values, matrix)

        larger_fraction_values = (x_edge_voxels + y_edge_voxels + z_edge_voxels) - fraction_values

        small_bool = fraction_values > 0
        large_bool = larger_fraction_values >= 1

        fraction_values[small_bool & large_bool] = 0
        fraction_values[fraction_values <= 0] = 1 + fraction_values[fraction_values <= 0]
        fraction_values[larger_fraction_values < 1] = larger_fraction_values[larger_fraction_values < 1]

        fraction_values = torch.abs(torch.prod(fraction_values, axis=-1))

        volume_fractions[filled_mask] = 1
        volume_fractions[border_mask] = fraction_values

        if partial_volume:
            mask = filled_mask | border_mask
        else:
            mask = filled_mask

        return mask.cpu().numpy(), volume_fractions[mask].cpu().numpy()


def define_rectangular_cuboid_structure_settings(start_mm: list, extent_mm: Union[int, list],
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
