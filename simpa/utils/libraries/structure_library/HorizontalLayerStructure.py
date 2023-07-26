# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import torch

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class HorizontalLayerStructure(GeometricalStructure):
    """
    Defines a Layer structure which spans the xy-plane in the SIMPA axis convention. The thickness of the layer is
    defined along the z-axis. This layer can be deformed by the simpa.utils.deformation_manager.
    Example usage:

        # single_structure_settings initialization
        structure = Settings()

        structure[Tags.PRIORITY] = 10
        structure[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        structure[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_END_MM],
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_END_MM] = self.params[1]
        return settings

    def get_enclosed_indices(self):
        start_mm = torch.tensor(self.params[0], dtype=torch.float).to(self.torch_device)
        end_mm = torch.tensor(self.params[1], dtype=torch.float).to(self.torch_device)
        partial_volume = self.params[2]
        start_voxels = start_mm / self.voxel_spacing
        direction_mm = end_mm - start_mm
        depth_voxels = direction_mm[2] / self.voxel_spacing

        if direction_mm[0] != 0 or direction_mm[1] != 0 or direction_mm[2] == 0:
            raise ValueError("Horizontal Layer structure needs a start and end vector in the form of [0, 0, n].")

        x, y, z = torch.meshgrid(torch.arange(self.volume_dimensions_voxels[0]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[1]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[2]).to(self.torch_device),
                                 indexing='ij')

        target_vector_voxels = torch.subtract(torch.stack([x, y, z], axis=-1), start_voxels)
        target_vector_voxels = target_vector_voxels[:, :, :, 2]
        if self.do_deformation:
            # the deformation functional needs mm as inputs and returns the result in reverse indexing order...
            deformation_values_mm = self.deformation_functional_mm(torch.arange(self.volume_dimensions_voxels[0]) *
                                                                   self.voxel_spacing,
                                                                   torch.arange(self.volume_dimensions_voxels[1]) *
                                                                   self.voxel_spacing).T
            target_vector_voxels = (target_vector_voxels + torch.from_numpy(deformation_values_mm.reshape(
                self.volume_dimensions_voxels[0],
                self.volume_dimensions_voxels[1], 1)).to(self.torch_device) / self.voxel_spacing).float()

        volume_fractions = torch.zeros(tuple(self.volume_dimensions_voxels), dtype=torch.float).to(self.torch_device)

        if partial_volume:
            bools_first_layer = ((target_vector_voxels >= -1) & (target_vector_voxels < 0))

            volume_fractions[bools_first_layer] = 1 - torch.abs(target_vector_voxels[bools_first_layer])

            initial_fractions = torch.max(volume_fractions, dim=2, keepdims=True)[0]
            floored_depth_voxels = torch.floor(depth_voxels - initial_fractions)

            bools_fully_filled_layers = ((target_vector_voxels >= 0) & (target_vector_voxels < floored_depth_voxels))

            bools_last_layer = ((target_vector_voxels >= floored_depth_voxels) &
                                (target_vector_voxels <= floored_depth_voxels + 1))

            volume_fractions[bools_last_layer] = depth_voxels - target_vector_voxels[bools_last_layer]
            volume_fractions[volume_fractions > depth_voxels] = depth_voxels
            volume_fractions[volume_fractions < 0] = 0

            bools_all_layers = bools_first_layer | bools_last_layer | bools_fully_filled_layers

        else:
            bools_fully_filled_layers = ((target_vector_voxels >= -0.5) & (target_vector_voxels < depth_voxels - 0.5))

            bools_all_layers = bools_fully_filled_layers

        volume_fractions[bools_fully_filled_layers] = 1

        return bools_all_layers.cpu().numpy(), volume_fractions[bools_all_layers].cpu().numpy()


def define_horizontal_layer_structure_settings(molecular_composition: MolecularComposition,
                                               z_start_mm: float = 0, thickness_mm: float = 0, priority: int = 10,
                                               consider_partial_volume: bool = False,
                                               adhere_to_deformation: bool = False):
    """
    TODO
    """
    return {
        Tags.STRUCTURE_START_MM: [0, 0, z_start_mm],
        Tags.STRUCTURE_END_MM: [0, 0, z_start_mm+thickness_mm],
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
    }
