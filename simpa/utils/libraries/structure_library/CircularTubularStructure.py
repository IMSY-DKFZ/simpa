# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import torch
import numpy as np

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class CircularTubularStructure(GeometricalStructure):
    """
    Defines a circular tube which is defined by a start and end point as well as a radius. This structure implements
    partial volume effects. The tube can be set to adhere to a deformation defined by the
    simpa.utils.deformation_manager. The start and end points of the tube will then be shifted along the z-axis
    accordingly.
    Example usage:

        # single_structure_settings initialization
        structure = Settings()

        structure[Tags.PRIORITY] = 9
        structure[Tags.STRUCTURE_START_MM] = [50, 0, 50]
        structure[Tags.STRUCTURE_END_MM] = [50, 100, 50]
        structure[Tags.STRUCTURE_RADIUS_MM] = 5
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_END_MM],
                  single_structure_settings[Tags.STRUCTURE_RADIUS_MM],
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_END_MM] = self.params[1]
        settings[Tags.STRUCTURE_RADIUS_MM] = self.params[2]
        return settings

    def get_enclosed_indices(self):
        start_mm, end_mm, radius_mm, partial_volume = self.params
        start_mm = torch.tensor(start_mm, dtype=torch.float, device=self.torch_device)
        end_mm = torch.tensor(end_mm, dtype=torch.float, device=self.torch_device)
        radius_mm = torch.tensor(radius_mm, dtype=torch.float, device=self.torch_device)
        start_voxels = start_mm / self.voxel_spacing
        end_voxels = end_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing

        target_vector = torch.stack(torch.meshgrid(torch.arange(start=0.5, end=self.volume_dimensions_voxels[0], dtype=torch.float, device=self.torch_device),
                                                   torch.arange(
                                                       start=0.5, end=self.volume_dimensions_voxels[1], dtype=torch.float, device=self.torch_device),
                                                   torch.arange(
                                                       start=0.5, end=self.volume_dimensions_voxels[2], dtype=torch.float, device=self.torch_device),
                                                   indexing='ij'), dim=-1)
        target_vector -= start_voxels

        if partial_volume:
            radius_margin = 0.5
        else:
            radius_margin = 0.7071

        if self.do_deformation:
            # the deformation functional needs mm as inputs and returns the result in reverse indexing order...
            deformation_values_mm = self.deformation_functional_mm(torch.arange(self.volume_dimensions_voxels[0]) *
                                                                   self.voxel_spacing,
                                                                   torch.arange(self.volume_dimensions_voxels[1]) *
                                                                   self.voxel_spacing).T
            deformation_values_mm = deformation_values_mm.reshape(self.volume_dimensions_voxels[0],
                                                                  self.volume_dimensions_voxels[1], 1, 1)
            deformation_values_mm = torch.tile(torch.as_tensor(
                deformation_values_mm, dtype=torch.float, device=self.torch_device), (1, 1, self.volume_dimensions_voxels[2], 3))
            deformation_values_mm /= self.voxel_spacing
            target_vector += deformation_values_mm
            del deformation_values_mm
        cylinder_vector = torch.subtract(end_voxels, start_voxels)

        target_radius = torch.linalg.norm(target_vector, axis=-1) * torch.sin(
            torch.arccos((torch.matmul(target_vector, cylinder_vector)) /
                         (torch.linalg.norm(target_vector, axis=-1) * torch.linalg.norm(cylinder_vector))))
        del target_vector

        volume_fractions = torch.zeros(tuple(self.volume_dimensions_voxels),
                                       dtype=torch.float, device=self.torch_device)

        filled_mask = target_radius <= radius_voxels - 1 + radius_margin
        border_mask = (target_radius > radius_voxels - 1 + radius_margin) & \
                      (target_radius < radius_voxels + 2 * radius_margin)

        volume_fractions[filled_mask] = 1

        volume_fractions[border_mask] = 1 - (target_radius - (radius_voxels - radius_margin))[border_mask]
        volume_fractions[volume_fractions < 0] = 0

        if partial_volume:
            mask = filled_mask | border_mask
        else:
            mask = filled_mask

        return mask.cpu().numpy(), volume_fractions[mask].cpu().numpy()


def define_circular_tubular_structure_settings(tube_start_mm: list,
                                               tube_end_mm: list,
                                               molecular_composition: MolecularComposition,
                                               radius_mm: float = 2,
                                               priority: int = 10,
                                               consider_partial_volume: bool = False,
                                               adhere_to_deformation: bool = False):
    """
    TODO
    """
    return {
        Tags.STRUCTURE_START_MM: tube_start_mm,
        Tags.STRUCTURE_END_MM: tube_end_mm,
        Tags.STRUCTURE_RADIUS_MM: radius_mm,
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.CIRCULAR_TUBULAR_STRUCTURE
    }
