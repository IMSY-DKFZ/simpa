# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import torch

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
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_RADIUS_MM],
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_RADIUS_MM] = self.params[1]
        return settings

    def get_enclosed_indices(self):
        start_mm, radius_mm, partial_volume = self.params
        start_mm = torch.tensor(start_mm, dtype=torch.float, device=self.torch_device)
        radius_mm = torch.tensor(radius_mm, dtype=torch.float, device=self.torch_device)

        start_voxels = start_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing

        target_vector = torch.stack(torch.meshgrid(torch.arange(start=0.5, end=self.volume_dimensions_voxels[0], device=self.torch_device),
                                                   torch.arange(
                                                       start=0.5, end=self.volume_dimensions_voxels[1], device=self.torch_device),
                                                   torch.arange(
                                                       start=0.5, end=self.volume_dimensions_voxels[2], device=self.torch_device),
                                                   indexing='ij'), dim=-1)
        target_vector -= start_voxels

        if partial_volume:
            radius_margin = 0.5
        else:
            radius_margin = 0.7071

        target_radius = torch.linalg.norm(target_vector, axis=-1)
        del target_vector

        volume_fractions = torch.zeros(tuple(self.volume_dimensions_voxels),
                                       dtype=torch.float, device=self.torch_device)
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

        return mask.cpu().numpy(), volume_fractions[mask].cpu().numpy()


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
