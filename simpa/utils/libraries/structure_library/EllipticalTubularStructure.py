# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import torch

from simpa.utils import Tags
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class EllipticalTubularStructure(GeometricalStructure):
    """
    Defines a elliptical tube which is defined by a start and end point as well as a radius and an eccentricity. The
    elliptical geometry corresponds to a circular tube of the specified radius which is compressed along the z-axis
    until it reaches the specified eccentricity under the assumption of a constant volume. This structure implements
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
        structure[Tags.STRUCTURE_ECCENTRICITY] = 0.8
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.ELLIPTICAL_TUBULAR_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_END_MM],
                  single_structure_settings[Tags.STRUCTURE_RADIUS_MM],
                  single_structure_settings[Tags.STRUCTURE_ECCENTRICITY],
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_END_MM] = self.params[1]
        settings[Tags.STRUCTURE_RADIUS_MM] = self.params[2]
        settings[Tags.STRUCTURE_ECCENTRICITY] = self.params[3]
        settings[Tags.CONSIDER_PARTIAL_VOLUME] = self.params[4]
        return settings

    def get_enclosed_indices(self):
        start_mm, end_mm, radius_mm, eccentricity, partial_volume = self.params
        start_mm = torch.tensor(start_mm, dtype=torch.float).to(self.torch_device)
        end_mm = torch.tensor(end_mm, dtype=torch.float).to(self.torch_device)
        radius_mm = torch.tensor(radius_mm, dtype=torch.float).to(self.torch_device)
        eccentricity = torch.tensor(eccentricity, dtype=torch.float).to(self.torch_device)
        partial_volume = torch.tensor(partial_volume, dtype=torch.float).to(self.torch_device)

        start_voxels = start_mm / self.voxel_spacing
        end_voxels = end_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing

        x, y, z = torch.meshgrid(torch.arange(self.volume_dimensions_voxels[0]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[1]).to(self.torch_device),
                                 torch.arange(self.volume_dimensions_voxels[2]).to(self.torch_device),
                                 indexing='ij')

        x = x + 0.5
        y = y + 0.5
        z = z + 0.5

        if partial_volume:
            radius_margin = 0.5
        else:
            radius_margin = 0.7071

        target_vector = torch.subtract(torch.stack([x, y, z], axis=-1), start_voxels)
        if self.do_deformation:
            # the deformation functional needs mm as inputs and returns the result in reverse indexing order...
            deformation_values_mm = self.deformation_functional_mm(torch.arange(self.volume_dimensions_voxels[0]) *
                                                                   self.voxel_spacing,
                                                                   torch.arange(self.volume_dimensions_voxels[1]) *
                                                                   self.voxel_spacing).T
            deformation_values_mm = deformation_values_mm.reshape(self.volume_dimensions_voxels[0],
                                                                  self.volume_dimensions_voxels[1], 1, 1)
            deformation_values_mm = torch.tile(torch.from_numpy(deformation_values_mm).to(
                self.torch_device), (1, 1, self.volume_dimensions_voxels[2], 3))
            target_vector = (target_vector + (deformation_values_mm / self.voxel_spacing)).float()
        cylinder_vector = torch.subtract(end_voxels, start_voxels)

        main_axis_length = radius_voxels/(1-eccentricity**2)**0.25
        main_axis_vector = torch.tensor([cylinder_vector[1], -cylinder_vector[0], 0]).to(self.torch_device)
        main_axis_vector = main_axis_vector/torch.linalg.norm(main_axis_vector) * main_axis_length

        minor_axis_length = main_axis_length*torch.sqrt(1-eccentricity**2)
        minor_axis_vector = torch.cross(cylinder_vector, main_axis_vector)
        minor_axis_vector = minor_axis_vector / torch.linalg.norm(minor_axis_vector) * minor_axis_length

        dot_product = torch.matmul(target_vector, cylinder_vector)/torch.linalg.norm(cylinder_vector)

        target_vector_projection = torch.multiply(dot_product[:, :, :, None], cylinder_vector)
        target_vector_from_projection = target_vector - target_vector_projection

        main_projection = torch.matmul(target_vector_from_projection, main_axis_vector) / main_axis_length

        minor_projection = torch.matmul(target_vector_from_projection, minor_axis_vector) / minor_axis_length

        radius_crit = torch.sqrt(((main_projection/main_axis_length)**2 + (minor_projection/minor_axis_length)**2) *
                                 radius_voxels**2)

        volume_fractions = torch.zeros(tuple(self.volume_dimensions_voxels), dtype=torch.float).to(self.torch_device)
        filled_mask = radius_crit <= radius_voxels - 1 + radius_margin
        border_mask = (radius_crit > radius_voxels - 1 + radius_margin) & \
                      (radius_crit < radius_voxels + 2 * radius_margin)

        volume_fractions[filled_mask] = 1
        volume_fractions[border_mask] = 1 - (radius_crit - (radius_voxels - radius_margin))[border_mask]
        volume_fractions[volume_fractions < 0] = 0
        volume_fractions[volume_fractions < 0] = 0

        if partial_volume:

            mask = filled_mask | border_mask
        else:
            mask = filled_mask

        return mask.cpu().numpy(), volume_fractions[mask].cpu().numpy()


def define_elliptical_tubular_structure_settings(tube_start_mm: list,
                                                 tube_end_mm: list,
                                                 molecular_composition: MolecularComposition,
                                                 radius_mm: float = 2,
                                                 eccentricity: float = 0.5,
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
        Tags.STRUCTURE_ECCENTRICITY: eccentricity,
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.ELLIPTICAL_TUBULAR_STRUCTURE
    }
