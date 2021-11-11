# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np

from simpa.utils import Tags
from simpa.utils.calculate import rotation
from simpa.utils.libraries.molecule_library import MolecularComposition
from simpa.utils.libraries.structure_library.StructureBase import GeometricalStructure


class VesselStructure(GeometricalStructure):
    """
    Defines a vessel tree that is generated randomly in the simulation volume. The generation process begins at the
    start with a specified radius. The vessel grows roughly in the specified direction. The deviation is specified by
    the curvature factor. Furthermore, the radius of the vessel can vary depending on the specified radius variation
    factor. The bifurcation length defines how long a vessel can get until it will bifurcate. This structure implements
    partial volume effects.
    Example usage:

        # single_structure_settings initialization
        structure_settings = Settings()

        structure_settings[Tags.PRIORITY] = 10
        structure_settings[Tags.STRUCTURE_START_MM] = [50, 0, 50]
        structure_settings[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
        structure_settings[Tags.STRUCTURE_RADIUS_MM] = 4
        structure_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.05
        structure_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
        structure_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 70
        structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
        structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure_settings[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE

    """

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  single_structure_settings[Tags.STRUCTURE_RADIUS_MM],
                  np.asarray(single_structure_settings[Tags.STRUCTURE_DIRECTION]),
                  single_structure_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM],
                  single_structure_settings[Tags.STRUCTURE_CURVATURE_FACTOR],
                  single_structure_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR],
                  single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_RADIUS_MM] = self.params[1]
        settings[Tags.STRUCTURE_DIRECTION] = self.params[2]
        settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = self.params[3]
        settings[Tags.STRUCTURE_CURVATURE_FACTOR] = self.params[4]
        settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = self.params[5]
        settings[Tags.CONSIDER_PARTIAL_VOLUME] = self.params[6]
        return settings

    def fill_internal_volume(self):
        self.geometrical_volume = self.get_enclosed_indices()

    def calculate_vessel_samples(self, position, direction, bifurcation_length, radius, radius_variation,
                                 volume_dimensions, curvature_factor):
        position_array = [position]
        radius_array = [radius]
        samples = 0

        while np.all(position < volume_dimensions) and np.all(0 <= position):
            if samples >= bifurcation_length:
                vessel_branch_positions1 = position
                vessel_branch_positions2 = position
                angles = np.random.normal(np.pi / 16, np.pi / 8, 3)
                vessel_branch_directions1 = np.squeeze(np.array(np.matmul(rotation(angles), direction)))
                vessel_branch_directions2 = np.squeeze(np.array(np.matmul(rotation(-angles), direction)))
                vessel_branch_radius1 = 1 / np.sqrt(2) * radius
                vessel_branch_radius2 = 1 / np.sqrt(2) * radius
                vessel_branch_radius_variation1 = 1 / np.sqrt(2) * radius_variation
                vessel_branch_radius_variation2 = 1 / np.sqrt(2) * radius_variation

                if vessel_branch_radius1 >= 0.5:
                    vessel1_pos, vessel1_rad = self.calculate_vessel_samples(vessel_branch_positions1,
                                                                             vessel_branch_directions1,
                                                                             bifurcation_length,
                                                                             vessel_branch_radius1,
                                                                             vessel_branch_radius_variation1,
                                                                             volume_dimensions, curvature_factor)
                    position_array += vessel1_pos
                    radius_array += vessel1_rad

                if vessel_branch_radius2 >= 0.5:
                    vessel2_pos, vessel2_rad = self.calculate_vessel_samples(vessel_branch_positions2,
                                                                             vessel_branch_directions2,
                                                                             bifurcation_length,
                                                                             vessel_branch_radius2,
                                                                             vessel_branch_radius_variation2,
                                                                             volume_dimensions, curvature_factor)
                    position_array += vessel2_pos
                    radius_array += vessel2_rad
                break

            position = np.add(position, direction)
            position_array.append(position)
            radius_array.append(np.random.uniform(-1, 1) * radius_variation + radius)

            step_vector = np.random.uniform(-1, 1, 3)
            step_vector = direction + curvature_factor * step_vector
            direction = step_vector / np.linalg.norm(step_vector)
            samples += 1

        return position_array, radius_array

    def get_enclosed_indices(self):
        start_mm, radius_mm, direction_mm, bifurcation_length_mm, curvature_factor, \
            radius_variation_factor, partial_volume = self.params
        start_voxels = start_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing
        direction_voxels = direction_mm / self.voxel_spacing
        direction_vector_voxels = direction_voxels / np.linalg.norm(direction_voxels)
        bifurcation_length_voxels = bifurcation_length_mm / self.voxel_spacing

        position_array, radius_array = self.calculate_vessel_samples(start_voxels, direction_vector_voxels,
                                                                     bifurcation_length_voxels, radius_voxels,
                                                                     radius_variation_factor,
                                                                     self.volume_dimensions_voxels,
                                                                     curvature_factor)

        position_array = np.array(position_array)

        x, y, z = np.ogrid[0:self.volume_dimensions_voxels[0],
                           0:self.volume_dimensions_voxels[1],
                           0:self.volume_dimensions_voxels[2]]

        volume_fractions = np.zeros(self.volume_dimensions_voxels)

        if partial_volume:
            radius_margin = 0.5
        else:
            radius_margin = 0.7071

        for position, radius in zip(position_array, radius_array):
            target_radius = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2 + (z - position[2]) ** 2)

            filled_mask = target_radius <= radius - 1 + radius_margin
            border_mask = (target_radius > radius - 1 + radius_margin) & \
                          (target_radius < radius + 2 * radius_margin)

            volume_fractions[filled_mask] = 1
            old_border_values = volume_fractions[border_mask]
            new_border_values = 1 - (target_radius - (radius - radius_margin))[border_mask]
            volume_fractions[border_mask] = np.maximum(old_border_values, new_border_values)

        return volume_fractions


def define_vessel_structure_settings(vessel_start_mm: list,
                                     vessel_direction_mm: list,
                                     molecular_composition: MolecularComposition,
                                     radius_mm: float = 2,
                                     curvature_factor: float = 0.05,
                                     radius_variation_factor: float = 1.0,
                                     bifurcation_length_mm: float = 7,
                                     priority: int = 10,
                                     consider_partial_volume: bool = False,
                                     adhere_to_deformation: bool = False):
    """
    TODO
    """
    return {
        Tags.STRUCTURE_START_MM: vessel_start_mm,
        Tags.STRUCTURE_DIRECTION: vessel_direction_mm,
        Tags.STRUCTURE_RADIUS_MM: radius_mm,
        Tags.STRUCTURE_CURVATURE_FACTOR: curvature_factor,
        Tags.STRUCTURE_RADIUS_VARIATION_FACTOR: radius_variation_factor,
        Tags.STRUCTURE_BIFURCATION_LENGTH_MM: bifurcation_length_mm,
        Tags.PRIORITY: priority,
        Tags.MOLECULE_COMPOSITION: molecular_composition,
        Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
        Tags.ADHERE_TO_DEFORMATION: adhere_to_deformation,
        Tags.STRUCTURE_TYPE: Tags.VESSEL_STRUCTURE
    }