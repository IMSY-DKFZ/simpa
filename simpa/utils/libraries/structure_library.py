# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import abstractmethod
from simpa.utils.settings_generator import Settings
from simpa.utils.tissue_properties import TissueProperties
from simpa.utils import Tags, SegmentationClasses
import operator
from simpa.utils.libraries.molecule_library import MolecularComposition
import traceback
import numpy as np
from simpa.utils import get_functional_from_deformation_settings


class Structures:
    """
    TODO
    """
    def __init__(self, settings: Settings):
        """
        TODO
        """
        self.structures = self.from_settings(settings)
        self.sorted_structures = sorted(self.structures, key=operator.attrgetter('priority'), reverse=True)

    def from_settings(self, global_settings):
        structures = list()
        if not Tags.STRUCTURES in global_settings:
            print("Did not find any structure definitions in the settings file!")
            return structures
        structure_settings = global_settings[Tags.STRUCTURES]
        for struc_tag_name in structure_settings:
            single_structure_settings = structure_settings[struc_tag_name]
            try:
                structure_class = globals()[single_structure_settings[Tags.STRUCTURE_TYPE]]
                structure = structure_class(global_settings, single_structure_settings)
                structures.append(structure)
            except Exception as e:
                print("An exception has occurred while trying to parse ", single_structure_settings[Tags.STRUCTURE_TYPE]," from the dictionary.")
                print("The structure type was", single_structure_settings[Tags.STRUCTURE_TYPE])
                print(traceback.format_exc())
                print("trying to continue as normal...")

        return structures


class GeometricalStructure:
    """
    TODO
    """

    def __init__(self, global_settings: Settings, single_structure_settings: Settings = None):

        self.voxel_spacing = global_settings[Tags.SPACING_MM]
        volume_x_dim = int(np.round(global_settings[Tags.DIM_VOLUME_X_MM] / self.voxel_spacing))
        volume_y_dim = int(np.round(global_settings[Tags.DIM_VOLUME_Y_MM] / self.voxel_spacing))
        volume_z_dim = int(np.round(global_settings[Tags.DIM_VOLUME_Z_MM] / self.voxel_spacing))
        self.volume_dimensions_voxels = np.asarray([volume_x_dim, volume_y_dim, volume_z_dim])

        self.volume_dimensions_mm = self.volume_dimensions_voxels * self.voxel_spacing
        self.do_deformation = (Tags.SIMULATE_DEFORMED_LAYERS in global_settings and
                               global_settings[Tags.SIMULATE_DEFORMED_LAYERS])
        if (Tags.ADHERE_TO_DEFORMATION in single_structure_settings and
                not single_structure_settings[Tags.ADHERE_TO_DEFORMATION]):
            self.do_deformation = False
        if self.do_deformation and Tags.DEFORMED_LAYERS_SETTINGS in global_settings:
            self.deformation_functional_mm = get_functional_from_deformation_settings(
                global_settings[Tags.DEFORMED_LAYERS_SETTINGS])
        else:
            self.deformation_functional_mm = None

        if single_structure_settings is None:
            self.molecule_composition = MolecularComposition()
            self.priority = 0
            return

        if Tags.PRIORITY in single_structure_settings:
            self.priority = single_structure_settings[Tags.PRIORITY]

        self.partial_volume = single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME]

        self.molecule_composition = single_structure_settings[Tags.MOLECULE_COMPOSITION]
        self.molecule_composition.update_internal_properties()

        self.geometrical_volume = np.zeros(self.volume_dimensions_voxels)
        self.params = self.get_params_from_settings(single_structure_settings)
        self.fill_internal_volume()

    def fill_internal_volume(self):
        indices, values = self.get_enclosed_indices()
        self.geometrical_volume[indices] = values

    @abstractmethod
    def get_enclosed_indices(self):
        pass

    @abstractmethod
    def get_params_from_settings(self, single_structure_settings):
        pass

    def properties_for_wavelength(self, wavelength) -> TissueProperties:
        return self.molecule_composition.get_properties_for_wavelength(wavelength)

    @abstractmethod
    def to_settings(self) -> Settings:
        """
        TODO
        :return : A tuple containing the settings key and the needed entries
        """
        settings_dict = Settings()
        settings_dict[Tags.PRIORITY] = self.priority
        settings_dict[Tags.STRUCTURE_TYPE] = self.__class__.__name__
        settings_dict[Tags.CONSIDER_PARTIAL_VOLUME] = self.partial_volume
        settings_dict[Tags.MOLECULE_COMPOSITION] = self.molecule_composition
        return settings_dict


class HorizontalLayerStructure(GeometricalStructure):

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
        start_mm = np.asarray(self.params[0])
        end_mm = np.asarray(self.params[1])
        partial_volume = self.params[2]
        start_voxels = start_mm / self.voxel_spacing
        direction_mm = end_mm - start_mm
        depth_voxels = direction_mm[2] / self.voxel_spacing

        if direction_mm[0] != 0 or direction_mm[1] != 0 or direction_mm[2] == 0:
            raise ValueError("Horizontal Layer structure needs a start and end vector in the form of [0, 0, n].")

        x, y, z = np.meshgrid(np.arange(self.volume_dimensions_voxels[0]),
                              np.arange(self.volume_dimensions_voxels[1]),
                              np.arange(self.volume_dimensions_voxels[2]),
                              indexing='ij')

        target_vector_voxels = np.subtract(np.stack([x, y, z], axis=-1), start_voxels)
        target_vector_voxels = target_vector_voxels[:, :, :, 2]
        if self.do_deformation:
            # the deformation functional needs mm as inputs and returns the result in reverse indexing order...
            deformation_values_mm = self.deformation_functional_mm(np.arange(self.volume_dimensions_voxels[0], step=1) *
                                                                   self.voxel_spacing,
                                                                   np.arange(self.volume_dimensions_voxels[1], step=1) *
                                                                   self.voxel_spacing).T
            target_vector_voxels = target_vector_voxels + (deformation_values_mm.reshape(self.volume_dimensions_voxels[0],
                                                                                         self.volume_dimensions_voxels[1], 1)
                                                           / self.voxel_spacing)

        volume_fractions = np.zeros(self.volume_dimensions_voxels)

        bools_first_layer = ((target_vector_voxels >= -1) & (target_vector_voxels < 0))
        volume_fractions[bools_first_layer] = 1-np.abs(target_vector_voxels[bools_first_layer])

        initial_fractions = np.max(volume_fractions, axis=2, keepdims=True)
        floored_depth_voxels = np.floor(depth_voxels-initial_fractions)

        bools_last_layer = ((target_vector_voxels >= floored_depth_voxels) &
                            (target_vector_voxels <= floored_depth_voxels + 1))

        volume_fractions[bools_last_layer] = depth_voxels - target_vector_voxels[bools_last_layer]
        volume_fractions[volume_fractions > depth_voxels] = depth_voxels
        volume_fractions[volume_fractions < 0] = 0

        if partial_volume:
            bools_fully_filled_layers = ((target_vector_voxels >= 0) & (target_vector_voxels < floored_depth_voxels))
        else:
            bools_fully_filled_layers = ((target_vector_voxels >= -0.5) & (target_vector_voxels < floored_depth_voxels + 0.5))
        volume_fractions[bools_fully_filled_layers] = 1

        if partial_volume:
            bools_all_layers = bools_first_layer | bools_last_layer | bools_fully_filled_layers
        else:
            bools_all_layers = bools_fully_filled_layers

        return bools_all_layers, volume_fractions[bools_all_layers]


class CircularTubularStructure(GeometricalStructure):

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_END_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_RADIUS_MM]),
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
        start_voxels = start_mm / self.voxel_spacing
        end_voxels = end_mm / self.voxel_spacing
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
        cylinder_vector = np.subtract(end_voxels, start_voxels)

        target_radius = np.linalg.norm(target_vector, axis=-1) * np.sin(
            np.arccos((np.dot(target_vector, cylinder_vector)) /
                      (np.linalg.norm(target_vector, axis=-1) * np.linalg.norm(cylinder_vector))))

        volume_fractions = np.zeros(self.volume_dimensions_voxels)

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

        return mask, volume_fractions[mask]


class SphericalStructure(GeometricalStructure):

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


class RectangularCuboidStructure(GeometricalStructure):

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


class ParallelepipedStructure(GeometricalStructure):
    """
    This class currently has no partial volume effects implemented. TODO
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
    from simpa.utils.deformation_manager import create_deformation_settings
    global_settings = Settings()
    global_settings[Tags.SPACING_MM] = 0.1

    global_settings[Tags.DIM_VOLUME_X_MM] = 10
    global_settings[Tags.DIM_VOLUME_Y_MM] = 10
    global_settings[Tags.DIM_VOLUME_Z_MM] = 10
    structure_settings = Settings()
    structure_settings[Tags.STRUCTURE_START_MM] = [1.31, 1.31, 1.31]
    structure_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [5, 0, 0]
    structure_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, 5, 2]
    structure_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 2, 5]
    structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

    structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()

    struc = ParallelepipedStructure(global_settings, structure_settings)

    vol = struc.geometrical_volume

    plt.imshow(vol[30, :, :])

    plt.show()


class EllipticalTubularStructure(GeometricalStructure):

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_END_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_RADIUS_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_ECCENTRICITY]),
                  np.asarray(single_structure_settings[Tags.CONSIDER_PARTIAL_VOLUME]))
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

        start_voxels = start_mm / self.voxel_spacing
        end_voxels = end_mm / self.voxel_spacing
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
        cylinder_vector = np.subtract(end_voxels, start_voxels)

        main_axis_length = radius_voxels/(1-eccentricity**2)**0.25
        main_axis_vector = np.array([cylinder_vector[1], -cylinder_vector[0], 0])
        main_axis_vector = main_axis_vector/np.linalg.norm(main_axis_vector) * main_axis_length

        minor_axis_length = main_axis_length*np.sqrt(1-eccentricity**2)
        minor_axis_vector = np.cross(cylinder_vector, main_axis_vector)
        minor_axis_vector = minor_axis_vector / np.linalg.norm(minor_axis_vector) * minor_axis_length

        dot_product = np.dot(target_vector, cylinder_vector)/np.linalg.norm(cylinder_vector)

        target_vector_projection = np.multiply(dot_product[:, :, :, np.newaxis], cylinder_vector)
        target_vector_from_projection = target_vector - target_vector_projection

        main_projection = np.dot(target_vector_from_projection, main_axis_vector) / main_axis_length

        minor_projection = np.dot(target_vector_from_projection, minor_axis_vector) / minor_axis_length

        radius_crit = np.sqrt(((main_projection/main_axis_length)**2 + (minor_projection/minor_axis_length)**2) *
                              radius_voxels**2)

        volume_fractions = np.zeros(self.volume_dimensions_voxels)
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

        return mask, volume_fractions[mask]

#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#     from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
#     global_settings = Settings()
#     global_settings[Tags.SPACING_MM] = 1
#     global_settings[Tags.DIM_VOLUME_X_MM] = 30
#     global_settings[Tags.DIM_VOLUME_Y_MM] = 30
#     global_settings[Tags.DIM_VOLUME_Z_MM] = 30
#     structure_settings = Settings()
#     structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
#     structure_settings[Tags.STRUCTURE_START_MM] = [15, 0, 15]
#     structure_settings[Tags.STRUCTURE_END_MM] = [15, 100, 15]
#     structure_settings[Tags.STRUCTURE_RADIUS_MM] = 5
#     structure_settings[Tags.STRUCTURE_ECCENTRICITY] = 0.56
#     structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True
#     ellipse = CircularTubularStructure(global_settings, structure_settings)
#     vol1 = ellipse.geometrical_volume
#
#     structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True
#     ellipse = EllipticalTubularStructure(global_settings, structure_settings)
#     vol2 = ellipse.geometrical_volume
#
#     main_axis = structure_settings[Tags.STRUCTURE_RADIUS_MM]/(1-structure_settings[Tags.STRUCTURE_ECCENTRICITY]**2)**0.25
#     minor_axis = main_axis * np.sqrt(1 - structure_settings[Tags.STRUCTURE_ECCENTRICITY] ** 2)
#     main_axis /= global_settings[Tags.SPACING_MM]
#     minor_axis /= global_settings[Tags.SPACING_MM]
#
#     ax = plt.subplot(111)
#     plt.imshow((vol1)[:, 0, :])
#     ax.add_patch(mpl.patches.Ellipse((15/global_settings[Tags.SPACING_MM] - 0.5,
#                                       15/global_settings[Tags.SPACING_MM] - 0.5),
#                                      2*main_axis, 2*minor_axis, 90, fill=False, color="red"))
#     plt.show()


class Background(GeometricalStructure):

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
