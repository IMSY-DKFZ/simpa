# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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
from simpa.utils import Tags
from simpa.utils.calculate import rotation
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
            except Exception:
                print("An exception has occurred while trying to parse ", single_structure_settings[Tags.STRUCTURE_TYPE]," from the dictionary.")
                print("The structure type was", single_structure_settings[Tags.STRUCTURE_TYPE])
                print(traceback.format_exc())
                print("trying to continue as normal...")

        return structures


class GeometricalStructure:
    """
    Base class for all model-based structures for ModelBasedVolumeCreator. A GeometricalStructure has an internal
    representation of its own geometry. This is represented by self.geometrical_volume which is a 3D array that defines
    for every voxel within the simulation volume if it is enclosed in the GeometricalStructure or if it is outside.
    Most of the GeometricalStructures implement a partial volume effect. So if a voxel has the value 1, it is completely
    enclosed by the GeometricalStructure. If a voxel has a value between 0 and 1, that fraction of the volume is
    occupied by the GeometricalStructure. If a voxel has the value 0, it is outside of the GeometricalStructure.
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
        """
        Fills self.geometrical_volume of the GeometricalStructure.
        """
        indices, values = self.get_enclosed_indices()
        self.geometrical_volume[indices] = values

    @abstractmethod
    def get_enclosed_indices(self):
        """
        Gets indices of the voxels that are either entirely or partially occupied by the GeometricalStructure.
        :return: mask for a numpy array
        """
        pass

    @abstractmethod
    def get_params_from_settings(self, single_structure_settings):
        """
        Gets all the parameters required for the specific GeometricalStructure.
        :param single_structure_settings: Settings which describe the specific GeometricalStructure.
        :return: Tuple of parameters
        """
        pass

    def properties_for_wavelength(self, wavelength) -> TissueProperties:
        """
        Returns the values corresponding to each optical/acoustic property used in SIMPA.
        :param wavelength: Wavelength of the queried properties
        :return: optical/acoustic properties
        """
        return self.molecule_composition.get_properties_for_wavelength(wavelength)

    @abstractmethod
    def to_settings(self) -> Settings:
        """
        Creates a Settings dictionary which contains all the parameters needed to create the same GeometricalStructure
        again.
        :return : A tuple containing the settings key and the needed entries
        """
        settings_dict = Settings()
        settings_dict[Tags.PRIORITY] = self.priority
        settings_dict[Tags.STRUCTURE_TYPE] = self.__class__.__name__
        settings_dict[Tags.CONSIDER_PARTIAL_VOLUME] = self.partial_volume
        settings_dict[Tags.MOLECULE_COMPOSITION] = self.molecule_composition
        return settings_dict


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


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
#     from simpa.utils.deformation_manager import create_deformation_settings
#     global_settings = Settings()
#     global_settings[Tags.SPACING_MM] = 0.1
#
#     global_settings[Tags.DIM_VOLUME_X_MM] = 10
#     global_settings[Tags.DIM_VOLUME_Y_MM] = 10
#     global_settings[Tags.DIM_VOLUME_Z_MM] = 10
#     structure_settings = Settings()
#     structure_settings[Tags.STRUCTURE_START_MM] = [1.31, 1.31, 1.31]
#     structure_settings[Tags.STRUCTURE_FIRST_EDGE_MM] = [5, 0, 0]
#     structure_settings[Tags.STRUCTURE_SECOND_EDGE_MM] = [0, 5, 2]
#     structure_settings[Tags.STRUCTURE_THIRD_EDGE_MM] = [0, 2, 5]
#     structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True
#
#     structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
#
#     struc = ParallelepipedStructure(global_settings, structure_settings)
#
#     vol = struc.geometrical_volume
#
#     plt.imshow(vol[30, :, :])
#
#     plt.show()


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


class Background(GeometricalStructure):
    """
    Defines a background that fills the whole simulation volume. It is always given the priority of 0 so that other
    structures can overwrite it when necessary.
    Example usage:
        background_dictionary = Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.1, 100.0, 0.9)
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    """

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY

    import time
    timer = time.time()

    _global_settings = Settings()
    _global_settings[Tags.SPACING_MM] = 2
    _global_settings[Tags.DIM_VOLUME_X_MM] = 80
    _global_settings[Tags.DIM_VOLUME_Y_MM] = 90
    _global_settings[Tags.DIM_VOLUME_Z_MM] = 100

    structure_settings = Settings()
    structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    structure_settings[Tags.STRUCTURE_START_MM] = [50, 0, 50]
    structure_settings[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    structure_settings[Tags.STRUCTURE_RADIUS_MM] = 4
    structure_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.2
    structure_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    structure_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 10
    structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

    vessel = VesselStructure(_global_settings, structure_settings)
    vol1 = vessel.geometrical_volume
    print("generation of the vessel took", time.time() - timer)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(vol1, shade=True)
    plt.show()
