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
                structure = structure_class(global_settings, Settings(single_structure_settings))
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
    def to_settings(self) -> dict:
        """
        TODO
        :return : A tuple containing the settings key and the needed entries
        """
        settings_dict = dict()
        settings_dict[Tags.PRIORITY] = self.priority
        settings_dict[Tags.STRUCTURE_TYPE] = self.__class__.__name__
        settings_dict[Tags.MOLECULE_COMPOSITION] = self.molecule_composition
        return settings_dict


class HorizontalLayerStructure(GeometricalStructure):

    def get_params_from_settings(self, single_structure_settings):
        params = (single_structure_settings[Tags.STRUCTURE_START_MM],
                  single_structure_settings[Tags.STRUCTURE_END_MM])
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_END_MM] = self.params[1]
        return settings

    def get_enclosed_indices(self):
        start_mm = np.asarray(self.params[0])
        end_mm = np.asarray(self.params[1])
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

        bools_fully_filled_layers = ((target_vector_voxels >= 0) & (target_vector_voxels < floored_depth_voxels))
        volume_fractions[bools_fully_filled_layers] = 1

        bools_all_layers = bools_first_layer | bools_last_layer | bools_fully_filled_layers
        return bools_all_layers, volume_fractions[bools_all_layers]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
    from simpa.utils.deformation_manager import create_deformation_settings
    global_settings = Settings()
    global_settings[Tags.SPACING_MM] = 1
    global_settings[Tags.SIMULATE_DEFORMED_LAYERS] = True
    global_settings[Tags.DEFORMED_LAYERS_SETTINGS] = create_deformation_settings(bounds_mm=[[0, 100], [0, 100]],
                                                                                 maximum_z_elevation_mm=3,
                                                                                 filter_sigma=0,
                                                                                 cosine_scaling_factor=4)
    global_settings[Tags.DIM_VOLUME_X_MM] = 100
    global_settings[Tags.DIM_VOLUME_Y_MM] = 100
    global_settings[Tags.DIM_VOLUME_Z_MM] = 10
    structure_settings = Settings()
    structure_settings[Tags.STRUCTURE_START_MM] = [0, 0, 1.3]
    structure_settings[Tags.STRUCTURE_END_MM] = [0, 0, 3.5]
    structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    structure_settings[Tags.ADHERE_TO_DEFORMATION] = True
    ls = HorizontalLayerStructure(global_settings, structure_settings)
    plt.subplot(121)
    plt.imshow(ls.geometrical_volume[int(global_settings[Tags.DIM_VOLUME_X_MM] /
                                         global_settings[Tags.SPACING_MM] / 2), :, :])
    plt.subplot(122)
    plt.imshow(ls.geometrical_volume[:, int(global_settings[Tags.DIM_VOLUME_Y_MM] /
                                            global_settings[Tags.SPACING_MM] / 2), :])
    plt.show()


class TubularStructure(GeometricalStructure):

    def get_params_from_settings(self, single_structure_settings):
        params = (np.asarray(single_structure_settings[Tags.STRUCTURE_START_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_END_MM]),
                  np.asarray(single_structure_settings[Tags.STRUCTURE_RADIUS]))
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_END_MM] = self.params[1]
        settings[Tags.STRUCTURE_RADIUS] = self.params[2]
        return settings

    def get_enclosed_indices(self):
        start_mm, end_mm, radius_mm = self.params
        start_voxels = start_mm / self.voxel_spacing
        end_voxels = end_mm / self.voxel_spacing
        radius_voxels = radius_mm / self.voxel_spacing

        x, y, z = np.meshgrid(np.arange(self.volume_dimensions_voxels[0]),
                              np.arange(self.volume_dimensions_voxels[1]),
                              np.arange(self.volume_dimensions_voxels[2]),
                              indexing='ij')

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start_voxels)
        cylinder_vector = np.subtract(end_voxels, start_voxels)

        target_radius = np.linalg.norm(target_vector, axis=-1) * np.sin(
            np.arccos((np.dot(target_vector, cylinder_vector)) /
                      (np.linalg.norm(target_vector, axis=-1) * np.linalg.norm(cylinder_vector))))

        return target_radius < radius_voxels, 1


class SphericalStructure(GeometricalStructure):

    def get_params_from_settings(self, single_structure_settings):
        params = single_structure_settings[Tags.STRUCTURE_START_MM], \
                 single_structure_settings[Tags.STRUCTURE_RADIUS]
        return params

    def to_settings(self):
        settings = super().to_settings()
        settings[Tags.STRUCTURE_START_MM] = self.params[0]
        settings[Tags.STRUCTURE_RADIUS] = self.params[2]
        return settings

    def get_enclosed_indices(self):
        start, radius = self.params
        x, y, z = np.meshgrid(np.arange(self.volume_dimensions_voxels[0]),
                              np.arange(self.volume_dimensions_voxels[1]),
                              np.arange(self.volume_dimensions_voxels[2]),
                              indexing='ij')

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start)
        target_radius = np.linalg.norm(target_vector, axis=-1)

        return target_radius < radius, 1


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
            super().__init__(global_settings, background_settings)
        else:
            super().__init__(global_settings)
            self.priority = 0

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        return settings_dict
