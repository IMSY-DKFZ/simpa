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


class Structures:
    """
    TODO
    """
    def __init__(self, settings: Settings):
        """
        TODO
        """
        self.structures = self.from_settings(settings)
        self.sorted_structures = sorted(self.structures, key=operator.attrgetter('priority'))

    def from_settings(self, settings):
        structures = list()
        if not Tags.STRUCTURES in settings:
            print("Did not find any structure definitions in the settings file!")
            return structures
        structure_settings = settings[Tags.STRUCTURES]
        for struc_tag_name in structure_settings:
            single_structure_settings = structure_settings[struc_tag_name]
            try:
                structure_class = globals()[single_structure_settings[Tags.STRUCTURE_TYPE]]
                structure = structure_class(single_structure_settings)
                structures.append(structure)
            except Exception as e:
                print("An exception has occurred while trying to parse the structure from the dictionary.")
                print("The structure type was", single_structure_settings[Tags.STRUCTURE_TYPE])
                print(traceback.format_exc())
                print("trying to continue as normal...")

        return structures


class Structure:
    """
    TODO
    """

    def __init__(self, single_structure_settings=None):

        if single_structure_settings is None:
            self.molecule_composition = MolecularComposition()
            self.priority = 0
            return

        if Tags.PRIORITY in single_structure_settings:
            self.priority = single_structure_settings[Tags.PRIORITY]

        self.molecule_composition = single_structure_settings[Tags.MOLECULE_COMPOSITION]
        self.molecule_composition.update_internal_properties()

    def properties_for_voxel_and_wavelength(self, x_idx_px, y_idx_px, z_idx_px, wavelength) -> TissueProperties:
        tp = self.molecule_composition.get_properties_for_wavelength(wavelength)
        volume_fraction = self.volume_fraction_for_voxel(x_idx_px, y_idx_px, z_idx_px)
        tp.volume_fraction = volume_fraction
        return tp

    @abstractmethod
    def volume_fraction_for_voxel(self, x_idx_px, y_idx_px, z_idx_px) -> float:
        pass

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


class GeometricalStructure(Structure):
    def __init__(self, single_structure_settings):
        super().__init__()
        voxel_spacing = single_structure_settings[Tags.SPACING_MM]
        volume_x_dim = int(round(single_structure_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing))
        volume_y_dim = int(round(single_structure_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing))
        volume_z_dim = int(round(single_structure_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing))
        self.volume_dimensions = (volume_x_dim, volume_y_dim, volume_z_dim)

        self.geometrical_volume = np.zeros(self.volume_dimensions)

        self.fill_volume(single_structure_settings)

    def fill_volume(self, single_structure_settings):
        params = self.get_params_from_settings(single_structure_settings)
        indices, values = self.get_enclosed_indices(params)
        self.geometrical_volume[indices] = values

    def volume_fraction_for_voxel(self, x_idx_px, y_idx_px, z_idx_px) -> float:
        return self.geometrical_volume[x_idx_px, y_idx_px, z_idx_px]

    @abstractmethod
    def get_enclosed_indices(self, params):
        pass

    @abstractmethod
    def get_params_from_settings(self, single_structure_settings):
        pass


class TubularStructure(GeometricalStructure):
    def get_params_from_settings(self, single_structure_settings):
        params = single_structure_settings[Tags.STRUCTURE_START], \
                 single_structure_settings[Tags.STRUCTURE_END], \
                 single_structure_settings[Tags.STRUCTURE_RADIUS]
        return params

    def get_enclosed_indices(self, params):
        start, end, radius = params
        x, y, z = np.meshgrid(np.arange(self.volume_dimensions[0]),
                              np.arange(self.volume_dimensions[1]),
                              np.arange(self.volume_dimensions[2]))

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start)
        cylinder_vector = np.subtract(end, start)

        target_radius = np.linalg.norm(target_vector, axis=-1) * np.sin(
            np.arccos((np.dot(target_vector, cylinder_vector)) /
                      (np.linalg.norm(target_vector, axis=-1) * np.linalg.norm(cylinder_vector))))

        return target_radius < radius, 1

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        return settings_dict


class SphericalStructure(GeometricalStructure):
    def get_params_from_settings(self, single_structure_settings):
        params = single_structure_settings[Tags.STRUCTURE_START], \
                 single_structure_settings[Tags.STRUCTURE_RADIUS]
        return params

    def get_enclosed_indices(self, params):
        start, radius = params
        x, y, z = np.meshgrid(np.arange(self.volume_dimensions[0]),
                              np.arange(self.volume_dimensions[1]),
                              np.arange(self.volume_dimensions[2]))

        target_vector = np.subtract(np.stack([x, y, z], axis=-1), start)
        target_radius = np.linalg.norm(target_vector, axis=-1)

        return target_radius < radius, 1

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        return settings_dict


settings = Settings({Tags.DIM_VOLUME_X_MM: 20,
                     Tags.DIM_VOLUME_Y_MM: 20,
                     Tags.DIM_VOLUME_Z_MM: 20,
                     Tags.SPACING_MM: 0.1,
                     Tags.STRUCTURE_START: [0, 0, 0],
                     Tags.STRUCTURE_END: [200, 200, 200],
                     Tags.STRUCTURE_RADIUS: 50})

test = SphericalStructure(settings)
import matplotlib.pyplot as plt

plt.imshow(test.geometrical_volume[0, :, :])
plt.show()


class Background(Structure):

    def __init__(self, background_settings=None):

        if background_settings is not None:
            background_settings[Tags.PRIORITY] = 0
            super().__init__(background_settings)
        else:
            super().__init__()
            self.priority = 0

    def volume_fraction_for_voxel(self, x_idx_px, y_idx_px, z_idx_px) -> float:
        return 1.0

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        return settings_dict
