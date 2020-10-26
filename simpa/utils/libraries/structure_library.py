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


class Structures:
    """
    TODO
    """
    def __init__(self, settings: Settings):
        """
        TODO
        """
        # TODO get all structures from dictionary
        self.structures = self.from_settings(settings)
        print(self.structures)

        # TODO sort the structures by priority
        self.sorted_structures = sorted(self.structures, key=operator.attrgetter('priority'))
        print(self.sorted_structures)

    def from_settings(self, settings):
        structures = list()
        if not Tags.STRUCTURES in settings:
            print("Did not find any structure definitions in the settings file!")
            return structures
        for struc_tag_name in settings[Tags.STRUCTURES]:
            structure_dict = settings[Tags.STRUCTURES][struc_tag_name]
            try:
                structure_class = globals()[structure_dict[Tags.STRUCTURE_TYPE]]
                structure = structure_class(structure_dict[Tags.MOLECULE_COMPOSITION])
                structures.append(structure)
                print(structure.priority)
            except Exception as e:
                print("An exception has occurred while trying to parse the structure from the dictionary.")
                print(e)
                print("trying to continue as normal...")

        return structures


class Structure:
    """
    TODO
    """

    def __init__(self, settings=None):
        self.priority = 0
        self.molecule_composition = MolecularComposition()

        if settings is None:
            return

        if Tags.PRIORITY in settings:
            self.priority = settings[Tags.PRIORITY]

        self.molecule_composition = MolecularComposition(settings=settings)

    @abstractmethod
    def properties_for_voxel(self, x_idx_px, y_idx_px, z_idx_px, wavelength) -> TissueProperties:
        pass

    @abstractmethod
    def to_settings(self) -> dict:
        """
        TODO
        :return : A tuple containing the settings key and the needed entries
        """
        settings_dict = dict()
        settings_dict[Tags.STRUCTURE_TYPE] = self.__class__.__name__
        settings_dict[Tags.MOLECULE_COMPOSITION] = self.molecule_composition
        return settings_dict


class Background(Structure):

    def __init__(self, settings=None):
        super().__init__(settings)

        if settings is None:
            return

    def properties_for_voxel(self, x_idx_px, y_idx_px, z_idx_px, wavelength) -> TissueProperties:
        background_properties = TissueProperties()
        background_properties.volume_fraction = 1
        background_properties[Tags.PROPERTY_DENSITY] = 1000
        background_properties[Tags.PROPERTY_SPEED_OF_SOUND] = 1500
        background_properties[Tags.PROPERTY_OXYGENATION] = 0.5
        background_properties[Tags.PROPERTY_SEGMENTATION] = SegmentationClasses.GENERIC
        background_properties[Tags.PROPERTY_ABSORPTION_PER_CM] = 0.1
        background_properties[Tags.PROPERTY_SCATTERING_PER_CM] = 100
        background_properties[Tags.PROPERTY_ANISOTROPY] = 0.9
        background_properties[Tags.PROPERTY_ALPHA_COEFF] = 1
        background_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER] = 1
        return background_properties

    def to_settings(self) -> dict:
        settings_dict = super().to_settings()
        settings_dict[Tags.PRIORITY] = 5
        print(settings_dict)
        return settings_dict
