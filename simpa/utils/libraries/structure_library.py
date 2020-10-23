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
from simpa.utils.tissueproperties import TissueProperties
from simpa.utils import Tags


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

        # TODO sort the structures by priority
        self.sorted_structures = None

    def from_settings(self, settings):
        structures = list()
        if not Tags.STRUCTURES in settings:
            print("Did not find any structure definitions in the settings file!")
            return structures
        for struc_tag_names in settings[Tags.STRUCTURES]:
            print("Trying to add structure under name struc_tag_names")
            structure = settings[Tags.STRUCTURES][struc_tag_names]
            try:
                structure_class = globals()[structure[Tags.STRUCTURE_TYPE]]
                structure = structure_class(structure)
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
    def __init__(self, settings):
        self.priority = priority

    @abstractmethod
    def properties_for_voxel(self, x_idx_px, y_idx_px, z_idx_px) -> TissueProperties:
        pass


