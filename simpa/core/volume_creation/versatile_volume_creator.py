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

from simpa.core.volume_creation import VolumeCreatorBase
from simpa.utils.libraries.structure_library import Structures
from simpa.utils import Tags
import numpy as np


class ModelBasedVolumeCreator(VolumeCreatorBase):
    """
    The model-based volume creator uses a set of rules how to generate structures
    to create a simulation volume.
    These structures are added to the dictionary and later combined by the algorithm::

        # Initialise settings dictionaries
        simulation_settings = Settings()
        all_structures = Settings()
        structure = Settings()

        # Definition of en example structure.
        # The concrete structure parameters will change depending on the
        # structure type
        structure[Tags.PRIORITY] = 1
        structure[Tags.STRUCTURE_START_MM] = [0, 0, 0]
        structure[Tags.STRUCTURE_END_MM] = [0, 0, 100]
        structure[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
        structure[Tags.CONSIDER_PARTIAL_VOLUME] = True
        structure[Tags.ADHERE_TO_DEFORMATION] = True
        structure[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

        all_structures["arbitrary_identifier"] = structure

        simulation_settings[Tags.STRUCTURES] = all_structures

        # ...
        # Define further simulation settings
        # ...

        simulate(simulation_settings)


    """

    def __init__(self):
        self.EPS = 1e-4

    def create_simulation_volume(self, settings) -> dict:
        """
        This method creates a in silico respresentation of a tissue as described in the settings file that is given.

        :param settings: a dictionary containing all relevant Tags for the simulation to be able to instantiate a tissue.
        :return: a path to a npz file containing characteristics of the simulated volume:
                absorption, scattering, anisotropy, oxygenation, and a segmentation mask. All of these are given as 3d
                numpy arrays.
        """

        volumes, x_dim_px, y_dim_px, z_dim_px = self.create_empty_volumes(settings)
        global_volume_fractions = np.zeros((x_dim_px, y_dim_px, z_dim_px))
        max_added_fractions = np.zeros((x_dim_px, y_dim_px, z_dim_px))
        wavelength = settings[Tags.WAVELENGTH]

        structure_list = Structures(settings)
        priority_sorted_structures = structure_list.sorted_structures

        for structure in priority_sorted_structures:
            print(type(structure))

            structure_properties = structure.properties_for_wavelength(wavelength)

            structure_volume_fractions = structure.geometrical_volume
            structure_indexes_mask = structure_volume_fractions > 0
            global_volume_fractions_mask = global_volume_fractions < 1
            mask = structure_indexes_mask & global_volume_fractions_mask
            added_volume_fraction = (global_volume_fractions + structure_volume_fractions)

            added_volume_fraction[added_volume_fraction <= 1 & mask] = structure_volume_fractions[
                added_volume_fraction <= 1 & mask]

            selector_more_than_1 = added_volume_fraction > 1
            if selector_more_than_1.any():
                remaining_volume_fraction_to_fill = 1 - global_volume_fractions[selector_more_than_1]
                fraction_to_be_filled = structure_volume_fractions[selector_more_than_1]
                added_volume_fraction[selector_more_than_1] = np.min([remaining_volume_fraction_to_fill,
                                                                      fraction_to_be_filled], axis=0)
            for key in volumes.keys():
                if structure_properties[key] is None:
                    continue
                if key == Tags.PROPERTY_SEGMENTATION:
                    added_fraction_greater_than_any_added_fraction = added_volume_fraction > max_added_fractions
                    volumes[key][added_fraction_greater_than_any_added_fraction & mask] = structure_properties[key]
                    max_added_fractions[added_fraction_greater_than_any_added_fraction & mask] = \
                        added_volume_fraction[added_fraction_greater_than_any_added_fraction & mask]
                else:
                    volumes[key][mask] += added_volume_fraction[mask] * structure_properties[key]

            global_volume_fractions[mask] += added_volume_fraction[mask]

        return volumes
