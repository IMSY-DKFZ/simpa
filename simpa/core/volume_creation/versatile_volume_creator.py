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

from simpa.core.volume_creation import VolumeCreatorBase
from simpa.utils.libraries.structure_library import Structures
from simpa.utils.tissue_properties import TissueProperties
import numpy as np


class VersatileVolumeCreator(VolumeCreatorBase):

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
        wavelength = 800

        structure_list = Structures(settings)

        for z_idx_px in range(z_dim_px):
            print(z_idx_px)
            for y_idx_px in range(y_dim_px):
                for x_idx_px in range(x_dim_px):
                    properties = np.asarray([structure.properties_for_voxel_and_wavelength(x_idx_px, y_idx_px,
                                                                                           z_idx_px, wavelength)
                                             for structure in structure_list.sorted_structures])
                    priorities = np.asarray([structure.priority for structure in structure_list.sorted_structures])

                    priorities = priorities[properties != np.array(None)]
                    properties = properties[properties != np.array(None)]
                    if len(properties) > 1:
                        merged_property = self.merge_structures(properties, priorities, max(priorities))
                    else:
                        merged_property = properties[0]
                    modify_volumes(volumes, merged_property, x_idx_px, y_idx_px, z_idx_px)

        return volumes

    def merge_structures(self, properties, priorities, min_priority):
        """
        TODO DO MERGIC
        :param properties:
        :param priorities:
        :param min_priority:
        """
        if min_priority == -1:
            # Are there no structures left and the weights sum to less than 1?
            # Proceed with the merging!
            return merge_properties(properties, priorities, True)

        prio_properties = properties[priorities >= min_priority]
        prio_priorities = priorities[priorities >= min_priority]
        weights = [prop.volume_fraction for prop in prio_properties]

        if np.sum(weights) < 1:
            # If the sum is smaller than one, check if there are more low priority structures!
            return self.merge_structures(properties, priorities, min_priority - 1)
        elif np.abs(np.sum(weights) - 1) < self.EPS:
            # If the sum is exactly one, merge the properties!
            return merge_properties(prio_properties, prio_priorities, True)
        else:
            # If the sum is larger than one, also merge the properties
            return merge_properties(prio_properties, prio_priorities, False)


def merge_properties(properties, priorities, force_unnormalised_merge):
    """
    #TODO
    :param properties:
    :param priorities:
    :param force_unnormalised_merge:
    """

    if force_unnormalised_merge:
        return TissueProperties.weighted_merge(properties)
    else:
        if min(priorities) < max(priorities):
            high_prio_properties = properties[priorities > min(priorities)]
            low_prio_properties = properties[priorities == min(priorities)]
            high_prio_weights = [prop.volume_fraction for prop in high_prio_properties]
            weight_sum = np.sum(high_prio_weights)
            #TODO Error if weight_sum > 1
            high_prio_property = TissueProperties.weighted_merge(high_prio_properties)
            low_prio_property = TissueProperties.weighted_merge(low_prio_properties)
            high_prio_property["weight"] = weight_sum
            low_prio_property["weight"] = 1-weight_sum
            return TissueProperties.weighted_merge([high_prio_property, low_prio_property])
        else:
            return TissueProperties.normalized_merge(properties)


def modify_volumes(volumes, merged_property, x_idx_px, y_idx_px, z_idx_px):
    """
    # TODO
    :param volumes:
    :param merged_property:
    :param x_idx_px:
    :param y_idx_px:
    :param z_idx_px:
    """
    keys = merged_property.keys()
    for key in keys:
        volumes[key][x_idx_px, y_idx_px, z_idx_px] = merged_property[key]
