# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils.constants import property_tags
from simpa.utils import Settings
import torch


class TissueProperties(dict):
    """
    The tissue properties contain a volumetric representation of each tissue parameter currently
    modelled in the SIMPA framework.

    It is a dictionary that is populated with each of the parameters.
    The values of the parameters can be either numbers or numpy arrays.
    It also contains a volume fraction field.
    """

    def __init__(self, settings: Settings):
        super().__init__()
        volume_x_dim, volume_y_dim, volume_z_dim = settings.get_volume_dimensions_voxels()
        self.volume_fraction = torch.zeros((volume_x_dim, volume_y_dim, volume_z_dim), dtype=torch.float32)
        for key in property_tags:
            self[key] = 0

    @staticmethod
    def normalized_merge(property_list: list):
        return_property = TissueProperties.weighted_merge(property_list)
        for key in property_tags:
            return_property[key] = return_property[key] / return_property.volume_fraction
        return return_property

    @staticmethod
    def weighted_merge(property_list: list):
        return_property = TissueProperties()
        for target_property in property_list:
            for key in property_tags:
                if target_property[key] is not None:
                    return_property[key] += target_property.volume_fraction * target_property[key]
            return_property.volume_fraction += target_property.volume_fraction
        return return_property
