# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.utils.constants import wavelength_independent_properties, wavelength_dependent_properties


class TissueProperties(dict):

    property_tags = wavelength_dependent_properties + wavelength_independent_properties

    def __init__(self):
        super().__init__()
        self.volume_fraction = 0
        for key in TissueProperties.property_tags:
            self[key] = 0

    @staticmethod
    def normalized_merge(property_list: list):
        return_property = TissueProperties.weighted_merge(property_list)
        for key in return_property.property_tags:
            return_property[key] = return_property[key] / return_property.volume_fraction
        return return_property

    @staticmethod
    def weighted_merge(property_list: list):
        return_property = TissueProperties()
        for target_property in property_list:
            for key in return_property.property_tags:
                if target_property[key] is not None:
                    return_property[key] += target_property.volume_fraction * target_property[key]
            return_property.volume_fraction += target_property.volume_fraction
        return return_property
