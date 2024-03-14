# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils.constants import property_tags


class TissueProperties(dict):

    def __init__(self):
        super().__init__()
        self.volume_fraction = 0
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
