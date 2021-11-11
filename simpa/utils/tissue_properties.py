# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags


class TissueProperties(dict):

    property_tags = [Tags.DATA_FIELD_ABSORPTION_PER_CM,
                     Tags.DATA_FIELD_SCATTERING_PER_CM,
                     Tags.DATA_FIELD_ANISOTROPY,
                     Tags.DATA_FIELD_GRUNEISEN_PARAMETER,
                     Tags.DATA_FIELD_SEGMENTATION,
                     Tags.DATA_FIELD_OXYGENATION,
                     Tags.DATA_FIELD_DENSITY,
                     Tags.DATA_FIELD_SPEED_OF_SOUND,
                     Tags.DATA_FIELD_ALPHA_COEFF]

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
