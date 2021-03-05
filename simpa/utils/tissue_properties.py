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

from simpa.utils import Tags


class TissueProperties(dict):

    property_tags = [Tags.PROPERTY_ABSORPTION_PER_CM,
                     Tags.PROPERTY_SCATTERING_PER_CM,
                     Tags.PROPERTY_ANISOTROPY,
                     Tags.PROPERTY_GRUNEISEN_PARAMETER,
                     Tags.PROPERTY_SEGMENTATION,
                     Tags.PROPERTY_OXYGENATION,
                     Tags.PROPERTY_DENSITY,
                     Tags.PROPERTY_SPEED_OF_SOUND,
                     Tags.PROPERTY_ALPHA_COEFF]

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
