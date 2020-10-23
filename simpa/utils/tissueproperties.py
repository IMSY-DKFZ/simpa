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

from simpa.utils import Tags


class TissueProperties(dict):

    keys = [Tags.PROPERTY_ABSORPTION_PER_CM,
            Tags.PROPERTY_SCATTERING_PER_CM,
            Tags.PROPERTY_ANISOTROPY,
            Tags.PROPERTY_SEGMENTATION,
            Tags.PROPERTY_OXYGENATION,
            Tags.PROPERTY_DENSITY,
            Tags.PROPERTY_SPEED_OF_SOUND,
            Tags.PROPERTY_ALPHA_COEFF]

    def __init__(self):
        super().__init__()
        self.weight = None
        for key in TissueProperties.keys:
            self[key] = None

    @staticmethod
    def normalized_merge(property_list: list):
        return_property = TissueProperties.weighted_merge(property_list)
        for key in return_property.keys:
            return_property[key] = return_property[key] / return_property.weight
        return return_property

    @staticmethod
    def weighted_merge(property_list: list):
        return_property = TissueProperties()
        for target_property in property_list:
            for key in return_property.keys:
                return_property[key] = target_property.weight * target_property[key]
            return_property.weight += target_property.weight
        return return_property
