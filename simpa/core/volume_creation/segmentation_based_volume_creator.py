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
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags
from simpa.utils.tissue_properties import TissueProperties
import numpy as np


class SegmentationBasedVolumeCreator(VolumeCreatorBase):
    """
    This volume creator expects a np.ndarray to be in the settigs
    under the Tags.INPUT_SEGMENTATION_VOLUME tag and uses this array
    together with a SegmentationClass mapping which is a dict defined in
    the settings under Tags.SEGMENTATION_CLASS_MAPPING.

    With this, an even greater utility is warranted.
    """

    def create_simulation_volume(self, settings: Settings) -> dict:
        volumes, x_dim_px, y_dim_px, z_dim_px = self.create_empty_volumes(settings)
        wavelength = settings[Tags.WAVELENGTH]

        segmentation_volume = settings[Tags.INPUT_SEGMENTATION_VOLUME]
        segmentation_classes = np.unique(segmentation_volume, return_counts=False)
        x_dim_seg_px, y_dim_seg_px, z_dim_seg_px = np.shape(segmentation_volume)

        if x_dim_px != x_dim_seg_px:
            raise ValueError("x_dim of volumes and segmentation must perfectly match but was {} and {}"
                             .format(x_dim_px, x_dim_seg_px))
        if y_dim_px != y_dim_seg_px:
            raise ValueError("y_dim of volumes and segmentation must perfectly match but was {} and {}"
                             .format(y_dim_px, y_dim_seg_px))
        if z_dim_px != z_dim_seg_px:
            raise ValueError("z_dim of volumes and segmentation must perfectly match but was {} and {}"
                             .format(z_dim_px, z_dim_seg_px))

        class_mapping = settings[Tags.SEGMENTATION_CLASS_MAPPING]

        for seg_class in segmentation_classes:
            class_properties = class_mapping[seg_class].get_properties_for_wavelength(wavelength)
            for prop_tag in TissueProperties.property_tags:
                volumes[prop_tag][segmentation_volume == seg_class] = class_properties[prop_tag]

        return volumes
