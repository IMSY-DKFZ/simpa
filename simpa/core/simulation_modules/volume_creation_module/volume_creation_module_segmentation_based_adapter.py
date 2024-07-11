# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.volume_creation_module import VolumeCreatorModuleBase
from simpa.utils import Tags
from simpa.utils.constants import property_tags
from simpa.io_handling import save_hdf5
import numpy as np
import torch


class SegmentationBasedVolumeCreationAdapter(VolumeCreatorModuleBase):
    """
    This volume creator expects a np.ndarray to be in the settings
    under the Tags.INPUT_SEGMENTATION_VOLUME tag and uses this array
    together with a SegmentationClass mapping which is a dict defined in
    the settings under Tags.SEGMENTATION_CLASS_MAPPING.

    With this, an even greater utility is warranted.
    """

    def create_simulation_volume(self) -> dict:
        volumes, x_dim_px, y_dim_px, z_dim_px = self.create_empty_volumes()
        wavelength = self.global_settings[Tags.WAVELENGTH]

        segmentation_volume = (np.char.mod(
            '%s', self.component_settings[Tags.INPUT_SEGMENTATION_VOLUME].astype(np.int64)))
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

        class_mapping = self.component_settings[Tags.SEGMENTATION_CLASS_MAPPING]

        for seg_class in segmentation_classes:
            class_properties = class_mapping[seg_class].get_properties_for_wavelength(self.global_settings, wavelength)
            for prop_tag in property_tags:
                if isinstance(class_properties[prop_tag], (int, float)) or class_properties[prop_tag] == None:  # scalar
                    assigned_prop = class_properties[prop_tag]
                    if assigned_prop is None:
                        assigned_prop = torch.nan
                    volumes[prop_tag][segmentation_volume == seg_class] = assigned_prop
                elif len(torch.Tensor.size(class_properties[prop_tag])) == 3:  # 3D map
                    assigned_prop = class_properties[prop_tag][torch.tensor(segmentation_volume == seg_class)]
                    assigned_prop[assigned_prop is None] = torch.nan
                    volumes[prop_tag][torch.tensor(segmentation_volume == seg_class)] = assigned_prop
                else:
                    raise AssertionError("Properties need to either be a scalar or a 3D map.")

        save_hdf5(self.global_settings, self.global_settings[Tags.SIMPA_OUTPUT_PATH], "/settings/")

        # convert volumes back to CPU
        for key in volumes.keys():
            volumes[key] = volumes[key].cpu().numpy().astype(np.float64, copy=False)

        return volumes
