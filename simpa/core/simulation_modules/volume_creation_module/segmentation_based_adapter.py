# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.core.simulation_modules.volume_creation_module import (
    VolumeCreationAdapterBase,
)
from simpa.utils import Tags
import numpy as np
import torch
from scipy.ndimage import uniform_filter


class SegmentationBasedAdapter(VolumeCreationAdapterBase):
    """
    Creates a volume based on a segmentation map. Inputs can be either a 3D integer label map (each voxel has a single class label)
    or a 4D fraction map (each voxel is 0-1 fraction of each class).
    3D label maps will be one-hot encoded into 4D tensor with dimensions [C, X, Y, Z]
    This creates some memory overhead but allows for easy handling of (pre-computed) fraction maps and partial-volume effects.

    If ``Tags.CONSIDER_PARTIAL_VOLUME`` is ``True``, boundary fractions are smoothed via scipy.ndimage.uniform_filter
    Smoothing can be controlled via ``Tags.PARTIAL_VOLUME_KERNEL_SIZE`` (default 3 -> 1-voxel transition).

    Fractions are normalised to sum to 1 at each voxel, and blended according to the class properties.
    For the resulting segmentation map, the class with the highest fraction at each voxel determines the label.
    Oxygenation is blended only where blood is present, and set to NaN where blood volume fraction is zero.
    """

    def create_simulation_volume(self) -> dict:
        volumes, x_dim_px, y_dim_px, z_dim_px = self.create_empty_volumes()
        wavelength = self.global_settings[Tags.WAVELENGTH]

        raw = self.component_settings[Tags.INPUT_SEGMENTATION_VOLUME]
        class_mapping = self.component_settings[Tags.SEGMENTATION_CLASS_MAPPING]
        # Sort so that blending order is deterministic; lower label = lower priority.
        segmentation_classes = sorted(class_mapping.keys())
        # Channel index == class label, so we need max_label + 1 channels.
        num_channels = max(segmentation_classes) + 1

        # normalise input to 4D float [C, X, Y, Z]
        if np.issubdtype(np.asarray(raw).dtype, np.floating) and np.ndim(raw) == 4:
            # Already 4D float [C, X, Y, Z]: use fractions directly.
            # Channel index must equal label value, so the array needs at least max_label + 1 channels
            if raw.shape[0] < num_channels:
                raise ValueError(
                    f"4D fraction map has {raw.shape[0]} channels but the class mapping "
                    f"requires at least {num_channels} (max label = {max(segmentation_classes)})."
                )
            fractions = torch.tensor(raw, dtype=torch.float, device=self.torch_device)
        elif np.issubdtype(np.asarray(raw).dtype, np.integer) and np.ndim(raw) == 3:
            # 3D integer label map → one-hot encode to 0/1 fractions per channel.
            seg = torch.tensor(np.asarray(raw), device=self.torch_device)
            fractions = torch.zeros(
                num_channels,
                x_dim_px,
                y_dim_px,
                z_dim_px,
                dtype=torch.float,
                device=self.torch_device,
            )
            for c in segmentation_classes:
                fractions[c] = (seg == c).float()

            if self.component_settings.get(Tags.CONSIDER_PARTIAL_VOLUME, False):
                # apply partial volume smoothing
                # (kernel_size - 1) / 2 voxels on each side of a boundary,
                kernel_size = self.component_settings.get(
                    Tags.PARTIAL_VOLUME_KERNEL_SIZE, 3
                )
                for c in segmentation_classes:
                    smoothed = uniform_filter(
                        fractions[c].cpu().numpy(), size=kernel_size
                    )
                    fractions[c] = torch.tensor(
                        smoothed, dtype=torch.float, device=self.torch_device
                    )
        else:
            raise ValueError(
                f"Unsupported input segmentation volume with dtype {raw.dtype} and shape {raw.shape}. "
                "Must be either 3D integer label map or 4D float fraction map."
            )

        # normalise fractions so they sum to 1 at each voxel (Should always hold for 3D label maps)
        # But might not be for 4D fraction maps, and can also be slightly off after smoothing due to boundary effects.
        fraction_sum = torch.zeros(
            x_dim_px, y_dim_px, z_dim_px, dtype=torch.float, device=self.torch_device
        )
        for c in segmentation_classes:
            fraction_sum += fractions[c]

        if (fraction_sum > 0).any():
            safe = fraction_sum.clone()
            safe[safe == 0] = 1.0
            for c in segmentation_classes:
                fractions[c] /= safe

        # Blend properties together.
        # max_class_fractions tracks the dominant class for the segmentation label.
        max_class_fractions = torch.zeros(
            x_dim_px, y_dim_px, z_dim_px, dtype=torch.float, device=self.torch_device
        )

        for seg_class in segmentation_classes:
            class_fractions = fractions[seg_class]
            class_props = class_mapping[seg_class].get_properties_for_wavelength(
                self.global_settings, wavelength
            )

            mask = class_fractions > 0
            if not mask.any():
                continue

            for key in volumes:
                prop = class_props[key]
                if prop is None:
                    continue

                if key == Tags.DATA_FIELD_SEGMENTATION:
                    # Assign the label of the class with the highest fraction at each voxel.
                    better = class_fractions > max_class_fractions
                    if better.any():
                        volumes[key][better] = prop
                        max_class_fractions[better] = class_fractions[better]

                else:
                    # Linear blending, property is weighted by the voxel fraction.
                    if isinstance(prop, (int, float)):
                        volumes[key][mask] += class_fractions[mask] * prop
                    elif isinstance(prop, torch.Tensor) and prop.dim() == 3:
                        volumes[key][mask] += (
                            class_fractions[mask] * prop.to(self.torch_device)[mask]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported property type for '{key}': {type(prop)}"
                        )

        # if no blood is present, oxygenation is undefined
        bvf = volumes.get(Tags.DATA_FIELD_BLOOD_VOLUME_FRACTION)
        if bvf is not None:
            volumes[Tags.DATA_FIELD_OXYGENATION][bvf == 0] = torch.nan

        for key in volumes:
            volumes[key] = volumes[key].cpu().numpy().astype(np.float64, copy=False)

        return volumes
