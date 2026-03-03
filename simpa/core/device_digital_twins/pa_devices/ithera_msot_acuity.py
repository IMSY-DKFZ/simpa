# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F

from simpa.core.device_digital_twins import PhotoacousticDevice, \
    CurvedArrayDetectionGeometry, MSOTAcuityIlluminationGeometry

from simpa.core.device_digital_twins.pa_devices import PhotoacousticDevice
from simpa.core.device_digital_twins.detection_geometries.curved_array import CurvedArrayDetectionGeometry
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.libraries.tissue_library import TissueLibrary
import numpy as np

class MSOTAcuityEcho(PhotoacousticDevice):
    """
    This class represents a digital twin of the MSOT Acuity Echo, manufactured by iThera Medical, Munich, Germany
    (https://www.ithera-medical.com/products/msot-acuity/). It is based on the real specifications of the device, but
    due to the limitations of the possibilities how to represent a device in the software frameworks,
    constitutes only an approximation.

    The origin for this device is the center of the membrane at the point of contact between the membrane and the
    tissue, i.e. the outer center of the membrane.

    Some important publications that showcase the use cases of the MSOT Acuity and Acuity Echo device are::

        Regensburger, Adrian P., et al. "Detection of collagens by multispectral optoacoustic
        tomography as an imaging biomarker for Duchenne muscular dystrophy."
        Nature Medicine 25.12 (2019): 1905-1915.

        Knieling, Ferdinand, et al. "Multispectral Optoacoustic Tomography for Assessment of
        Crohn's Disease Activity."
        The New England journal of medicine 376.13 (2017): 1292.

    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(MSOTAcuityEcho, self).__init__(device_position_mm=device_position_mm)

        self.mediprene_membrane_height_mm = 1
        self.probe_height_mm = 43.2
        self.focus_in_field_of_view_mm = 8
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0, self.focus_in_field_of_view_mm]))

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       0, 0, 0, 50])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

        self.field_of_view_extent_mm[4] -= self.focus_in_field_of_view_mm
        self.field_of_view_extent_mm[5] -= self.focus_in_field_of_view_mm

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=153,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=self.field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        illumination_geometry = MSOTAcuityIlluminationGeometry()

        # y position relative to the membrane:
        # The laser is located 43.2 mm  behind the membrane with an angle of 22.4 degrees.
        y_pos_relative_to_membrane = np.tan(np.deg2rad(22.4)) * 43.2
        self.add_illumination_geometry(illumination_geometry,
                                       illuminator_position_relative_to_pa_device=np.array([0,
                                                                                            -y_pos_relative_to_membrane,
                                                                                            -43.2]))

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings):
        """
        Updates the volume creation settings of the model based volume creator according to the size of the device.
        :param global_settings: Settings for the entire simulation pipeline.
        :type global_settings: Settings
        """
        try:
            volume_creator_settings = Settings(global_settings.get_volume_creation_settings())
        except KeyError as e:
            self.logger.warning("You called the update_settings_for_use_of_model_based_volume_creator method "
                                "even though there are no volume creation settings defined in the "
                                "settings dictionary.")
            return

        probe_size_mm = self.probe_height_mm
        mediprene_layer_height_mm = self.mediprene_membrane_height_mm
        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm
        spacing_mm = global_settings[Tags.SPACING_MM]
        old_volume_height_pixels = round(global_settings[Tags.DIM_VOLUME_Z_MM] / spacing_mm)

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_thickness = np.random.normal(0.4, 0.1)
        else:
            us_gel_thickness = 0

        z_dim_position_shift_mm = mediprene_layer_height_mm + heavy_water_layer_height_mm + us_gel_thickness

        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + z_dim_position_shift_mm

        # adjust the z-dim to msot probe height
        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        # adjust the x-dim to msot probe width
        # 1 voxel is added (0.5 on both sides) to make sure no rounding errors lead to a detector element being outside
        # of the simulated volume.

        if global_settings[Tags.DIM_VOLUME_X_MM] < round(self.detection_geometry.probe_width_mm) + spacing_mm:
            width_shift_for_structures_mm = (round(self.detection_geometry.probe_width_mm) + spacing_mm -
                                             global_settings[Tags.DIM_VOLUME_X_MM]) / 2
            global_settings[Tags.DIM_VOLUME_X_MM] = round(self.detection_geometry.probe_width_mm) + spacing_mm
            self.logger.debug(f"Changed Tags.DIM_VOLUME_X_MM to {global_settings[Tags.DIM_VOLUME_X_MM]}")
        else:
            width_shift_for_structures_mm = 0

        self.logger.debug(volume_creator_settings)

        for structure_key in volume_creator_settings[Tags.STRUCTURES]:
            self.logger.debug("Adjusting " + str(structure_key))
            structure_dict = volume_creator_settings[Tags.STRUCTURES][structure_key]
            if Tags.STRUCTURE_START_MM in structure_dict:
                for molecule in structure_dict[Tags.MOLECULE_COMPOSITION]:
                    old_volume_fraction = getattr(molecule, "volume_fraction")
                    if isinstance(old_volume_fraction, torch.Tensor):
                        if old_volume_fraction.shape[2] == old_volume_height_pixels:
                            width_shift_pixels = round(width_shift_for_structures_mm / spacing_mm)
                            z_shift_pixels = round(z_dim_position_shift_mm / spacing_mm)
                            padding_height = (z_shift_pixels, 0, 0, 0, 0, 0)
                            padding_width = ((width_shift_pixels, width_shift_pixels), (0, 0), (0, 0))
                            padded_up = F.pad(old_volume_fraction, padding_height, mode='constant', value=0)
                            padded_vol = np.pad(padded_up.numpy(), padding_width, mode='edge')
                            setattr(molecule, "volume_fraction", torch.from_numpy(padded_vol))
                structure_dict[Tags.STRUCTURE_START_MM][0] = structure_dict[Tags.STRUCTURE_START_MM][
                    0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_START_MM][2] = structure_dict[Tags.STRUCTURE_START_MM][
                    2] + z_dim_position_shift_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][0] = structure_dict[Tags.STRUCTURE_END_MM][
                    0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][
                    2] + z_dim_position_shift_mm

        if Tags.CONSIDER_PARTIAL_VOLUME_IN_DEVICE in volume_creator_settings:
            consider_partial_volume = volume_creator_settings[Tags.CONSIDER_PARTIAL_VOLUME_IN_DEVICE]
        else:
            consider_partial_volume = False

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_layer_settings = Settings({
                Tags.PRIORITY: 5,
                Tags.STRUCTURE_START_MM: [0, 0,
                                          heavy_water_layer_height_mm + mediprene_layer_height_mm],
                Tags.STRUCTURE_END_MM: [0, 0,
                                        heavy_water_layer_height_mm + mediprene_layer_height_mm + us_gel_thickness],
                Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
                Tags.MOLECULE_COMPOSITION: TissueLibrary.ultrasound_gel(),
                Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
            })

            volume_creator_settings[Tags.STRUCTURES]["us_gel"] = us_gel_layer_settings

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 5,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
            Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
            Tags.MOLECULE_COMPOSITION: TissueLibrary.mediprene(),
            Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
        })

        volume_creator_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer_settings

        self.device_position_mm = np.add(self.device_position_mm, np.array([width_shift_for_structures_mm, 0,
                                                                            probe_size_mm]))
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0,
                                                                   self.focus_in_field_of_view_mm]))
        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=153,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=self.field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        for illumination_geom in self.illumination_geometries:
            illumination_geom.device_position_mm = np.add(illumination_geom.device_position_mm,
                                                          np.array([width_shift_for_structures_mm, 0, probe_size_mm]))

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TissueLibrary.heavy_water(),
            Tags.STRUCTURE_TYPE: Tags.BACKGROUND
        })
        volume_creator_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background_settings


    def update_settings_for_use_of_segmentation_based_volume_creator(
        self,
        global_settings: Settings,
    ):
        """
        Updates the volume creation settings of the segmentation based volume creator according to the size of the
        device.  Adds a heavy-water layer, a mediprene membrane layer, and optionally a US-gel layer (controlled via Tags.US_GEL in the volume creation settings).
        Only necessary if the segmentation map does not already include these layers.

        Supports both input formats accepted by ``SegmentationBasedAdapter``:

        * 3D integer label map``[X, Y, Z]``: new z-voxels are filled with the new integer label.
        * 4D float fraction map ``[C, X, Y, Z]``: existing channels are zero-padded in z and a new
          channel (index == new label) is appended with fraction 1.0 in the new z-slice.
          This requires labels to be dense (0...N-1) so that the channel index equals the label.

        :param global_settings: Settings for the entire simulation pipeline.
        """
        try:
            volume_creator_settings = Settings(
                global_settings.get_volume_creation_settings()
            )
        except KeyError:
            self.logger.warning(
                "You called the update_settings_for_use_of_segmentation_based_volume_creator method "
                "even though there are no volume creation settings defined in the "
                "settings dictionary."
            )
            return

        segmentation_map = volume_creator_settings[Tags.INPUT_SEGMENTATION_VOLUME]
        segmentation_class_mapping = volume_creator_settings[Tags.SEGMENTATION_CLASS_MAPPING]
        spacing_mm = global_settings[Tags.SPACING_MM]
        z_dim_position_shift_mm = 0

        def _prepend_layer(seg, n_px, label):
            """Prepend n_px voxels along z for a given label.

            3D [X,Y,Z]: fills the new voxels with the integer label.
            4D [C,X,Y,Z]: pads existing channels with 0 in z, then appends a new channel
            (index == label) with 1.0 in the new z-slice.  Labels must be dense so that
            label == seg.shape[0] (i.e. the new channel index matches the label value).
            """
            if seg.ndim == 3:
                return np.pad(seg, ((0, 0), (0, 0), (n_px, 0)), mode="constant", constant_values=label)
            else:  # 4D [C, X, Y, Z]
                # Zero-pad all existing class channels in z (new region belongs to a new class)
                padded = np.pad(seg, ((0, 0), (0, 0), (0, 0), (n_px, 0)), mode="constant", constant_values=0)
                # Append new channel: fraction 1.0 in the prepended slice, 0.0 everywhere else
                new_ch = np.zeros((1, seg.shape[1], seg.shape[2], padded.shape[3]), dtype=padded.dtype)
                new_ch[0, :, :, :n_px] = 1.0
                return np.concatenate([padded, new_ch], axis=0)

        # Allocate new integer labels for the device layers by incrementing past the highest
        # existing label, so they never collide with user-defined segmentation classes.
        used_labels = set(int(l) for l in segmentation_class_mapping.keys())

        def _next_unused_label():
            label = max(used_labels) + 1
            used_labels.add(label)
            return label

        # --- US gel ---
        # If specified, add layer of US gel
        # Thickness is drawn from a normal distribution (like in model-based volume creation case).
        if volume_creator_settings.get(Tags.US_GEL, False):
            us_gel_thickness_mm = np.random.normal(0.4, 0.1)
            us_gel_thickness_px = int(round(us_gel_thickness_mm / spacing_mm))
            us_gel_label = _next_unused_label()
            segmentation_map = _prepend_layer(segmentation_map, us_gel_thickness_px, us_gel_label)
            segmentation_class_mapping[us_gel_label] = TissueLibrary.ultrasound_gel()
            z_dim_position_shift_mm += us_gel_thickness_px * spacing_mm
            self.logger.debug("Added an ultrasound gel layer to the segmentation map.")

        # --- Mediprene membrane ---
        # Fixed physical thickness of the membrane (self.mediprene_membrane_height_mm).
        mediprene_thickness_px = int(round(self.mediprene_membrane_height_mm / spacing_mm))
        mediprene_label = _next_unused_label()
        segmentation_map = _prepend_layer(segmentation_map, mediprene_thickness_px, mediprene_label)
        segmentation_class_mapping[mediprene_label] = TissueLibrary.mediprene()
        z_dim_position_shift_mm += mediprene_thickness_px * spacing_mm
        self.logger.debug("Added a mediprene layer to the segmentation map.")

        # --- Heavy water ---
        # Fills the rest of the probe with heavy water (probe height minus the membrane thickness).
        heavy_water_label = _next_unused_label()
        segmentation_class_mapping[heavy_water_label] = TissueLibrary.heavy_water()
        heavy_water_layer_height_mm = self.probe_height_mm - self.mediprene_membrane_height_mm
        heavy_water_layer_height_px = int(round(heavy_water_layer_height_mm / spacing_mm))
        segmentation_map = _prepend_layer(segmentation_map, heavy_water_layer_height_px, heavy_water_label)
        z_dim_position_shift_mm += heavy_water_layer_height_px * spacing_mm
        self.logger.debug(
            f"Added a {heavy_water_layer_height_px * spacing_mm:.2f} mm heavy water layer to the segmentation map."
        )

        # Update global z-dimension 
        global_settings[Tags.DIM_VOLUME_Z_MM] += z_dim_position_shift_mm
        self.logger.debug(f"Changed Tags.DIM_VOLUME_Z_MM to {global_settings[Tags.DIM_VOLUME_Z_MM]}")

        # Widen the volume along x if narrower than the probe footprint
        # One extra voxel (0.5 on each side) guards against rounding errors
        if global_settings[Tags.DIM_VOLUME_X_MM] < round(self.detection_geometry.probe_width_mm) + spacing_mm:
            width_shift_for_structures_mm = (
                round(self.detection_geometry.probe_width_mm)
                + spacing_mm
                - global_settings[Tags.DIM_VOLUME_X_MM]
            ) / 2
            # Split the total pixel shift symmetrically left/right.
            total_shift_pixels = int(round(2 * width_shift_for_structures_mm / spacing_mm))
            left_shift_pixels = total_shift_pixels // 2
            right_shift_pixels = total_shift_pixels - left_shift_pixels
            # For the segmentation map itself, x is axis 0 for 3D but axis 1 for 4D [C,X,Y,Z].
            seg_padding_width = (
                ((0, 0), (left_shift_pixels, right_shift_pixels), (0, 0), (0, 0))
                if np.ndim(segmentation_map) == 4
                else ((left_shift_pixels, right_shift_pixels), (0, 0), (0, 0))
            )
            segmentation_map = np.pad(segmentation_map, seg_padding_width, mode="edge")
            global_settings[Tags.DIM_VOLUME_X_MM] = int(round(self.detection_geometry.probe_width_mm)) + spacing_mm
            self.logger.debug(
                f"Changed Tags.DIM_VOLUME_X_MM to {global_settings[Tags.DIM_VOLUME_X_MM]}, and expanded "
                f"the segmentation map accordingly using edge padding"
            )
        else:
            width_shift_for_structures_mm = 0

        # Write the modified segmentation map back into the settings.
        global_settings[Tags.VOLUME_CREATION_MODEL_SETTINGS][Tags.INPUT_SEGMENTATION_VOLUME] = segmentation_map
        self.logger.debug("The segmentation volume has been adjusted to fit the MSOT device")

        # Shift device, detector, and illuminator positions
        # The probe layers are prepended at z=0, pushing the tissue surface downward by probe_height_mm.
        # The x-shift centres the device over the (potentially widened) volume.
        self.device_position_mm = np.add(
            self.device_position_mm,
            np.array([width_shift_for_structures_mm, 0, self.probe_height_mm]),
        )
        self.detection_geometry_position_vector = np.add(
            self.device_position_mm, np.array([0, 0, self.focus_in_field_of_view_mm])
        )
        detection_geometry = CurvedArrayDetectionGeometry(
            pitch_mm=0.34,
            radius_mm=40,
            number_detector_elements=256,
            detector_element_width_mm=0.24,
            detector_element_length_mm=13,
            center_frequency_hz=3.96e6,
            bandwidth_percent=153,
            sampling_frequency_mhz=40,
            angular_origin_offset=np.pi,
            device_position_mm=self.detection_geometry_position_vector,
            field_of_view_extent_mm=self.field_of_view_extent_mm,
        )
        self.set_detection_geometry(detection_geometry)
        for illumination_geom in self.illumination_geometries:
            illumination_geom.device_position_mm = np.add(
                illumination_geom.device_position_mm,
                np.array([width_shift_for_structures_mm, 0, self.probe_height_mm]),
            )

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"MSOTAcuityEcho": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityEcho()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
