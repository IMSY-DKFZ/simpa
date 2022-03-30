# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags, Settings
from simpa.utils.tissue_properties import TissueProperties
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.core.device_digital_twins import DigitalDeviceTwinBase, PhotoacousticDevice
import numpy as np


class FieldOfViewCropping(ProcessingComponent):

    def __init__(self, global_settings, settings_key=None):
        if settings_key is None:
            # TODO Extract from global settings all the fields that should be cropped
            global_settings["FieldOfViewCropping"] = Settings({
                      Tags.DATA_FIELD: TissueProperties.property_tags +
                                           [Tags.DATA_FIELD_FLUENCE,
                                            Tags.DATA_FIELD_INITIAL_PRESSURE]})
        super(FieldOfViewCropping, self).__init__(global_settings, "FieldOfViewCropping")
    """
    Applies Gaussian noise to the defined data field.
    The noise will be applied to all wavelengths.
    Component Settings
       **Tags.DATA_FIELD required
    """

    def run(self, device: DigitalDeviceTwinBase):
        self.logger.info("Cropping field of view...")

        if Tags.DATA_FIELD not in self.component_settings.keys():
            msg = f"The field {Tags.DATA_FIELD} must be set in order to use the fov cropping."
            self.logger.critical(msg)
            raise KeyError(msg)

        if not isinstance(self.component_settings[Tags.DATA_FIELD], list):
            msg = f"The field {Tags.DATA_FIELD} must be of type list."
            self.logger.critical(msg)
            raise TypeError(msg)

        data_fields = self.component_settings[Tags.DATA_FIELD]

        if isinstance(device, PhotoacousticDevice):
            field_of_view_mm = device.detection_geometry.get_field_of_view_mm()
        else:
            field_of_view_mm = device.get_field_of_view_mm()
        self.logger.debug(f"FOV (mm): {field_of_view_mm}")
        field_of_view_voxels = (field_of_view_mm / self.global_settings[Tags.SPACING_MM]).astype(np.int32)
        self.logger.debug(f"FOV (voxels): {field_of_view_voxels}")

        # In case it should be cropped from A to A, then crop from A to A+1
        x_offset_correct = 1 if (field_of_view_voxels[1] - field_of_view_voxels[0]) < 1 else 0
        y_offset_correct = 1 if (field_of_view_voxels[3] - field_of_view_voxels[2]) < 1 else 0
        z_offset_correct = 1 if (field_of_view_voxels[5] - field_of_view_voxels[4]) < 1 else 0

        self.logger.debug(f"field of view to crop: {field_of_view_voxels}")

        for data_field in data_fields:
            self.logger.debug(f"Cropping data field {data_field}...")

            # load
            wavelength = self.global_settings[Tags.WAVELENGTH]
            data_array = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)
            self.logger.debug(f"data array shape before cropping: {np.shape(data_array)}")
            self.logger.debug(f"data array shape len: {len(np.shape(data_array))}")

            # input validation
            if not isinstance(data_array, np.ndarray):
                self.logger.warning(f"The data field {data_field} was not of type np.ndarray. Skipping...")
                continue
            data_field_shape = np.shape(data_array)
            if len(data_field_shape) == 3:
                if ((np.array([field_of_view_voxels[1] - field_of_view_voxels[0],
                              field_of_view_voxels[3] - field_of_view_voxels[2],
                              field_of_view_voxels[5] - field_of_view_voxels[4]]) - data_field_shape) == 0).all():
                    self.logger.warning(f"The data field {data_field} is already cropped. Skipping...")
                    continue

                # crop
                data_array = np.squeeze(data_array[field_of_view_voxels[0]:field_of_view_voxels[1] + x_offset_correct,
                                        field_of_view_voxels[2]:field_of_view_voxels[3] + y_offset_correct,
                                        field_of_view_voxels[4]:field_of_view_voxels[5] + z_offset_correct])

            elif len(data_field_shape) == 2:
                # Assumption that the data field is already in 2D shape in the y-plane
                if (np.array([field_of_view_voxels[1] - field_of_view_voxels[0],
                              field_of_view_voxels[5] - field_of_view_voxels[4]]) - data_field_shape == 0).all():
                    self.logger.warning(f"The data field {data_field} is already cropped. Skipping...")
                    continue

                # crop
                data_array = np.squeeze(data_array[field_of_view_voxels[0]:field_of_view_voxels[1] + x_offset_correct,
                                        field_of_view_voxels[4]:field_of_view_voxels[5] + z_offset_correct])

            self.logger.debug(f"data array shape after cropping: {np.shape(data_array)}")
            # save
            save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Cropping field of view...[Done]")
