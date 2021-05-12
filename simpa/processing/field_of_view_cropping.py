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

from simpa.utils import Tags, Settings
from simpa.utils.tissue_properties import TissueProperties
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.simulation_components import ProcessingComponent
from simpa.core.device_digital_twins import DigitalDeviceTwinBase
import numpy as np


class FieldOfViewCroppingProcessingComponent(ProcessingComponent):

    def __init__(self, global_settings, settings_key=None):
        if settings_key is None:
            global_settings["FieldOfViewCroppingProcessingComponent"] = Settings({
                      Tags.DATA_FIELD: TissueProperties.property_tags +
                                           [Tags.OPTICAL_MODEL_FLUENCE,
                                            Tags.OPTICAL_MODEL_INITIAL_PRESSURE]})
        super(FieldOfViewCroppingProcessingComponent, self).__init__(global_settings,
                                                                     "FieldOfViewCroppingProcessingComponent")
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

        field_of_view_mm = device.get_field_of_view_mm(self.global_settings)
        field_of_view_voxels = (field_of_view_mm / self.global_settings[Tags.SPACING_MM]).astype(np.int)

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
            if len(data_field_shape) != 3:
                self.logger.warning(f"The data field {data_field} was not three-dimensional. Skipping...")
                continue

            # crop
            data_array = np.squeeze(data_array[field_of_view_voxels[0]:field_of_view_voxels[1]+1,
                                    field_of_view_voxels[2]:field_of_view_voxels[3]+1,
                                    field_of_view_voxels[4]:field_of_view_voxels[5]+1])

            self.logger.debug(f"data array shape after cropping: {np.shape(data_array)}")
            # save
            save_data_field(data_array, self.global_settings[Tags.SIMPA_OUTPUT_PATH], data_field, wavelength)

        self.logger.info("Cropping field of view...[Done]")