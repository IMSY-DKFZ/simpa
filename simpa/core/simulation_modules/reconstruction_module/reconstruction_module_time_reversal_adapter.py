# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.utils.settings import Settings
from simpa.core.simulation_modules.reconstruction_module import ReconstructionAdapterBase
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry
import numpy as np
import scipy.io as sio
import subprocess
import os
import inspect


class TimeReversalAdapter(ReconstructionAdapterBase):
    """
    The time reversal adapter includes the time reversal reconstruction
    algorithm implemented by the k-Wave toolkit into SIMPA.

    Time reversal reconstruction uses the time series data and computes the forward simulation model
    backwards in time::

        Treeby, Bradley E., Edward Z. Zhang, and Benjamin T. Cox.
        "Photoacoustic tomography in absorbing acoustic media using
        time reversal." Inverse Problems 26.11 (2010): 115003.


    """

    def get_acoustic_properties(self, input_data: dict, detection_geometry):
        """
        This method extracts the acoustic tissue properties from the settings dictionary and
        amends the information to the input_data.

        :param input_data: a dictionary containing the information needed for time reversal.
        :param detection_geometry: PA device that is used for reconstruction
        """

        if Tags.ACOUSTIC_SIMULATION_3D not in self.component_settings or not \
                self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            axes = (0, 1)
        else:
            axes = (0, 2)

        pa_device = detection_geometry
        pa_device.check_settings_prerequisites(self.global_settings)

        # spacing
        if Tags.SPACING_MM in self.component_settings and self.component_settings[Tags.SPACING_MM]:
            spacing_in_mm = self.component_settings[Tags.SPACING_MM]
        elif Tags.SPACING_MM in self.global_settings and self.global_settings[Tags.SPACING_MM]:
            spacing_in_mm = self.global_settings[Tags.SPACING_MM]
        else:
            raise AttributeError("Please specify a value for SPACING_MM")

        detector_positions = detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
        detector_positions_voxels = np.round(detector_positions / spacing_in_mm).astype(int)

        volume_x_dim = int(np.ceil(self.global_settings[Tags.DIM_VOLUME_X_MM] / spacing_in_mm) + 1)   # plus 2 because of off-
        volume_y_dim = int(np.ceil(self.global_settings[Tags.DIM_VOLUME_Y_MM] / spacing_in_mm) + 1)      # by-one error in matlab
        volume_z_dim = int(np.ceil(self.global_settings[Tags.DIM_VOLUME_Z_MM] / spacing_in_mm) + 1)      # otherwise

        if Tags.ACOUSTIC_SIMULATION_3D not in self.component_settings or not \
                self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            sizes = (volume_z_dim, volume_x_dim)
            sensor_map = np.zeros(sizes)
            sensor_map[detector_positions_voxels[:, 2]+1, detector_positions_voxels[:, 0]+1] = 1
        else:
            sizes = (volume_z_dim, volume_y_dim, volume_x_dim)
            sensor_map = np.zeros(sizes)
            sensor_map[detector_positions_voxels[:, 2]+1,
                       detector_positions_voxels[:, 1]+1,
                       detector_positions_voxels[:, 0]+1] = 1

        # check that the spacing is large enough for all detector elements to be on the sensor map
        det_elements_sensor_map = np.count_nonzero(sensor_map)
        if det_elements_sensor_map != pa_device.number_detector_elements:
            raise AttributeError("The spacing is too large to fit every detector element on the sensor map."
                                 "Please increase it! "
                                 f"Expected {pa_device.number_detector_elements} elements but it "
                                 f"were {det_elements_sensor_map}.")

        # TODO: Include possibility to
        possible_acoustic_properties = [Tags.DATA_FIELD_SPEED_OF_SOUND,
                                        Tags.DATA_FIELD_DENSITY,
                                        Tags.DATA_FIELD_ALPHA_COEFF
                                        ]
        input_data[Tags.KWAVE_PROPERTY_SENSOR_MASK] = sensor_map

        for acoustic_property in possible_acoustic_properties:
            if acoustic_property in self.component_settings:
                try:
                    input_data[acoustic_property] = self.component_settings[acoustic_property]
                except ValueError or KeyError:
                    self.logger.error("{} not specified.".format(acoustic_property))

        return input_data, spacing_in_mm

    def reorder_time_series_data(self, time_series_sensor_data, detection_geometry):
        """
        Reorders the time series data to match the order that is assumed by kwave
        during image reconstruction with TimeReversal.

        The main issue here is, that, while forward modelling allows for the definition of
        3D cuboid bounding boxes for the detector elements, TimeReversal does not implement
        this feature.
        Instead, a binary mask is given and these are indexed in a column-row-wise manner in
        the output.
        The default np.argsort() method does not yield the same result as expected by
        k-Wave. Hence, this workaround.
        """

        detector_positions = detection_geometry.get_detector_element_positions_base_mm()
        angles = np.arctan2(detector_positions[:, 2], detector_positions[:, 0])
        matlab_order = np.argsort(angles)
        return time_series_sensor_data[matlab_order]

    def reconstruction_algorithm(self, time_series_sensor_data, detection_geometry):
        input_data = dict()

        # If the detecttion_geometry is something else than linear, the time series data have to be reordered for matlab
        if not isinstance(detection_geometry, LinearArrayDetectionGeometry):
            time_series_sensor_data = self.reorder_time_series_data(time_series_sensor_data, detection_geometry)

        input_data[Tags.DATA_FIELD_TIME_SERIES_DATA] = time_series_sensor_data
        input_data, spacing_in_mm = self.get_acoustic_properties(input_data, detection_geometry)
        acoustic_path = self.global_settings[Tags.SIMPA_OUTPUT_PATH] + ".mat"

        possible_k_wave_parameters = [Tags.MODEL_SENSOR_FREQUENCY_RESPONSE,
                                      Tags.KWAVE_PROPERTY_ALPHA_POWER, Tags.GPU, Tags.KWAVE_PROPERTY_PMLInside, Tags.KWAVE_PROPERTY_PMLAlpha, Tags.KWAVE_PROPERTY_PlotPML,
                                      Tags.RECORDMOVIE, Tags.MOVIENAME,
                                      Tags.SENSOR_DIRECTIVITY_PATTERN]

        pa_device = detection_geometry
        k_wave_settings = Settings({
            Tags.SENSOR_NUM_ELEMENTS: pa_device.number_detector_elements,
            Tags.SENSOR_DIRECTIVITY_SIZE_M: pa_device.detector_element_width_mm / 1000,
            Tags.SENSOR_CENTER_FREQUENCY_HZ: pa_device.center_frequency_Hz,
            Tags.SENSOR_BANDWIDTH_PERCENT: pa_device.bandwidth_percent,
            Tags.SPACING_MM: spacing_in_mm
        })

        for parameter in possible_k_wave_parameters:
            if parameter in self.component_settings:
                k_wave_settings[parameter] = self.component_settings[parameter]
            elif parameter in self.global_settings:
                k_wave_settings[parameter] = self.global_settings[parameter]

        if Tags.K_WAVE_SPECIFIC_DT in self.global_settings and Tags.K_WAVE_SPECIFIC_NT in self.global_settings:
            k_wave_settings["dt"] = self.global_settings[Tags.K_WAVE_SPECIFIC_DT]
            k_wave_settings["Nt"] = self.global_settings[Tags.K_WAVE_SPECIFIC_NT]
        else:
            num_samples = time_series_sensor_data.shape[1]
            time_per_sample_s = 1 / (self.component_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] * 1000000)
            k_wave_settings["dt"] = time_per_sample_s
            k_wave_settings["Nt"] = num_samples
        input_data["settings"] = k_wave_settings
        sio.savemat(acoustic_path, input_data, long_field_names=True)

        if Tags.ACOUSTIC_SIMULATION_3D in self.component_settings and \
                self.component_settings[Tags.ACOUSTIC_SIMULATION_3D]:
            time_reversal_script = "time_reversal_3D"
            axes = (0, 2)
        else:
            time_reversal_script = "time_reversal_2D"
            axes = (0, 1)

        base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        cmd = list()
        cmd.append(self.component_settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-automation")
        cmd.append("-wait")
        cmd.append("-r")
        cmd.append("addpath('" + base_script_path + "');" +
                   time_reversal_script + "('" + acoustic_path + "');exit;")

        cur_dir = os.getcwd()
        os.chdir(self.global_settings[Tags.SIMULATION_PATH])
        self.logger.info(cmd)
        subprocess.run(cmd)

        reconstructed_data = sio.loadmat(acoustic_path + "tr.mat")[Tags.DATA_FIELD_RECONSTRUCTED_DATA]

        reconstructed_data = np.flipud(np.rot90(reconstructed_data, 1, axes))

        field_of_view_mm = detection_geometry.get_field_of_view_mm()
        field_of_view_voxels = (field_of_view_mm / spacing_in_mm).astype(np.int32)
        self.logger.debug(f"FOV (voxels): {field_of_view_voxels}")
        # In case it should be cropped from A to A, then crop from A to A+1
        x_offset_correct = 1 if (field_of_view_voxels[1] - field_of_view_voxels[0]) < 1 else 0
        y_offset_correct = 1 if (field_of_view_voxels[3] - field_of_view_voxels[2]) < 1 else 0
        z_offset_correct = 1 if (field_of_view_voxels[5] - field_of_view_voxels[4]) < 1 else 0

        if len(np.shape(reconstructed_data)) == 2:
            reconstructed_data = np.squeeze(reconstructed_data[field_of_view_voxels[0]:field_of_view_voxels[1] + x_offset_correct,
                                    field_of_view_voxels[4]:field_of_view_voxels[5] + z_offset_correct])
        elif len(np.shape(reconstructed_data)) == 3:
            reconstructed_data = np.squeeze(reconstructed_data[field_of_view_voxels[0]:field_of_view_voxels[1] + x_offset_correct,
                                    field_of_view_voxels[2]:field_of_view_voxels[3] + y_offset_correct,
                                    field_of_view_voxels[4]:field_of_view_voxels[5] + z_offset_correct])
        else:
            self.logger.critical("Unexpected number of dimensions in reconstructed image. "
                                 f"Expected 2 or 3 but was {len(np.shape(reconstructed_data))}")


        os.chdir(cur_dir)
        os.remove(acoustic_path)
        os.remove(acoustic_path + "tr.mat")

        return reconstructed_data
