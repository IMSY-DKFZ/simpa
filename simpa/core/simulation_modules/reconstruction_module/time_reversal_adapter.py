# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.utils.signals import reorder_sensor_data

from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import compute_image_dimensions
from simpa.utils import Tags, round_x5_away_from_zero
from simpa.utils.settings import Settings
from simpa.core.simulation_modules.reconstruction_module import ReconstructionAdapterBase
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry
import numpy as np


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
        # we add eps of 1e-10 because numpy rounds 0.5 to the next even number
        detector_positions_voxels = round_x5_away_from_zero(detector_positions / spacing_in_mm)

        # plus 2 because of off-
        volume_x_dim = int(np.ceil(self.global_settings[Tags.DIM_VOLUME_X_MM] / spacing_in_mm) + 1)
        # by-one error in matlab
        volume_y_dim = int(np.ceil(self.global_settings[Tags.DIM_VOLUME_Y_MM] / spacing_in_mm) + 1)
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
            k_wave_settings[Tags.K_WAVE_SPECIFIC_DT] = self.global_settings[Tags.K_WAVE_SPECIFIC_DT]
            k_wave_settings[Tags.K_WAVE_SPECIFIC_NT] = self.global_settings[Tags.K_WAVE_SPECIFIC_NT]
        else:
            num_samples = time_series_sensor_data.shape[1]
            time_per_sample_s = 1 / (self.component_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] * 1000000)
            k_wave_settings[Tags.K_WAVE_SPECIFIC_DT] = time_per_sample_s
            k_wave_settings[Tags.K_WAVE_SPECIFIC_NT] = num_samples

        reconstructed_data = self.run_k_wave_timereversal(input_data, k_wave_settings)

        field_of_view_mm = detection_geometry.get_field_of_view_mm()
        _, _, _, xdim_start, xdim_end,  ydim_start, ydim_end, zdim_start, zdim_end = compute_image_dimensions(
            field_of_view_mm, spacing_in_mm, self.logger)
        field_of_view_voxels = [xdim_start, xdim_end, zdim_start, zdim_end, ydim_start, ydim_end]  # change ordering
        field_of_view_voxels = [int(dim) for dim in field_of_view_voxels]  # cast to int

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

        return reconstructed_data

    def run_k_wave_timereversal(self, data: dict, k_wave_settings: Settings):
        """
        Run k-Wave time reversal for the provided medium and sensor configuration.

        Parameters
        ----------
        :param data: Dictionary containing simulation input fields. Expected keys include:
            - Tags.DATA_FIELD_TIME_SERIES_DATA (np.ndarray)
            - Tags.Tags.KWAVE_PROPERTY_SENSOR_MASK (np.ndarray)
            - Optional: Tags.DATA_FIELD_SPEED_OF_SOUND, Tags.DATA_FIELD_DENSITY,
                Tags.DATA_FIELD_ALPHA_COEFF (scalar)

        :input k_wave_settings: k-Wave configuration and simulation parameters (spacing, sensor
            settings, PML, GPU flag, etc.).

        :return: Reconstructed image
        """

        sensor_mask = data[Tags.KWAVE_PROPERTY_SENSOR_MASK]
        ndim = sensor_mask.ndim
        assert ndim in (2, 3), "Only 2D and 3D simulations are supported."

        # Setup grid
        N = np.asarray(sensor_mask.shape, dtype=int)
        spacing_mm = k_wave_settings[Tags.SPACING_MM] / 1000.0
        d = spacing_mm * np.ones(ndim)
        kgrid = kWaveGrid(N, d)
        kgrid.setTime(k_wave_settings[Tags.K_WAVE_SPECIFIC_NT], k_wave_settings[Tags.K_WAVE_SPECIFIC_DT])

        # Setup source
        source = kSource()
        # source.p0 = 0

        # Setup medium properties
        medium = kWaveMedium(
            sound_speed=data.get(Tags.DATA_FIELD_SPEED_OF_SOUND, 1540),
            alpha_coeff=data.get(Tags.DATA_FIELD_ALPHA_COEFF, 0.01),
            alpha_power=float(k_wave_settings[Tags.KWAVE_PROPERTY_ALPHA_POWER]),
            alpha_mode='no_dispersion',
            density=data.get(Tags.DATA_FIELD_DENSITY, 1000),
        )

        # Setup sensor
        sensor = kSensor()
        sensor.mask = sensor_mask

        if k_wave_settings.get(Tags.MODEL_SENSOR_FREQUENCY_RESPONSE, False):
            center_freq = float(k_wave_settings[Tags.SENSOR_CENTER_FREQUENCY_HZ])  # [Hz]
            bandwidth = float(k_wave_settings[Tags.SENSOR_BANDWIDTH_PERCENT])  # [%]
            sensor.frequency_response([center_freq, bandwidth])

        time_series_data = data[Tags.DATA_FIELD_TIME_SERIES_DATA]
        if ndim == 2:
            #  kwave expects the time series data to be in an order
            # "based on the angle that each sensor point makes with the centre of the grid"
            time_series_data = reorder_sensor_data(kgrid, sensor, time_series_data)

        sensor.time_reversal_boundary_data = time_series_data

        simulation_options = SimulationOptions(
            data_cast='single',
            pml_inside=k_wave_settings[Tags.KWAVE_PROPERTY_PMLInside],
            pml_alpha=k_wave_settings[Tags.KWAVE_PROPERTY_PMLAlpha],
            pml_auto=True,
            save_to_disk=True
        )

        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=k_wave_settings[Tags.GPU]
        )

        kspaceFirstOrdernD = kspaceFirstOrder3D if ndim == 3 else kspaceFirstOrder2D
        reconstructed_data = kspaceFirstOrdernD(kgrid, source, sensor, medium, simulation_options, execution_options)
        reconstructed_data = reconstructed_data['p_final']

        return reconstructed_data
