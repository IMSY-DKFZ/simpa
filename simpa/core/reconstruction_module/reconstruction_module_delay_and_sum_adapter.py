"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.core.reconstruction_module import ReconstructionAdapterBase
from simpa.io_handling.io_hdf5 import load_data_field
import numpy as np
import torch
from simpa.utils.settings import Settings
from simpa.core.reconstruction_module.reconstruction_utils import get_apodization_factor, bandpass_filtering, apply_b_mode, \
    reconstruction_mode_transformation
from simpa.core.device_digital_twins import DetectionGeometryBase


class ImageReconstructionModuleDelayAndSumAdapter(ReconstructionAdapterBase):

    def reconstruction_algorithm(self, time_series_sensor_data, detection_geometry: DetectionGeometryBase):
        """
        Applies the Delay and Sum beamforming algorithm [1] to the time series sensor data (2D numpy array where the
        first dimension corresponds to the sensor elements and the second to the recorded time steps) with the given
        beamforming settings (dictionary).
        A reconstructed image (2D numpy array) is returned.
        This implementation uses PyTorch Tensors to perform computations and is able to run on GPUs.

        [1] T. Kirchner et al. 2018, "Signed Real-Time Delay Multiply and Sum Beamforming for Multispectral
        Photoacoustic Imaging", https://doi.org/10.3390/jimaging4100121
        """

        # check for B-mode methods and perform envelope detection on time series data if specified

        if Tags.RECONSTRUCTION_BMODE_BEFORE_RECONSTRUCTION in self.component_settings\
                and self.component_settings[Tags.RECONSTRUCTION_BMODE_BEFORE_RECONSTRUCTION] \
                and Tags.RECONSTRUCTION_BMODE_METHOD in self.component_settings:
            time_series_sensor_data = apply_b_mode(
                time_series_sensor_data, method=self.component_settings[Tags.RECONSTRUCTION_BMODE_METHOD])

        ### INPUT CHECKING AND VALIDATION ###
        # check settings dictionary for elements and read them in

        # speed of sound: use given speed of sound, otherwise use average from simulation if specified
        if Tags.PROPERTY_SPEED_OF_SOUND in self.global_settings and self.global_settings[Tags.PROPERTY_SPEED_OF_SOUND]:
            speed_of_sound_in_m_per_s = self.global_settings[Tags.PROPERTY_SPEED_OF_SOUND]
        elif Tags.WAVELENGTH in self.global_settings and self.global_settings[Tags.WAVELENGTH]:
            sound_speed_m = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.PROPERTY_SPEED_OF_SOUND)
            speed_of_sound_in_m_per_s = np.mean(sound_speed_m)
        else:
            raise AttributeError("Please specify a value for PROPERTY_SPEED_OF_SOUND"
                                 "or WAVELENGTH to obtain the average speed of sound")

        # time spacing: use kWave specific dt from simulation if set, otherwise sampling rate if specified,
        if Tags.K_WAVE_SPECIFIC_DT in self.global_settings and self.global_settings[Tags.K_WAVE_SPECIFIC_DT]:
            time_spacing_in_ms = self.global_settings[Tags.K_WAVE_SPECIFIC_DT] * 1000
        elif Tags.SENSOR_SAMPLING_RATE_MHZ in self.global_settings and self.global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ]:
            time_spacing_in_ms = 1.0 / (self.global_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] * 1000)
        else:
            raise AttributeError("Please specify a value for SENSOR_SAMPLING_RATE_MHZ or K_WAVE_SPECIFIC_DT")

        # spacing
        if Tags.SPACING_MM in self.global_settings and self.global_settings[Tags.SPACING_MM]:
            spacing_in_mm = self.global_settings[Tags.SPACING_MM]
        else:
            raise AttributeError("Please specify a value for SPACING_MM")

        # get device specific sensor positions
        detection_geometry.check_settings_prerequisites(self.global_settings)

        sensor_positions = detection_geometry.get_detector_element_positions_accounting_for_field_of_view()

        # time series sensor data must be numpy array
        if isinstance(sensor_positions, np.ndarray):
            sensor_positions = torch.from_numpy(sensor_positions)
        if isinstance(time_series_sensor_data, np.ndarray):
            time_series_sensor_data = torch.from_numpy(time_series_sensor_data)
        assert isinstance(time_series_sensor_data, torch.Tensor), \
            'The time series sensor data must have been converted to a tensor'

        # move tensors to GPU if available, otherwise use CPU
        if Tags.GPU not in self.global_settings:
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        else:
            dev = "cuda" if self.global_settings[Tags.GPU] else "cpu"

        torch_device = torch.device(dev)
        sensor_positions = sensor_positions.to(torch_device)
        time_series_sensor_data = time_series_sensor_data.to(torch_device)

        # array must be of correct dimension
        assert time_series_sensor_data.ndim == 2, 'Time series data must have exactly 2 dimensions' \
                                                  ', one for the sensor elements and one for time. ' \
                                                  'Stack images and sensor positions for 3D reconstruction' \
                                                  'Apply beamforming per wavelength if you have a 3D array. '

        # check reconstruction mode - pressure by default
        if Tags.RECONSTRUCTION_MODE in self.component_settings:
            mode = self.component_settings[Tags.RECONSTRUCTION_MODE]
        else:
            mode = Tags.RECONSTRUCTION_MODE_PRESSURE
        time_series_sensor_data = reconstruction_mode_transformation(time_series_sensor_data, mode=mode)

        # apply by default bandpass filter using tukey window with alpha=0.5 on time series data in frequency domain
        if Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING not in self.component_settings \
                or self.component_settings[Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING] is not False:

            cutoff_lowpass = self.component_settings[Tags.BANDPASS_CUTOFF_LOWPASS] \
                if Tags.BANDPASS_CUTOFF_LOWPASS in self.component_settings else int(8e6)
            cutoff_highpass = self.component_settings[Tags.BANDPASS_CUTOFF_HIGHPASS] \
                if Tags.BANDPASS_CUTOFF_HIGHPASS in self.component_settings else int(0.1e6)
            tukey_alpha = self.component_settings[Tags.TUKEY_WINDOW_ALPHA] if Tags.TUKEY_WINDOW_ALPHA in self.component_settings else 0.5
            time_series_sensor_data = bandpass_filtering(time_series_sensor_data,
                                                         time_spacing_in_ms=time_spacing_in_ms,
                                                         cutoff_lowpass=cutoff_lowpass,
                                                         cutoff_highpass=cutoff_highpass,
                                                         tukey_alpha=tukey_alpha)

        ### ALGORITHM ITSELF ###

        ## compute size of beamformed image from field of view ##
        field_of_view = detection_geometry.get_field_of_view_extent_mm()
        xdim = int(np.abs(field_of_view[0] - field_of_view[1]) / spacing_in_mm) + 1
        zdim = int(np.abs(field_of_view[2] - field_of_view[3]) / spacing_in_mm) + 1
        ydim = int(np.abs(field_of_view[4] - field_of_view[5]) / spacing_in_mm) + 1

        if zdim == 1:
            sensor_positions[:, 1] = 0  # Assume imaging plane

        if time_series_sensor_data.shape[0] < sensor_positions.shape[0]:
            self.logger.warning("Warning: The time series data has less sensor element entries than the given sensor positions. "
                                "This might be due to a low simulated resolution, please increase it.")

        n_sensor_elements = time_series_sensor_data.shape[0]

        self.logger.debug(f'Number of pixels in X dimension: {xdim}, Y dimension: {ydim}, Z dimension: {zdim} '
                          f',number of sensor elements: {n_sensor_elements}')

        # construct output image
        output = torch.zeros((xdim, ydim, zdim), dtype=torch.float32, device=torch_device)

        xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim, device=torch_device),
                                        torch.arange(ydim, device=torch_device),
                                        torch.arange(zdim, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))

        delays = torch.sqrt((yy * spacing_in_mm - sensor_positions[:, 2][jj]) ** 2 +
                            (xx * spacing_in_mm - torch.abs(sensor_positions[:, 0][jj])) ** 2 +
                            (zz * spacing_in_mm - torch.abs(sensor_positions[:, 1][jj])) ** 2) \
            / (speed_of_sound_in_m_per_s * time_spacing_in_ms)

        # perform index validation
        invalid_indices = torch.where(torch.logical_or(delays < 0, delays >= float(time_series_sensor_data.shape[1])))
        torch.clip_(delays, min=0, max=time_series_sensor_data.shape[1] - 1)

        # interpolation of delays
        lower_delays = (torch.floor(delays)).long()
        upper_delays = lower_delays + 1
        torch.clip_(upper_delays, min=0, max=time_series_sensor_data.shape[1] - 1)
        lower_values = time_series_sensor_data[jj, lower_delays]
        upper_values = time_series_sensor_data[jj, upper_delays]
        values = lower_values * (upper_delays - delays) + upper_values * (delays - lower_delays)

        # perform apodization if specified
        if Tags.RECONSTRUCTION_APODIZATION_METHOD in self.component_settings:
            apodization = get_apodization_factor(apodization_method=self.component_settings[Tags.RECONSTRUCTION_APODIZATION_METHOD],
                                                 dimensions=(xdim, ydim, zdim), n_sensor_elements=n_sensor_elements,
                                                 device=torch_device)
            values = values * apodization

        # set values of invalid indices to 0 so that they don't influence the result
        values[invalid_indices] = 0
        sum = torch.sum(values, dim=3)
        counter = torch.count_nonzero(values, dim=3)
        torch.divide(sum, counter, out=output)

        reconstructed = output.cpu().numpy()

        # check for B-mode methods and perform envelope detection on beamformed image if specified

        if Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION in self.component_settings \
                and self.component_settings[Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION] \
                and Tags.RECONSTRUCTION_BMODE_METHOD in self.component_settings:
            reconstructed = apply_b_mode(
                reconstructed, method=self.component_settings[Tags.RECONSTRUCTION_BMODE_METHOD])

        return reconstructed.squeeze()


def reconstruct_delay_and_sum_pytorch(time_series_sensor_data: np.ndarray,
                                      detection_geometry: DetectionGeometryBase,
                                      settings: dict = None,
                                      sound_of_speed: int = 1540,
                                      time_spacing: float = 2.5e-8,
                                      sensor_spacing: float = 0.1) -> np.ndarray:
    """
    Convenience function for reconstructing time series data using Delay and Sum algorithm implemented in PyTorch

    :param time_series_sensor_data: (2D numpy array) sensor data of shape (sensor elements, time steps)
    :param detection_geometry: The DetectionGeometryBase that should be used to reconstruct the given time series data
    :param settings: (dict) settings dictionary: by default there is none and the other parameters are used instead,
                     but if parameters are given in the settings those will be used instead of parsed arguments)
    :param sound_of_speed: (int) speed of sound in medium in meters per second (default: 1540 m/s)
    :param time_spacing: (float) time between sampling points in seconds (default: 2.5e-8 s which is equal to 40 MHz)
    :param sensor_spacing: (float) space between sensor elements in millimeters (default: 0.1 mm)
    :return: (2D numpy array) reconstructed image as 2D numpy array
    """

    # create settings if they don't exist yet
    if settings is None:
        settings = Settings()

    # parse reconstruction settings if they are not given in the settings
    if Tags.PROPERTY_SPEED_OF_SOUND not in settings or settings[Tags.PROPERTY_SPEED_OF_SOUND] is None:
        settings[Tags.PROPERTY_SPEED_OF_SOUND] = sound_of_speed

    if Tags.SENSOR_SAMPLING_RATE_MHZ not in settings or settings[Tags.SENSOR_SAMPLING_RATE_MHZ] is None:
        settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = (1.0 / time_spacing) / 1000000

    if Tags.SPACING_MM not in settings or settings[Tags.SPACING_MM] is None:
        settings[Tags.SPACING_MM] = sensor_spacing

    adapter = ImageReconstructionModuleDelayAndSumAdapter(settings)
    return adapter.reconstruction_algorithm(time_series_sensor_data, detection_geometry)
