# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from typing import Tuple
from simpa.log.file_logger import Logger
from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils.settings import Settings
from simpa.io_handling.io_hdf5 import load_data_field
from simpa.utils import Tags
import torch
import torch.fft
from torch import Tensor
import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey


def get_apodization_factor(apodization_method: str = Tags.RECONSTRUCTION_APODIZATION_BOX,
                           dimensions: tuple = None, n_sensor_elements=None,
                           device: torch.device = 'cpu') -> torch.tensor:
    """
    Construct apodization factors according to `apodization_method` [hann, hamming or box apodization (default)]
    for given dimensions and `n_sensor_elements`.

    :param apodization_method: (str) Apodization method, one of Tags.RECONSTRUCTION_APODIZATION_HANN,
                        Tags.RECONSTRUCTION_APODIZATION_HAMMING and Tags.RECONSTRUCTION_APODIZATION_BOX (default)
    :param dimensions: (tuple) size of each dimension of reconstructed image as int, might have 2 or 3 entries.
    :param n_sensor_elements: (int) number of sensor elements
    :param device: (torch device) PyTorch tensor device
    :return: (torch tensor) tensor with apodization factors which can be multipied with DAS values
    """

    if dimensions is None or n_sensor_elements is None:
        raise AttributeError("dimensions and n_sensor_elements must be specified and not be None")

    # hann window
    if apodization_method == Tags.RECONSTRUCTION_APODIZATION_HANN:
        hann = torch.hann_window(n_sensor_elements, device=device)
        output = hann.expand(dimensions + (n_sensor_elements,))
    # hamming window
    elif apodization_method == Tags.RECONSTRUCTION_APODIZATION_HAMMING:
        hamming = torch.hamming_window(n_sensor_elements, device=device)
        output = hamming.expand(dimensions + (n_sensor_elements,))
    # box window apodization as default
    else:
        output = torch.ones(dimensions + (n_sensor_elements,), device=device)

    return output


def bandpass_filtering(data: np.ndarray, time_spacing_in_ms: float = None,
                       cutoff_lowpass: int = int(8e6), cutoff_highpass: int = int(0.1e6),
                       tukey_alpha: float = 0.5) -> np.ndarray:
    """
    Apply a bandpass filter with cutoff values at `cutoff_lowpass` and `cutoff_highpass` MHz 
    and a tukey window with alpha value of `tukey_alpha` inbetween on the `data` in Fourier space.
    Note that the filter operates on both, negative and positive frequencies similarly.

    :param data: (numpy array) data to be filtered
    :param time_spacing_in_ms: (float) time spacing in milliseconds, e.g. 2.5e-5
    :param cutoff_lowpass: (int) Signal above this value will be ignored (in MHz)
    :param cutoff_highpass: (int) Signal below this value will be ignored (in MHz)
    :param tukey_alpha: (float) transition value between 0 (rectangular) and 1 (Hann window)
    :return: (numpy array) filtered data
    """

    # construct bandpass filter given the cutoff values and time spacing
    frequencies = np.fft.fftfreq(data.shape[-1], d=time_spacing_in_ms/1000)

    if cutoff_highpass > cutoff_lowpass:
        raise ValueError("The highpass cutoff value must be lower than the lowpass cutoff value.")

    # find closest indices for frequencies
    small_index = (np.abs(frequencies - cutoff_highpass)).argmin()
    large_index = (np.abs(frequencies - cutoff_lowpass)).argmin()

    # filter negative and positive frequencies with same window
    win = tukey(large_index - small_index, alpha=tukey_alpha)
    window = np.zeros(frequencies.shape)
    window[small_index:large_index] = win
    if small_index == 0:
        small_index = 1
        win = win[:-1]
    window[-large_index:-small_index] = win

    # transform data into Fourier space, multiply filter and transform back
    data_in_fourier_space = np.fft.fft(data)
    filtered_data_in_fourier_space = data_in_fourier_space * np.broadcast_to(window, np.shape(data_in_fourier_space))
    return np.fft.ifft(filtered_data_in_fourier_space).real


def bandpass_filtering_with_settings(data: np.ndarray, global_settings: Settings, component_settings: Settings,
                                     device: DetectionGeometryBase) -> np.ndarray:
    """
    Apply a bandpass filter with cutoff values at `cutoff_lowpass` and `cutoff_highpass` MHz
    and a tukey window with alpha value of `tukey_alpha` inbetween on the `data` in Fourier space.
    Those values are obtained from the `global_settings`, `component_settings`, and `device`.
    Note that the filter operates on both, negative and positive frequencies similarly.

    :param data: (numpy array) data to be filtered
    :param global_settings: (Settings) settings for the whole simulation
    :param component_settings: (Settings) settings for the reconstruction module
    :param device:
    :return: (numpy array) filtered data
    """
    if Tags.K_WAVE_SPECIFIC_DT in global_settings and global_settings[Tags.K_WAVE_SPECIFIC_DT]:
        time_spacing_in_ms = global_settings[Tags.K_WAVE_SPECIFIC_DT] * 1000
    elif device.sampling_frequency_MHz is not None:
        time_spacing_in_ms = 1.0 / (device.sampling_frequency_MHz * 1000)
    else:
        raise AttributeError("Please specify a value for SENSOR_SAMPLING_RATE_MHZ or K_WAVE_SPECIFIC_DT")
    cutoff_lowpass = component_settings[Tags.BANDPASS_CUTOFF_LOWPASS] \
        if Tags.BANDPASS_CUTOFF_LOWPASS in component_settings else int(8e6)
    cutoff_highpass = component_settings[Tags.BANDPASS_CUTOFF_HIGHPASS] \
        if Tags.BANDPASS_CUTOFF_HIGHPASS in component_settings else int(0.1e6)
    tukey_alpha = component_settings[
        Tags.TUKEY_WINDOW_ALPHA] if Tags.TUKEY_WINDOW_ALPHA in component_settings else 0.5

    if data is None or time_spacing_in_ms is None:
        raise AttributeError("data and time spacing must be specified")

    return bandpass_filtering(data, time_spacing_in_ms, cutoff_lowpass, cutoff_highpass, tukey_alpha)


def apply_b_mode(data: np.ndarray = None, method: str = None) -> np.ndarray:
    """
    Applies B-Mode specified method to data. Method is either
    envelope detection using hilbert transform (Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM),
    absolute value (Tags.RECONSTRUCTION_BMODE_METHOD_ABS) or
    none if nothing is specified is performed.

    :param data: (numpy array) data used for applying B-Mode method
    :param method: (str) Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM or Tags.RECONSTRUCTION_BMODE_METHOD_ABS
    :return: (numpy array) data with B-Mode method applied, all
    """
    # input checks
    if data is None:
        raise AttributeError("data must be specified")

    if data.ndim < 2:
        raise AttributeError("data must have at least two dimensions")

    if method == Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM:
        # perform envelope detection using hilbert transform in depth direction
        hilbert_transformed = hilbert(data, axis=1)
        output = np.abs(hilbert_transformed)
    elif method == Tags.RECONSTRUCTION_BMODE_METHOD_ABS:
        # perform envelope detection using absolute value
        output = np.abs(data)
    else:
        print("You have not specified a B-mode method")
        output = data

    # sanity check that no elements are below zero
    if output[output < 0].sum() != 0:
        print("There are still negative values in the data.")

    return output


def reconstruction_mode_transformation(time_series_sensor_data: torch.tensor = None,
                                       mode: str = Tags.RECONSTRUCTION_MODE_PRESSURE) -> torch.tensor:
    """
    Transformes `time_series_sensor_data` for other modes, for example `Tags.RECONSTRUCTION_MODE_DIFFERENTIAL`.
    Default mode is `Tags.RECONSTRUCTION_MODE_PRESSURE`.

    :param time_series_sensor_data: (torch tensor) Time series data to be transformed
    :param mode: (str) reconstruction mode: Tags.RECONSTRUCTION_MODE_PRESSURE (default)
                or Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
    :return: (torch tensor) potentially transformed tensor
    """

    # depending on mode use pressure data or its derivative
    if mode == Tags.RECONSTRUCTION_MODE_DIFFERENTIAL:
        zeros = torch.zeros([time_series_sensor_data.shape[0], 1], names=None).to(time_series_sensor_data.device)
        time_vector = torch.arange(1, time_series_sensor_data.shape[1]+1).to(time_series_sensor_data.device)
        time_derivative_pressure = time_series_sensor_data[:, 1:] - time_series_sensor_data[:, 0:-1]
        time_derivative_pressure = torch.cat([time_derivative_pressure, zeros], dim=1)
        time_derivative_pressure = torch.mul(time_derivative_pressure, time_vector)
        output = time_derivative_pressure  # use time derivative pressure
    elif mode == Tags.RECONSTRUCTION_MODE_PRESSURE:
        output = time_series_sensor_data  # already in pressure format
    else:
        raise AttributeError(
            "An invalid reconstruction mode was set, only differential and pressure are supported.")
    return output


def preparing_reconstruction_and_obtaining_reconstruction_settings(
        time_series_sensor_data: np.ndarray, component_settings: Settings, global_settings: Settings,
        detection_geometry: DetectionGeometryBase, logger: Logger) -> Tuple[torch.tensor, torch.tensor,
                                                                            float, float, float,
                                                                            torch.device]:
    """
    Performs all preparation steps that need to be done before reconstructing an image:
    - performs envelope detection of time series data if specified
    - obtains speed of sound value from settings
    - obtains time spacing value from settings or PA device
    - obtain spacing from settings
    - checks PA device prerequisites
    - obtains sensor positions from PA device
    - moves data arrays on correct torch device
    - computed differential mode if specified
    - perform bandpass filtering if specified

    Returns:

    time_series_sensor_data: (torch tensor) potentially preprocessed time series data
    sensor_positions: (torch tensor) sensor element positions of PA device
    speed_of_sound_in_m_per_s: (float) speed of sound in m/s
    spacing_in_mm: (float) spacing of voxels in reconstructed image in mm
    time_spacing_in_ms: (float) temporal spacing of the time series data in ms
    torch_device: (torch device) either cpu or cuda GPU device used for the tensors
    """

    ### INPUT CHECKING AND VALIDATION ###
    # check settings dictionary for elements and read them in

    # speed of sound: use given speed of sound, otherwise use average from simulation if specified
    if Tags.DATA_FIELD_SPEED_OF_SOUND in component_settings and component_settings[Tags.DATA_FIELD_SPEED_OF_SOUND]:
        speed_of_sound_in_m_per_s = component_settings[Tags.DATA_FIELD_SPEED_OF_SOUND]
    elif Tags.WAVELENGTH in global_settings and global_settings[Tags.WAVELENGTH]:
        sound_speed_m = load_data_field(global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_SPEED_OF_SOUND)
        speed_of_sound_in_m_per_s = np.mean(sound_speed_m)
    else:
        raise AttributeError("Please specify a value for PROPERTY_SPEED_OF_SOUND"
                             "or WAVELENGTH to obtain the average speed of sound")

    # time spacing: use kWave specific dt from simulation if set, otherwise sampling rate if specified,
    if Tags.K_WAVE_SPECIFIC_DT in global_settings and global_settings[Tags.K_WAVE_SPECIFIC_DT]:
        time_spacing_in_ms = global_settings[Tags.K_WAVE_SPECIFIC_DT] * 1000
    elif detection_geometry.sampling_frequency_MHz is not None:
        time_spacing_in_ms = 1.0 / (detection_geometry.sampling_frequency_MHz * 1000)
    else:
        raise AttributeError("Please specify a value for SENSOR_SAMPLING_RATE_MHZ or K_WAVE_SPECIFIC_DT")

    logger.debug(f"Using a time_spacing of {time_spacing_in_ms}")

    # spacing
    if Tags.SPACING_MM in component_settings and component_settings[Tags.SPACING_MM]:
        spacing_in_mm = component_settings[Tags.SPACING_MM]
        logger.debug(f"Reconstructing with spacing from component_settings: {spacing_in_mm}")
    elif Tags.SPACING_MM in global_settings and global_settings[Tags.SPACING_MM]:
        spacing_in_mm = global_settings[Tags.SPACING_MM]
        logger.debug(f"Reconstructing with spacing from global_settings: {spacing_in_mm}")
    else:
        raise AttributeError("Please specify a value for SPACING_MM in either the component_settings or"
                             "the global_settings.")

    # get device specific sensor positions
    detection_geometry.check_settings_prerequisites(global_settings)

    sensor_positions = detection_geometry.get_detector_element_positions_base_mm()

    # time series sensor data must be numpy array
    if isinstance(sensor_positions, np.ndarray):
        sensor_positions = torch.from_numpy(sensor_positions)
    if isinstance(time_series_sensor_data, np.ndarray):
        time_series_sensor_data = torch.from_numpy(time_series_sensor_data)
    assert isinstance(time_series_sensor_data, torch.Tensor), \
        'The time series sensor data must have been converted to a tensor'

    # move tensors to GPU if available, otherwise use CPU
    if Tags.GPU not in global_settings:
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:
        dev = "cuda" if global_settings[Tags.GPU] else "cpu"

    torch_device = torch.device(dev)
    sensor_positions = sensor_positions.to(torch_device)
    time_series_sensor_data = time_series_sensor_data.to(torch_device)

    # array must be of correct dimension
    assert time_series_sensor_data.ndim == 2, 'Time series data must have exactly 2 dimensions' \
                                              ', one for the sensor elements and one for time. ' \
                                              'Stack images and sensor positions for 3D reconstruction' \
                                              'Apply beamforming per wavelength if you have a 3D array. '

    # check reconstruction mode - pressure by default
    if Tags.RECONSTRUCTION_MODE in component_settings:
        mode = component_settings[Tags.RECONSTRUCTION_MODE]
    else:
        mode = Tags.RECONSTRUCTION_MODE_PRESSURE
    time_series_sensor_data = reconstruction_mode_transformation(time_series_sensor_data, mode=mode)

    return (time_series_sensor_data, sensor_positions, speed_of_sound_in_m_per_s, spacing_in_mm,
            time_spacing_in_ms, torch_device)


def compute_image_dimensions(detection_geometry: DetectionGeometryBase, spacing_in_mm: float,
                             speed_of_sound_in_m_per_s: float, logger: Logger) -> Tuple[int, int, int, int, int,
                                                                                        int, int, int, int]:
    """
    compute size of beamformed image from field of view of detection geometry

    Returns x,z,y dimensions of reconstructed image volume in pixels as well 
    as the range for each dimension as start and end pixels.
    """
    field_of_view = detection_geometry.field_of_view_extent_mm
    logger.debug(f"Field of view: {field_of_view}")

    xdim_start = int(field_of_view[0] / spacing_in_mm)
    xdim_end = int(field_of_view[1] / spacing_in_mm)
    zdim_start = int(field_of_view[2] / spacing_in_mm)
    zdim_end = int(field_of_view[3] / spacing_in_mm)
    ydim_start = int(field_of_view[4] / spacing_in_mm)
    ydim_end = int(field_of_view[5] / spacing_in_mm)

    xdim = (xdim_end - xdim_start)
    ydim = (ydim_end - ydim_start)
    zdim = (zdim_end - zdim_start)

    if xdim < 1:
        xdim = 1
    if ydim < 1:
        ydim = 1
    if zdim < 1:
        zdim = 1

    logger.debug(f"FOV X: {xdim_start * spacing_in_mm} - {xdim_end * spacing_in_mm}")
    logger.debug(f"FOV Y: {ydim_start * spacing_in_mm} - {ydim_end * spacing_in_mm}")
    logger.debug(f"FOV Z: {zdim_start * spacing_in_mm} - {zdim_end * spacing_in_mm}")

    return xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end


def compute_delay_and_sum_values(time_series_sensor_data: Tensor, sensor_positions: torch.tensor, xdim: int,
                                 ydim: int, zdim: int, xdim_start: int, xdim_end: int, ydim_start: int, ydim_end: int,
                                 zdim_start: int, zdim_end: int, spacing_in_mm: float, speed_of_sound_in_m_per_s: float,
                                 time_spacing_in_ms: float, logger: Logger, torch_device: torch.device,
                                 component_settings: Settings) -> Tuple[torch.tensor, int]:
    """
    Perform the core computation of Delay and Sum, without summing up the delay dependend values.

    Returns
    - values (torch tensor) of the time series data corrected for delay and sensor positioning, ready to be summed up
    - and n_sensor_elements (int) which might be used for later computations
    """

    if time_series_sensor_data.shape[0] < sensor_positions.shape[0]:
        logger.warning("Warning: The time series data has less sensor element entries than the given sensor positions. "
                       "This might be due to a low simulated resolution, please increase it.")

    n_sensor_elements = time_series_sensor_data.shape[0]

    logger.debug(f'Number of pixels in X dimension: {xdim}, Y dimension: {ydim}, Z dimension: {zdim} '
                 f',number of sensor elements: {n_sensor_elements}')

    if zdim == 1:
        xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))
    else:
        xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim_start, zdim_end, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))

    delays = torch.sqrt((yy * spacing_in_mm - sensor_positions[:, 2][jj]) ** 2 +
                        (xx * spacing_in_mm - sensor_positions[:, 0][jj]) ** 2 +
                        (zz * spacing_in_mm - sensor_positions[:, 1][jj]) ** 2) \
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
    if Tags.RECONSTRUCTION_APODIZATION_METHOD in component_settings:
        apodization = get_apodization_factor(apodization_method=component_settings[Tags.RECONSTRUCTION_APODIZATION_METHOD],
                                             dimensions=(xdim, ydim, zdim), n_sensor_elements=n_sensor_elements,
                                             device=torch_device)
        values = values * apodization

    # set values of invalid indices to 0 so that they don't influence the result
    values[invalid_indices] = 0

    del delays  # free memory of delays

    return values, n_sensor_elements
