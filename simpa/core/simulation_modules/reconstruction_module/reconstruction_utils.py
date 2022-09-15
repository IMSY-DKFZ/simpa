# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from asyncio.log import logger
from time import time
from typing import Tuple, Union

from simpa.log.file_logger import Logger
from simpa.core.device_digital_twins import DetectionGeometryBase
from simpa.utils.settings import Settings
from simpa.io_handling.io_hdf5 import load_data_field
from simpa.utils import Tags
from simpa.utils.calculate import bilinear_interpolation, print_memory_stats
import torch
import torch.fft
from torch import Tensor
import numpy as np
from scipy.signal import hilbert, butter, lfilter
from scipy.signal.windows import tukey
from scipy.ndimage import zoom

# TODO: Delete
import sys
import gc
import time


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

def bandpass_filter_with_settings(data: np.ndarray, global_settings: Settings, component_settings: Settings,
                                           device: DetectionGeometryBase) -> np.ndarray:
        """
        Applies corresponding bandpass filter which can be set in 
        `component_settings[Tags.BANDPASS_FILTER_METHOD]`, using Tukey window-based filter as default.

        :param data: (numpy array) data to be filtered
        :param global_settings: (Settings) settings for the whole simulation
        :param component_settings: (Settings) settings for the reconstruction module
        :param device:
        :return: (numpy array) filtered data
        """

        # select corresponding filtering method depending on tag in settings
        if Tags.BANDPASS_FILTER_METHOD in component_settings:
            if component_settings[Tags.BANDPASS_FILTER_METHOD] == Tags.TUKEY_BANDPASS_FILTER:
                return tukey_bandpass_filtering_with_settings(data, global_settings,component_settings, device)
            elif component_settings[Tags.BANDPASS_FILTER_METHOD] == Tags.BUTTERWORTH_BANDPASS_FILTER:
                return butter_bandpass_filtering_with_settings(data, global_settings,component_settings, device)
            else:
                return tukey_bandpass_filtering_with_settings(data, global_settings,component_settings, device)
        else:
            return tukey_bandpass_filtering_with_settings(data, global_settings,component_settings, device)

def butter_bandpass_filtering(data: np.array,  time_spacing_in_ms: float = None,
                              cutoff_lowpass: int = int(8e6), cutoff_highpass: int = int(0.1e6),
                              order: int = 1) -> np.ndarray:
    """
    Apply a butterworth bandpass filter of `order` with cutoff values at `cutoff_lowpass`
    and `cutoff_highpass` MHz on the `data` using the scipy.signal.butter filter.

    :param data: (numpy array) data to be filtered
    :param time_spacing_in_ms: (float) time spacing in milliseconds, e.g. 2.5e-5
    :param cutoff_lowpass: (int) Signal above this value will be ignored (in MHz)
    :param cutoff_highpass: (int) Signal below this value will be ignored (in MHz)
    :param order: (int) order of the filter
    :return: (numpy array) filtered data
    """

    # determines nyquist frequency
    nyquist = 0.5 / time_spacing_in_ms*1000

    # computes the critical frequencies
    if cutoff_lowpass is None:
        low = 0.000001
    else:
        low = (cutoff_lowpass / nyquist)
    if cutoff_highpass is None:
        high = 0.999999999
    else:
        high = (cutoff_highpass / nyquist)
    
    b, a = butter(N=order, Wn=[high, low], btype='band')
    y = lfilter(b, a, data)
    
    return y

def butter_bandpass_filtering_with_settings(data: np.ndarray, global_settings: Settings, component_settings: Settings,
                                           device: DetectionGeometryBase) -> np.ndarray:
    """
    Apply a butterworth bandpass filter of `order` with cutoff values at `cutoff_lowpass`
    and `cutoff_highpass` MHz on the `data` using the scipy.signal.butter filter.
    Those values are obtained from the `global_settings`, `component_settings`, and `device`.

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
    filter_order = component_settings[
        Tags.BUTTERWORTH_FILTER_ORDER] if Tags.BUTTERWORTH_FILTER_ORDER in component_settings else 1

    if data is None or time_spacing_in_ms is None:
        raise AttributeError("data and time spacing must be specified")

    return butter_bandpass_filtering(data, time_spacing_in_ms, cutoff_lowpass, cutoff_highpass, filter_order)


def tukey_bandpass_filtering(data: np.ndarray, time_spacing_in_ms: float = None,
                       cutoff_lowpass: int = int(8e6), cutoff_highpass: int = int(0.1e6),
                       tukey_alpha: float = 0.5, resampling_for_fft: bool =False) -> np.ndarray:
    """
    Apply a tukey bandpass filter with cutoff values at `cutoff_lowpass` and `cutoff_highpass` MHz 
    and a tukey window with alpha value of `tukey_alpha` inbetween on the `data` in Fourier space.
    Note that the filter operates on both, negative and positive frequencies similarly.
    Filtering is performed along the last dimension.

    :param data: (numpy array) data to be filtered
    :param time_spacing_in_ms: (float) time spacing in milliseconds, e.g. 2.5e-5
    :param cutoff_lowpass: (int) Signal above this value will be ignored (in MHz)
    :param cutoff_highpass: (int) Signal below this value will be ignored (in MHz)
    :param tukey_alpha: (float) transition value between 0 (rectangular) and 1 (Hann window)
    :param resampling_for_fft: (bool) whether the data is resampled to a power of 2 in time dimension
    before applying the FFT and resampled back after filtering
    :return: (numpy array) filtered data
    """

    # input checking
    if cutoff_highpass > cutoff_lowpass:
        raise ValueError("The highpass cutoff value must be lower than the lowpass cutoff value.")

    # no resampling by default
    resampling_factor = 1 
    original_size = data.shape[-1]
    target_size = original_size

    # resampling if requested
    if resampling_for_fft:
        # resampling settings
        order = 0
        mode = 'constant'

        target_size = int(2**(np.ceil(np.log2(original_size)))) # compute next larger power of 2
        resampling_factor = original_size/target_size
        zoom_factors = [1]*data.ndim # resampling factor for each dimension
        zoom_factors[-1] = 1.0/resampling_factor

        data = zoom(data, zoom_factors, order=order, mode=mode)

    # compute closest indices for cutoff frequencies, limited by the Nyquist frequency
    single_voxel = resampling_factor / (time_spacing_in_ms/1000 * target_size)
    small_index = int(np.minimum((cutoff_highpass / single_voxel), target_size/2.0))
    large_index = int(np.minimum((cutoff_lowpass / single_voxel), target_size/2.0))

    # construct bandpass filter given the cutoff values with tukey window in negative and positive frequencies
    win = tukey(large_index - small_index, alpha=tukey_alpha)
    window = np.zeros(target_size)
    window[small_index:large_index] = win
    if small_index == 0:
        small_index = 1
        win = win[:-1]
    window[-large_index:-small_index] = win

    # transform data into Fourier space, multiply filter and transform back
    data_in_fourier_space = np.fft.fft(data)
    filtered_data_in_fourier_space = data_in_fourier_space * np.broadcast_to(window, np.shape(data_in_fourier_space))
    filtered_data =  np.fft.ifft(filtered_data_in_fourier_space).real

    # resample back to original size if necessary
    if resampling_for_fft:
        inverse_zoom_factors = [1.0/factor for factor in zoom_factors]
        return zoom(filtered_data, inverse_zoom_factors, order=order, mode=mode)
    else:
        return filtered_data


def tukey_bandpass_filtering_with_settings(data: np.ndarray, global_settings: Settings, component_settings: Settings,
                                           device: DetectionGeometryBase) -> np.ndarray:
    """
    Apply a tukey bandpass filter with cutoff values at `cutoff_lowpass` and `cutoff_highpass` MHz
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
    resampling_for_fft = component_settings[Tags.RECONSTRUCTION_PERFORM_RESAMPLING_FOR_FFT] \
        if Tags.RECONSTRUCTION_PERFORM_RESAMPLING_FOR_FFT in component_settings else False

    if data is None or time_spacing_in_ms is None:
        raise AttributeError("data and time spacing must be specified")

    return tukey_bandpass_filtering(data, time_spacing_in_ms, cutoff_lowpass, cutoff_highpass, tukey_alpha, resampling_for_fft)


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
    speed_of_sound_in_m_per_s: (float or np.ndarray) speed of sound in m/s
    spacing_in_mm: (float) spacing of voxels in reconstructed image in mm
    time_spacing_in_ms: (float) temporal spacing of the time series data in ms
    torch_device: (torch device) either cpu or cuda GPU device used for the tensors
    """

    ### INPUT CHECKING AND VALIDATION ###
    # check settings dictionary for elements and read them in

    # speed of sound: use given speed of sound
    if Tags.DATA_FIELD_SPEED_OF_SOUND in component_settings:
        speed_of_sound_in_m_per_s = component_settings[Tags.DATA_FIELD_SPEED_OF_SOUND]
        if Tags.SOS_HETEROGENOUS not in global_settings or not global_settings[Tags.SOS_HETEROGENOUS]:
            speed_of_sound_in_m_per_s = np.mean(speed_of_sound_in_m_per_s)
            logger.debug(f"Using a fixed SoS-value of {speed_of_sound_in_m_per_s:.2f}")
        else:
            logger.debug(f"Using a heterogenous SoS-map ({type(speed_of_sound_in_m_per_s)}) " \
                         f"with shape {speed_of_sound_in_m_per_s.shape}")
    else:
        # this loads the sos-map based on volume generation
        speed_of_sound_in_m_per_s = load_data_field(
            global_settings[Tags.SIMPA_OUTPUT_PATH],
            Tags.DATA_FIELD_SPEED_OF_SOUND) 
        if Tags.SOS_HETEROGENOUS not in global_settings or not global_settings[Tags.SOS_HETEROGENOUS]:
            speed_of_sound_in_m_per_s = np.mean(speed_of_sound_in_m_per_s)
            logger.debug(f"Using a fixed SoS-value of {speed_of_sound_in_m_per_s:.2f}")
        else:
            logger.debug(f"Using a heterogenous SoS-map ({type(speed_of_sound_in_m_per_s)}) " \
                         f"with shape {speed_of_sound_in_m_per_s.shape}")

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

    # get device specific sensor positions (in FOV coordinate system, where origin = device_base_position)
    sensor_positions = detection_geometry.get_detector_element_positions_base_mm()

    # time series sensor data must be numpy array
    if isinstance(sensor_positions, np.ndarray):
        sensor_positions = torch.from_numpy(sensor_positions)
    if isinstance(time_series_sensor_data, np.ndarray):
        time_series_sensor_data = torch.from_numpy(time_series_sensor_data)
    assert isinstance(time_series_sensor_data, torch.Tensor), \
        'The time series sensor data must have been converted to a tensor'

    # move tensors to GPU if available, otherwise use CPU
    dev = get_reconstruction_processing_unit(global_settings)

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


def get_reconstruction_processing_unit(global_settings):
    """
    Get device (CPU/GPU) for reconstructing the image.
    :param global_settings: global SIMPA settings
    :return: device for reconstruction
    """
    logger = Logger()
    # if no tag is set, try to use GPU if available
    if Tags.GPU not in global_settings:
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:  # else set tag as user wants
        dev = "cuda" if global_settings[Tags.GPU] else "cpu"

    if dev == 'cuda' and not torch.cuda.is_available():
        # torch will likely raise an error if no GPU is available but set as device -> log it to SIMPA log
        logger.error('Cuda is not available! Check your torch/cuda version.')

    if dev == 'cpu':  # warn the user that CPU reconstruction is slow
        logger.warning(f"Reconstructing on CPU is slow. Check if cuda is available 'torch.cuda.is_available()'.")
    return dev


def compute_image_dimensions(detection_geometry: DetectionGeometryBase, spacing_in_mm: float,
                             speed_of_sound_in_m_per_s: Union[float, np.ndarray],
                             logger: Logger) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """
    compute size of beamformed image from field of view of detection geometry

    Returns x,z,y dimensions of reconstructed image volume in pixels as well 
    as the range for each dimension as start and end pixels.
    """
    field_of_view = detection_geometry.field_of_view_extent_mm
    logger.debug(f"Field of view: {field_of_view}")

    #if isinstance(speed_of_sound_in_m_per_s, float):  
    xdim_start = int(np.round(field_of_view[0] / spacing_in_mm))
    xdim_end = int(np.round(field_of_view[1] / spacing_in_mm))
    zdim_start = int(np.round(field_of_view[2] / spacing_in_mm))
    zdim_end = int(np.round(field_of_view[3] / spacing_in_mm))
    ydim_start = int(np.round(field_of_view[4] / spacing_in_mm))
    ydim_end = int(np.round(field_of_view[5] / spacing_in_mm))

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
                                 zdim_start: int, zdim_end: int, spacing_in_mm: float,
                                 speed_of_sound_in_m_per_s: Union[float,np.ndarray],
                                 time_spacing_in_ms: float, logger: Logger, torch_device: torch.device,
                                 component_settings: Settings, global_settings: Settings, 
                                 device_base_position_mm: np.ndarray) -> Tuple[torch.tensor, int]:
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

    # TODO: delete
    logger.debug(f"MODIFIED VERSION OF COMPUTE DELAY AND SUM VALUES based on Toms off-by-one")

    x_offset = 0.5 if xdim % 2 == 0 else 0  # to ensure pixels are symmetrically arranged around the 0 like the
    # sensor positions, add an offset of 0.5 pixels if the dimension is even

    x = xdim_start + torch.arange(xdim, device=torch_device, dtype=torch.float32) + x_offset
    y = ydim_start + torch.arange(ydim, device=torch_device, dtype=torch.float32)
    if zdim == 1:
        xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))
        z = torch.arange(zdim, device=torch_device, dtype=torch.float32)
    else:
        xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim_start, zdim_end, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))
        z = zdim_start + torch.arange(zdim, device=torch_device, dtype=torch.float32)
    j = torch.arange(n_sensor_elements, device=torch_device, dtype=torch.float32)

    if not isinstance(speed_of_sound_in_m_per_s, np.ndarray):
        xx, yy, zz, jj = torch.meshgrid(x, y, z, j)
        jj = jj.long()
        logger.debug("Considering fixed SoS-value in reconstruction algorithm")
        delays = torch.sqrt((yy * spacing_in_mm - sensor_positions[:, 2][jj]) ** 2 +
                            (xx * spacing_in_mm - sensor_positions[:, 0][jj]) ** 2 +
                            (zz * spacing_in_mm - sensor_positions[:, 1][jj]) ** 2) \
            / (speed_of_sound_in_m_per_s * time_spacing_in_ms)
    else:
        jj = j.long()
        start = time.time() #TODO: DELETE
        OWN_INTERPOLATION = True # TODO ########################################################################
        if OWN_INTERPOLATION: #slower but probably more senseful
            delays = calculate_delays_for_heterogen_sos(sensor_positions, xdim, ydim, zdim,
                        x, y, z, spacing_in_mm, time_spacing_in_ms, speed_of_sound_in_m_per_s, 
                        n_sensor_elements, global_settings, device_base_position_mm, logger,
                        torch_device)
        else: #faster but maybe offset issues (see analytical solution comparison)
            delays = calculate_delays_for_heterogen_sos_OLD(sensor_positions, xdim, ydim, zdim,
                        x, y, z, spacing_in_mm, time_spacing_in_ms, speed_of_sound_in_m_per_s, 
                        n_sensor_elements, global_settings, device_base_position_mm, logger,
                        torch_device)
        end = time.time()
        print("TIME", end-start)
   
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

# TODO: Leave just one of the following implementations 
#  - calculate_delays_for_heterogen_sos_OLD uses grid_sample and seems more buggy (due to weird system) [comparison to analytical solution
#     suggests that its less accurate] but is really fast: TIME 5.872145652770996
#  - calculate_delays_for_heterogen_sos uses own interpolation and seems to be more accurate, but it is really slow: TIME 75.16549825668335
# TODO: delete the grid_points_middle boolean after checking which implementation is better for big spacing
# TODO: delete the debug parameter

def calculate_delays_for_heterogen_sos_OLD(sensor_positions: torch.tensor, xdim: int, ydim: int, zdim: int,
                                           x: torch.tensor, y: torch.tensor, z: torch.tensor,
                                           spacing_in_mm: float, time_spacing_in_ms: float,
                                           speed_of_sound_in_m_per_s: np.ndarray,
                                           n_sensor_elements: int, global_settings: Settings, 
                                           device_base_position_mm: np.ndarray, logger: Logger,
                                           torch_device: torch.device,
                                           grid_points_middle: bool = True, global_ds_calc: bool = False,
                                           get_ds: bool = False, get_interpols: bool = False, verbose: bool = False,
                                           uneven_steps: bool = False) -> torch.tensor:
    """
    Returns the delays indicating which time series data has to be summed up taking a heterogenous 
    speed-of-sound-map into account, i.e. performing a line integral over the inverse speed-of-sound map.
    
    :param sensor_positions: sensor positions in mm in the FOV coordinate system,
                             where the origin is at the device base position
    :type sensor_positions: torch.tensor
    :param xdim: number of x values of sources in FOV
    :type xdim: int
    :param ydim: number of y values of sources in FOV
    :type ydim: int
    :param zdim: number of z values of sources in FOV
    :type zdim: int
    :param x: pixel indices in x-direction in FOV coordinate system
    :type x: torch.tensor
    :param y: pixel indices in y-direction in FOV coordinate system
    :type y: torch.tensor
    :param z: pixel indices in z-direction in FOV coordinate system
    :param spacing_in_mm: spacing of voxels in reconstructed image (FOV) in mm
    :type spacing_in_mm: float
    :param time_spacing_in_ms: temporal spacing of the time series data in ms
    :type time_spacing_in_ms: float
    :param speed_of_sound_in_m_per_s: (heterogenous) speed-of-sound map
    :type speed_of_sound_in_m_per_s: np.ndarray
    :param n_sensor_elements: number of sensor elements of the given 
    :type 
    :param global settings: (Settings) settings for the whole simulation
    :param device_base_position_mm: (np.ndarray) position of the device base in mm
    :param logger: (logger) logger instance in order to log and print warnings/errors/debug hints
    :param torch_device: (torch.device) either cpu or cuda GPU device used for the tensors

    :return: delays indices indicating which time series data
             one has to sum up
    :rtype: torch.tensor
    """
    logger.debug("Considering heterogenous SoS-map in reconstruction algorithm")

    STEPS_MIN = 2
    UNEVEN_STEPS = uneven_steps

    ############################
    mixed_coordinate_system_calculation = True #TODO: decide for one implementation,
    # so far mixed_coordinate_system_calculation is more accurate!!!!
    if global_ds_calc:
        mixed_coordinate_system_calculation = False

    if zdim == 1:
        # get relevant sos slice
        transducer_plane = int(round((device_base_position_mm[1]/global_settings[Tags.SPACING_MM]))) - 1
        # take wanted sos slice and invert values
        sos_map = 1/speed_of_sound_in_m_per_s[:,transducer_plane,:]
        # convert it to tensor and desired shape
        sos_tensor = torch.from_numpy(sos_map.T) # has to be transposed!
        sos_tensor = sos_tensor.unsqueeze(0) # 1 x H x W
        sos_tensor = sos_tensor.unsqueeze(0) # 1 x 1 x H x W
        sos_tensor = sos_tensor.to(torch_device)

        xx, yy = torch.meshgrid(x, y)

        device_base_position_mm = torch.from_numpy(device_base_position_mm)
        device_base_position_mm = device_base_position_mm.to(torch_device)

        if mixed_coordinate_system_calculation: #TODO: MORE ACCURATE
            xx = xx*spacing_in_mm
            yy = yy*spacing_in_mm
            sensor_positions = sensor_positions.clone()[:,::2] # leave out 3rd dimension

            # calculate the step size of the different rays
            source_pos_in_mm = torch.dstack([xx,yy])
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()              

            # scale the source and the sensor positions to [-1,1] range (needed for interpolator-function),
            # where -1 denotes x=0 and 1 denots x = 'volume_x_dim_mm' and same for y
            volume_dim_mm_vector = torch.tensor([global_settings[Tags.DIM_VOLUME_X_MM],
                                                global_settings[Tags.DIM_VOLUME_Z_MM]],
                                                device=torch_device)
            device_base_position_xy_mm = device_base_position_mm[::2] # leave out second component
            
            if not grid_points_middle:
                source_pos_in_mm += global_settings[Tags.SPACING_MM]/2
                sensor_positions += global_settings[Tags.SPACING_MM]/2

            source_pos_scaled = 2*(source_pos_in_mm+device_base_position_xy_mm)/volume_dim_mm_vector-1 
            sensor_positions_scaled = 2*(sensor_positions+device_base_position_xy_mm)/volume_dim_mm_vector-1
        else:
            # global position according to whole image in mm
            xx = xx*spacing_in_mm + device_base_position_mm[0]
            yy = yy*spacing_in_mm + device_base_position_mm[2]
            sensor_positions = sensor_positions + device_base_position_mm
            sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension

            # calculate the distances ds of all sources and detector sensors size (i.e. for all rays)
            source_pos_in_mm = torch.dstack([xx,yy])
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()          

            # scale the source and the sensor positions to [-1,1] range (needed for interpolator-function),
            # where -1 denotes x=0 and 1 denots x = 'volume_x_dim_mm' and same for y
            volume_dim_mm_vector = torch.tensor([global_settings[Tags.DIM_VOLUME_X_MM],
                                                global_settings[Tags.DIM_VOLUME_Z_MM]],
                                                device=torch_device)

            if not grid_points_middle:
                source_pos_in_mm = source_pos_in_mm + global_settings[Tags.SPACING_MM]/2
                sensor_positions = sensor_positions + global_settings[Tags.SPACING_MM]/2

            source_pos_scaled = 2*source_pos_in_mm/volume_dim_mm_vector-1 
            sensor_positions_scaled = 2*sensor_positions/volume_dim_mm_vector-1
        # free some unneeded gpu memory
        del volume_dim_mm_vector
        del source_pos_in_mm
        del sensor_positions

        # calculate the number of steps per ray, we want that for the longest ray that the ray is sampled at least
        # at every pixel
        steps = int(max((xdim**2+ydim**2)**0.5 + 0.5, ds.max()/(2**0.5*spacing_in_mm) + 0.5, STEPS_MIN))
        if steps%2 == 0 and UNEVEN_STEPS:
            steps +=1
        logger.debug(f"Using {steps} steps for numerical calculation of the line integrals")
        # steps        
        s = torch.linspace(0, 1, steps, device=torch_device)

        delays = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)

        # loop over the sensors because vectorization of the loop needs too much memory
        for sensor in range(n_sensor_elements):
            # calculate the sampled positions (#steps samples) for all rays coming from all source points to a given sensor
            rays = (1-s[None, None, :, None])* source_pos_scaled[:, :, None, :] +  \
                (s[None, None, :, None] * sensor_positions_scaled[None, None, sensor, None, :]) # shape: xdim x ydim x #steps x 2
            rays = rays.reshape(1, xdim*ydim, steps, 2)
            # calculate the corresponding interpolated sos values at those points
            interpolated_sos =  torch.nn.functional.grid_sample(sos_tensor, rays, mode="bilinear",
                                                                padding_mode="border", align_corners=False)
            ###################################################
            if verbose:
                print(interpolated_sos.reshape(xdim*ydim,steps))
                print(rays.reshape(xdim*ydim,steps,2))
            #####################################################
            # integrate the interpolated inverse sos-values
            integrals = torch.trapezoid(interpolated_sos)
            integrals = integrals.reshape((xdim, ydim))
            # calculate the delays by multiplying it with the step size ds/(steps-1) and divide it by the time spacing
            # in order to get the delay indices for the time series data
            delays[:,:, sensor] = integrals*ds[:,:,sensor]/((steps-1) * time_spacing_in_ms)

        if get_ds:
            if get_interpols:
                return delays.unsqueeze(2), ds, interpolated_sos.cpu().numpy() #TODO: delete
            else:
                return delays.unsqueeze(2), ds #TODO: delete
        else:
            return delays.unsqueeze(2) # xdim x ydim x 1 x #sensors

    else:
        logger.warning("3-dimensional heterogenous SoS reconstruction is not implemented yet.")

        xx, yy, zz = torch.meshgrid(x, y, z)


def calculate_delays_for_heterogen_sos(sensor_positions: torch.tensor, xdim: int, ydim: int, zdim: int,
                                       x: torch.tensor, y: torch.tensor, z: torch.tensor,
                                       spacing_in_mm: float, time_spacing_in_ms: float,
                                       speed_of_sound_in_m_per_s: np.ndarray,
                                       n_sensor_elements: int, global_settings: Settings, 
                                       device_base_position_mm: np.ndarray, logger: Logger,
                                       torch_device: torch.device,
                                       grid_points_middle: bool = True, global_ds_calc: bool = False,
                                       get_ds: bool = False, get_interpols: bool = False, verbose: bool = False,
                                       uneven_steps: bool = False) -> torch.tensor:
    """
    Returns the delays indicating which time series data has to be summed up taking a heterogenous 
    speed-of-sound-map into account, i.e. performing a line integral over the inverse speed-of-sound map.
    
    :param sensor_positions: sensor positions in mm in the FOV coordinate system,
                             where the origin is at the device base position
    :type sensor_positions: torch.tensor
    :param xdim: number of x values of sources in FOV
    :type xdim: int
    :param ydim: number of y values of sources in FOV
    :type ydim: int
    :param zdim: number of z values of sources in FOV
    :type zdim: int
    :param x: pixel indices in x-direction in FOV coordinate system
    :type x: torch.tensor
    :param y: pixel indices in y-direction in FOV coordinate system
    :type y: torch.tensor
    :param z: pixel indices in z-direction in FOV coordinate system
    :param spacing_in_mm: spacing of voxels in reconstructed image (FOV) in mm
    :type spacing_in_mm: float
    :param time_spacing_in_ms: temporal spacing of the time series data in ms
    :type time_spacing_in_ms: float
    :param speed_of_sound_in_m_per_s: (heterogenous) speed-of-sound map
    :type speed_of_sound_in_m_per_s: np.ndarray
    :param n_sensor_elements: number of sensor elements of the given 
    :type 
    :param global settings: (Settings) settings for the whole simulation
    :param device_base_position_mm: (np.ndarray) position of the device base in mm
    :param logger: (logger) logger instance in order to log and print warnings/errors/debug hints
    :param torch_device: (torch.device) either cpu or cuda GPU device used for the tensors

    :return: delays indices indicating which time series data
             one has to sum up
    :rtype: torch.tensor
    """
    logger.debug("Considering heterogenous SoS-map in reconstruction algorithm")



    #################################
    #torch.cuda.empty_cache()
    memo_debug = False
    if memo_debug:
        print("MEMORY CORRECTED (ACTUALLY WORSE")
        print("AT BEGINNING:")
        print_memory_stats()
    ############################

    STEPS_MIN = 2
    UNEVEN_STEPS = uneven_steps

    ############################ TODO:Decide###############################
    mixed_coordinate_system_calculation = True #TODO: decide for one implementation,
    # so far mixed_coordinate_system_calculation is more accurate!!!!
    #######################################################################
    if global_ds_calc:
        mixed_coordinate_system_calculation = False


    if zdim == 1:
        # get relevant sos slice
        transducer_plane = int(round((device_base_position_mm[1]/global_settings[Tags.SPACING_MM]))) - 1
        # take wanted sos slice and invert values
        sos_map = 1/speed_of_sound_in_m_per_s[:,transducer_plane,:]
        # convert it to tensor and desired shape
        sos_tensor = torch.from_numpy(sos_map) # IMPORTANT: check whether to be or not transposed!
        sos_tensor = sos_tensor.to(torch_device)

        # calculate absolute positions, where origin is  the origin of the whole volume (volume coordinate system)
        xx, yy = torch.meshgrid(x, y)

        device_base_position_mm = torch.from_numpy(device_base_position_mm)
        device_base_position_mm = device_base_position_mm.to(torch_device)

        if memo_debug:
            print("Before ds:")
            print_memory_stats()

        if mixed_coordinate_system_calculation: #TODO: MORE ACCURATE
            print("fov calculation")
            xx = xx*spacing_in_mm
            yy = yy*spacing_in_mm
            sensor_positions = sensor_positions.clone()[:,::2] # leave out 3rd dimension

            # calculate the step size of the different rays
            source_pos_in_mm = torch.dstack([xx,yy])

            # calculate the distances  
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt() # needs a lot memory            

            device_base_position_xy_mm = device_base_position_mm[::2] # leave out second component
            # get global positions (in mm)
            source_pos_in_mm += device_base_position_xy_mm
            sensor_positions += device_base_position_xy_mm
        else:
            print("global_calculation")
            # global position according to whole image in mm
            xx = xx*spacing_in_mm + device_base_position_mm[0]
            yy = yy*spacing_in_mm + device_base_position_mm[2]
            sensor_positions = sensor_positions + device_base_position_mm
            sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension

            # calculate the distances ds of all sources and detector sensors size (i.e. for all rays)
            source_pos_in_mm = torch.dstack([xx,yy]) 

            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt() # needs a lot memory     
        # free some unneeded gpu memory
        del xx, yy

        ######################## TODO:DELETE
        if memo_debug:
            print("after ds:")
            print_memory_stats()
        #########################

        # calculate the number of steps per ray, we want that for the longest ray that the ray is sampled at least
        # at every pixel
        steps = int(max((xdim**2+ydim**2)**0.5 + 0.5, ds.max()/(2**0.5*spacing_in_mm) + 0.5, STEPS_MIN))
        if steps%2 == 0 and UNEVEN_STEPS:
            steps +=1
        logger.debug(f"Using {steps} steps for numerical calculation of the line integrals")
        # steps        
        s = torch.linspace(0, 1, steps, device=torch_device)

        delays = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)

        # loop over the sensors because vectorization of the loop needs too much memory
        for sensor in range(n_sensor_elements):
            # calculate the sampled positions (#steps samples) for all rays coming from all source points to a given sensor
            rays = (1-s[None, None, :, None])* source_pos_in_mm[:, :, None, :] +  \
                (s[None, None, :, None] * sensor_positions[None, None, sensor, None, :]) # shape: xdim x ydim x #steps x 2           
            
            # translate mm into pixel values
            rays = rays/global_settings[Tags.SPACING_MM]

            if grid_points_middle:
                rays -= 0.5

            rays = rays.reshape(xdim*ydim*steps,2)
           
            ###########TODO:Delete######
            if memo_debug:
                print(sensor)  
                print("Before interpolation:")
                print_memory_stats()
                #sys.exit()

            # calculate the corresponding interpolated sos values at those points
            interpolated_sos = bilinear_interpolation(sos_tensor, rays[:,0], rays[:,1])
            interpolated_sos = interpolated_sos.reshape(xdim*ydim, steps)
            ##########TODO: Delete##############
            if verbose:
                print(interpolated_sos)
                print(rays.reshape(xdim*ydim,steps,2))
            ########################
            # integrate the interpolated inverse sos-values
            integrals = torch.trapezoid(interpolated_sos)
            integrals = integrals.reshape((xdim, ydim))
            # calculate the delays by multiplying it with the step size ds/(steps-1) and divide it by the time spacing
            # in order to get the delay indices for the time series data
            delays[:,:, sensor] = integrals*ds[:,:,sensor]/((steps-1) * time_spacing_in_ms) 
            

            # to reduce memory allocation for the next run
            if not get_interpols: # TODO
                del integrals, interpolated_sos, rays
            else:
                del integrals, rays

            #TODO delete       
            ############ MEMORY PROBLEM FIX ATTEMPT ############
            # problem : always in global system calculation!!
            #ds = (source_pos_in_mm - sensor_positions[None, None, sensor, :]).pow(2).sum(-1).sqrt()
            #delays[:,:, sensor] = integrals*ds/((steps-1) * time_spacing_in_ms)
            #######################         
     
            
        if get_ds:
            if get_interpols:
                return delays.unsqueeze(2), ds, interpolated_sos.cpu().numpy() #TODO: delete
            else:
                return delays.unsqueeze(2), ds #TODO: delete
        else:
            return delays.unsqueeze(2) # xdim x ydim x 1 x #sensors

    else:
        logger.warning("3-dimensional heterogenous SoS reconstruction is not implemented yet.")

        xx, yy, zz = torch.meshgrid(x, y, z)


def calculate_delays_for_heterogen_sos_mem(sensor_positions: torch.tensor, xdim: int, ydim: int, zdim: int,
                                       x: torch.tensor, y: torch.tensor, z: torch.tensor,
                                       spacing_in_mm: float, time_spacing_in_ms: float,
                                       speed_of_sound_in_m_per_s: np.ndarray,
                                       n_sensor_elements: int, global_settings: Settings, 
                                       device_base_position_mm: np.ndarray, logger: Logger,
                                       torch_device: torch.device,
                                       grid_points_middle: bool = True, global_ds_calc: bool = False,
                                       get_ds: bool = False, get_interpols: bool = False, verbose: bool = False,
                                       uneven_steps: bool = False) -> torch.tensor:
    """
    Returns the delays indicating which time series data has to be summed up taking a heterogenous 
    speed-of-sound-map into account, i.e. performing a line integral over the inverse speed-of-sound map.
    
    :param sensor_positions: sensor positions in mm in the FOV coordinate system,
                             where the origin is at the device base position
    :type sensor_positions: torch.tensor
    :param xdim: number of x values of sources in FOV
    :type xdim: int
    :param ydim: number of y values of sources in FOV
    :type ydim: int
    :param zdim: number of z values of sources in FOV
    :type zdim: int
    :param x: pixel indices in x-direction in FOV coordinate system
    :type x: torch.tensor
    :param y: pixel indices in y-direction in FOV coordinate system
    :type y: torch.tensor
    :param z: pixel indices in z-direction in FOV coordinate system
    :param spacing_in_mm: spacing of voxels in reconstructed image (FOV) in mm
    :type spacing_in_mm: float
    :param time_spacing_in_ms: temporal spacing of the time series data in ms
    :type time_spacing_in_ms: float
    :param speed_of_sound_in_m_per_s: (heterogenous) speed-of-sound map
    :type speed_of_sound_in_m_per_s: np.ndarray
    :param n_sensor_elements: number of sensor elements of the given 
    :type 
    :param global settings: (Settings) settings for the whole simulation
    :param device_base_position_mm: (np.ndarray) position of the device base in mm
    :param logger: (logger) logger instance in order to log and print warnings/errors/debug hints
    :param torch_device: (torch.device) either cpu or cuda GPU device used for the tensors

    :return: delays indices indicating which time series data
             one has to sum up
    :rtype: torch.tensor
    """
    logger.debug("Considering heterogenous SoS-map in reconstruction algorithm")

    ############################
    #torch.cuda.empty_cache()
    memo_debug = False
    print("NOT MEMORY CORRECTED (ACTUALLY BETTER)")
    if memo_debug:
        print("AT BEGINNING:")
        print_memory_stats()
    ############################

    STEPS_MIN = 2
    UNEVEN_STEPS = uneven_steps
    
    ############################ TODO:Decide
    mixed_coordinate_system_calculation = True #TODO: decide for one implementation,
    # so far mixed_coordinate_system_calculation is more accurate!!!!
    if global_ds_calc:
        mixed_coordinate_system_calculation = False


    if zdim == 1:
        # get relevant sos slice
        transducer_plane = int(round((device_base_position_mm[1]/global_settings[Tags.SPACING_MM]))) - 1
        # take wanted sos slice and invert values
        sos_map = 1/speed_of_sound_in_m_per_s[:,transducer_plane,:]
        # convert it to tensor and desired shape
        sos_tensor = torch.from_numpy(sos_map) # IMPORTANT: check whether to be or not transposed!
        sos_tensor = sos_tensor.to(torch_device)

        # calculate absolute positions, where origin is  the origin of the whole volume (volume coordinate system)
        xx, yy = torch.meshgrid(x, y)

        device_base_position_mm = torch.from_numpy(device_base_position_mm)
        device_base_position_mm = device_base_position_mm.to(torch_device)

        if memo_debug:
            print("Before ds:")
            print_memory_stats()

        if mixed_coordinate_system_calculation: #TODO: MORE ACCURATE
            xx = xx*spacing_in_mm
            yy = yy*spacing_in_mm
            sensor_positions = sensor_positions.clone()[:,::2] # leave out 3rd dimension

            # calculate the step size of the different rays
            source_pos_in_mm = torch.dstack([xx,yy])

            # calculate the distances  
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()          

            device_base_position_xy_mm = device_base_position_mm[::2] # leave out second component
            # get global positions (in mm)
            source_pos_in_mm += device_base_position_xy_mm
            sensor_positions += device_base_position_xy_mm
            # get the global pixel indices
            if grid_points_middle:
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] - 0.5
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM] - 0.5  
            else:
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM]
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM]  
        else:
            # global position according to whole image in mm
            xx = xx*spacing_in_mm + device_base_position_mm[0]
            yy = yy*spacing_in_mm + device_base_position_mm[2]
            sensor_positions = sensor_positions + device_base_position_mm
            sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension

            # calculate the distances ds of all sources and detector sensors size (i.e. for all rays)
            source_pos_in_mm = torch.dstack([xx,yy]) 

            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()
            
            # get the global pixel indices
            if grid_points_middle:
                # take into account that the position of sos-map-pixel (i,j) is [i*spacing+spacing/2, j*spacing+spacing/2]
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] - 0.5 
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM] - 0.5  
            else:
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM]
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM]

        # free some unneeded gpu memory
        del xx, yy
        del source_pos_in_mm
        del sensor_positions
        ######################## TODO:DELETE
        if memo_debug:
            print("after ds:")
            print_memory_stats()
        #########################

        # calculate the number of steps per ray, we want that for the longest ray that the ray is sampled at least
        # at every pixel
        steps = int(max((xdim**2+ydim**2)**0.5 + 0.5, ds.max()/(2**0.5*spacing_in_mm) + 0.5, STEPS_MIN)) # MEMORY PROBLEM
        if steps%2 == 0 and UNEVEN_STEPS:
            steps +=1
        logger.debug(f"Using {steps} steps for numerical calculation of the line integrals")
        # steps        
        s = torch.linspace(0, 1, steps, device=torch_device)

        delays = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)

        # loop over the sensors because vectorization of the loop needs too much memory
        for sensor in range(n_sensor_elements):
            # calculate the sampled positions (#steps samples) for all rays coming from all source points to a given sensor
            rays = (1-s[None, None, :, None])* source_pos_scaled[:, :, None, :] +  \
                (s[None, None, :, None] * sensor_positions_scaled[None, None, sensor, None, :]) # shape: xdim x ydim x #steps x 2           
            rays = rays.reshape(xdim*ydim*steps,2)
            
            ###########TODO:Delete######
            if memo_debug:  
                print("Before interpolation:")
                print_memory_stats()
            ####

            # calculate the corresponding interpolated sos values at those points
            interpolated_sos = bilinear_interpolation(sos_tensor, rays[:,0], rays[:,1])
            interpolated_sos = interpolated_sos.reshape(xdim*ydim, steps)
            ##########TODO: Delete##############
            if verbose:
                print(interpolated_sos)
                print(rays.reshape(xdim*ydim,steps,2))
            ########################
            # integrate the interpolated inverse sos-values
            integrals = torch.trapezoid(interpolated_sos)
            integrals = integrals.reshape((xdim, ydim))
            # calculate the delays by multiplying it with the step size ds/(steps-1) and divide it by the time spacing
            # in order to get the delay indices for the time series data
            delays[:,:, sensor] = integrals*ds[:,:,sensor]/((steps-1) * time_spacing_in_ms) #MEMORY PROBLEM
            

        if get_ds:
            if get_interpols:
                return delays.unsqueeze(2), ds, interpolated_sos.cpu().numpy() #TODO: delete
            else:
                return delays.unsqueeze(2), ds #TODO: delete
        else:
            return delays.unsqueeze(2) # xdim x ydim x 1 x #sensors

    else:
        logger.warning("3-dimensional heterogenous SoS reconstruction is not implemented yet.")

        xx, yy, zz = torch.meshgrid(x, y, z)