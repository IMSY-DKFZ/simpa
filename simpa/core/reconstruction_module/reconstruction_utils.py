"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
import torch
import torch.fft
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


def bandpass_filtering(data: torch.tensor = None, time_spacing_in_ms: float = None,
                       cutoff_lowpass: int = int(8e6), cutoff_highpass: int = int(0.1e6),
                       tukey_alpha: float = 0.5) -> torch.tensor:
    """
    Apply a bandpass filter with cutoff values at `cutoff_lowpass` and `cutoff_highpass` MHz 
    and a tukey window with alpha value of `tukey_alpha` inbetween on the `data` in Fourier space.

    :param data: (torch tensor) data to be filtered
    :param time_spacing_in_ms: (float) time spacing in milliseconds, e.g. 2.5e-5
    :param cutoff_lowpass: (int) Signal above this value will be ignored (in MHz)
    :param cutoff_highpass: (int) Signal below this value will be ignored (in MHz)
    :param tukey_alpha: (float) transition value between 0 (rectangular) and 1 (Hann window)
    :return: (torch tensor) filtered data
    """
    if data is None or time_spacing_in_ms is None:
        raise AttributeError("data and time spacing must be specified")

    # construct bandpass filter given the cutoff values and time spacing
    frequencies = np.fft.fftfreq(data.shape[1], d=time_spacing_in_ms/1000)

    if cutoff_highpass > cutoff_lowpass:
        raise ValueError("The highpass cutoff value must be lower than the lowpass cutoff value.")

    # find closest indices for frequencies
    small_index = (np.abs(frequencies - cutoff_highpass)).argmin()
    large_index = (np.abs(frequencies - cutoff_lowpass)).argmin()

    win = torch.tensor(tukey(large_index - small_index, alpha=tukey_alpha), device=data.device)
    window = torch.zeros(frequencies.shape, device=data.device)
    window[small_index:large_index] = win

    # transform data into Fourier space, multiply filter and transform back
    data_in_fourier_space = torch.fft.fft(data)
    filtered_data_in_fourier_space = data_in_fourier_space * window.expand_as(data_in_fourier_space)
    return torch.abs(torch.fft.ifft(filtered_data_in_fourier_space))


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