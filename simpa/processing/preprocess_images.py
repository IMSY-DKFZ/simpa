"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.log import Logger
logger = Logger()
import numpy as np
import torch


def crop(image_data, height_start, width_start, target_height, target_width):
    """
    crop the given image.

    :param image_data: (numpy array) Image to be cropped.
    :param height_start: (int) height_start in (height_start,width_start) i.e coordinates of the upper left corner.
    :param width_start: (int) width_start in (height_start,width_start) i.e coordinates of the upper left corner.
    :param target_height: (int) Height of the cropped image.
    :param target_width: (int) Width of the cropped image.:
    :return: Cropped image.
    """

    return image_data[height_start:height_start + target_height, width_start:width_start + target_width]


def top_center_crop(image_data, output_size):
    """
    Center crop the given image.

    :param image_data: (numpy array) Image to be cropped.
    :param output_size: (int, list or tuple) Size as (height, width) of the cropped image.
                        If given as int, the output size will be quadratic.
    :return: Cropped image.
    """

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif not (isinstance(output_size, tuple) or isinstance(output_size, list)):
        raise TypeError("Output size has to be tuple or list.")

    target_height, target_width = output_size
    input_height, input_width = image_data.shape

    width_start = int((input_width - target_width)/2)

    return crop(image_data, 0, width_start, target_height, target_width)


def top_center_crop_power_two(image_data):
    """
    Center crop the given image to the sizes of the largest powers of two smaller
    than the image height and width, respectively.

    :param image_data: (numpy array) Image to be preprocessed.
    :return: Cropped image.
    """

    input_height, input_width = image_data.shape
    target_height = 1 << (input_height.bit_length() - 1)
    target_width = 1 << (input_width.bit_length() - 1)

    return top_center_crop(image_data, (target_height, target_width))


def min_max_normalization(data: np.ndarray = None) -> np.ndarray:
    """
    Normalizes the given data by applying min max normalization.
    The resulting array has values between 0 and 1 inclusive.

    :param data: (numpy array) data to be normalized
    :return: (numpy array) normalized array
    """

    if data is None:
        raise AttributeError("Data must not be none in order to normalize it.")

    min = data.min()
    max = data.max()
    output = (data - min) / (max - min)

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
