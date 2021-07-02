"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.core.reconstruction_module import ReconstructionAdapterBase
import numpy as np
import torch
from simpa.utils.settings import Settings
from simpa.core.reconstruction_module.reconstruction_utils import compute_delay_and_sum_values, compute_image_dimensions, \
    apply_b_mode, preparing_reconstruction_and_obtaining_reconstruction_settings
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

        time_series_sensor_data, sensor_positions, speed_of_sound_in_m_per_s, spacing_in_mm, time_spacing_in_ms, torch_device = preparing_reconstruction_and_obtaining_reconstruction_settings(
            time_series_sensor_data, self.component_settings, self.global_settings, detection_geometry, self.logger)

        ### ALGORITHM ITSELF ###

        xdim, zdim, ydim = compute_image_dimensions(
            detection_geometry, spacing_in_mm, speed_of_sound_in_m_per_s, self.logger)

        if zdim == 1:
            sensor_positions[:, 1] = 0  # Assume imaging plane

        # construct output image
        output = torch.zeros((xdim, ydim, zdim), dtype=torch.float32, device=torch_device)

        values, _ = compute_delay_and_sum_values(time_series_sensor_data, sensor_positions, xdim,
                                                 ydim, zdim, spacing_in_mm, speed_of_sound_in_m_per_s,
                                                 time_spacing_in_ms, self.logger, torch_device,
                                                 self.component_settings)

        _sum = torch.sum(values, dim=3)
        counter = torch.count_nonzero(values, dim=3)
        torch.divide(_sum, counter, out=output)

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
