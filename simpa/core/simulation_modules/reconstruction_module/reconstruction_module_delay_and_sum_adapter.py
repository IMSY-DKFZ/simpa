# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.core.simulation_modules.reconstruction_module import ReconstructionAdapterBase
import numpy as np
import torch
from simpa.utils.settings import Settings
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import compute_delay_and_sum_values, compute_image_dimensions, \
    preparing_reconstruction_and_obtaining_reconstruction_settings
from simpa.core.device_digital_twins import DetectionGeometryBase


class DelayAndSumAdapter(ReconstructionAdapterBase):

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

        xdim, zdim, ydim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end = compute_image_dimensions(
            detection_geometry, spacing_in_mm, speed_of_sound_in_m_per_s, self.logger)

        if zdim == 1:
            sensor_positions[:, 1] = 0  # Assume imaging plane

        # construct output image
        output = torch.zeros((xdim, ydim, zdim), dtype=torch.float32, device=torch_device)

        values, _ = compute_delay_and_sum_values(time_series_sensor_data, sensor_positions, xdim,
                                                 ydim, zdim, xdim_start, xdim_end, ydim_start, ydim_end, zdim_start, zdim_end, spacing_in_mm, speed_of_sound_in_m_per_s,
                                                 time_spacing_in_ms, self.logger, torch_device,
                                                 self.component_settings)

        _sum = torch.sum(values, dim=3)
        counter = torch.count_nonzero(values, dim=3)
        torch.divide(_sum, counter, out=output)

        reconstructed = output.cpu().numpy()

        return reconstructed.squeeze()


def reconstruct_delay_and_sum_pytorch(time_series_sensor_data: np.ndarray,
                                      detection_geometry: DetectionGeometryBase,
                                      settings: dict = None) -> np.ndarray:
    """
    Convenience function for reconstructing time series data using Delay and Sum algorithm implemented in PyTorch

    :param time_series_sensor_data: (2D numpy array) sensor data of shape (sensor elements, time steps)
    :param detection_geometry: The DetectionGeometryBase that should be used to reconstruct the given time series data
    :param settings: (dict) settings dictionary: by default there is none and the other parameters are used instead,
                     but if parameters are given in the settings those will be used instead of parsed arguments)
    :return: (2D numpy array) reconstructed image as 2D numpy array
    """

    # create settings if they don't exist yet
    if settings is None:
        settings = Settings()

    # parse reconstruction settings if they are not given in the settings
    component_settings = settings.get_reconstruction_settings()
    if Tags.DATA_FIELD_SPEED_OF_SOUND not in component_settings or component_settings[Tags.DATA_FIELD_SPEED_OF_SOUND]\
            is None:
        if Tags.DATA_FIELD_SPEED_OF_SOUND in settings and settings[Tags.DATA_FIELD_SPEED_OF_SOUND] is not None:
            component_settings[Tags.DATA_FIELD_SPEED_OF_SOUND] = settings[Tags.DATA_FIELD_SPEED_OF_SOUND]

    if Tags.SENSOR_SAMPLING_RATE_MHZ not in component_settings or component_settings[Tags.SENSOR_SAMPLING_RATE_MHZ]\
            is None:
        if Tags.SENSOR_SAMPLING_RATE_MHZ in settings and settings[Tags.SENSOR_SAMPLING_RATE_MHZ] is not None:
            component_settings[Tags.SENSOR_SAMPLING_RATE_MHZ] = settings[Tags.SENSOR_SAMPLING_RATE_MHZ]

    if Tags.SPACING_MM not in component_settings or component_settings[Tags.SPACING_MM] is None:
        if Tags.SPACING_MM in settings and settings[Tags.SPACING_MM] is not None:
            component_settings[Tags.SPACING_MM] = settings[Tags.SPACING_MM]

    adapter = DelayAndSumAdapter(settings)
    return adapter.reconstruction_algorithm(time_series_sensor_data, detection_geometry)
