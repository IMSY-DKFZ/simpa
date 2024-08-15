# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.core.simulation_modules.reconstruction_module import ReconstructionAdapterBase
from simpa.core.device_digital_twins import DetectionGeometryBase
import numpy as np
import torch
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import compute_delay_and_sum_values, \
    preparing_reconstruction_and_obtaining_reconstruction_settings, compute_image_dimensions
from simpa.core.simulation_modules.reconstruction_module import create_reconstruction_settings


class SignedDelayMultiplyAndSumAdapter(ReconstructionAdapterBase):

    def reconstruction_algorithm(self, time_series_sensor_data, detection_geometry: DetectionGeometryBase):
        """
        Applies the signed Delay Multiply and Sum beamforming algorithm [1] to the time series sensor data
        (2D numpy array where the
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
            detection_geometry, spacing_in_mm, self.logger)

        if zdim == 1:
            sensor_positions[:, 1] = 0  # Assume imaging plane

        # construct output image
        output = torch.zeros((xdim, ydim, zdim), dtype=torch.float32, device=torch_device)

        values, n_sensor_elements = compute_delay_and_sum_values(time_series_sensor_data, sensor_positions, xdim,
                                                                 ydim, zdim, xdim_start, xdim_end, ydim_start, ydim_end,
                                                                 zdim_start, zdim_end, spacing_in_mm, speed_of_sound_in_m_per_s,
                                                                 time_spacing_in_ms, self.logger, torch_device,
                                                                 self.component_settings)

        DAS = torch.sum(values, dim=3)

        for x in range(xdim):
            yy, zz, nn, mm = torch.meshgrid(torch.arange(ydim, device=torch_device),
                                            torch.arange(zdim, device=torch_device),
                                            torch.arange(n_sensor_elements, device=torch_device),
                                            torch.arange(n_sensor_elements, device=torch_device))
            M = values[x, yy, zz, nn] * values[x, yy, zz, mm]
            M = torch.sign(M) * torch.sqrt(torch.abs(M))
            # only take upper triangle without diagonal and sum up along n and m axis (last two)
            output[x] = torch.triu(M, diagonal=1).sum(dim=(-1, -2))
        output = torch.sign(DAS) * output
        reconstructed = output.cpu().numpy()

        return reconstructed.squeeze()


def reconstruct_signed_delay_multiply_and_sum_pytorch(time_series_sensor_data: np.ndarray,
                                                      detection_geometry: DetectionGeometryBase,
                                                      speed_of_sound_in_m_per_s: int = 1540,
                                                      time_spacing_in_s: float = 2.5e-8,
                                                      sensor_spacing_in_mm: float = 0.1,
                                                      recon_mode: str = Tags.RECONSTRUCTION_MODE_PRESSURE,
                                                      apodization: str = Tags.RECONSTRUCTION_APODIZATION_BOX
                                                      ) -> np.ndarray:
    """
    Convenience function for reconstructing time series data using Delay and Sum algorithm implemented in PyTorch

    :param time_series_sensor_data: (2D numpy array) sensor data of shape (sensor elements, time steps)
    :param speed_of_sound_in_m_per_s: (int) speed of sound in medium in meters per second (default: 1540 m/s)
    :param time_spacing_in_s: (float) time between sampling points in seconds (default: 2.5e-8 s which is equal to 40 MHz)
    :param sensor_spacing_in_mm: (float) space between sensor elements in millimeters (default: 0.1 mm)
    :param recon_mode: SIMPA Tag defining the reconstruction mode - pressure default OR differential
    :param apodization: SIMPA Tag defining the apodization function (default box)
    :return: (2D numpy array) reconstructed image as 2D numpy array
    """
    # create settings
    settings = create_reconstruction_settings(speed_of_sound_in_m_per_s, time_spacing_in_s, sensor_spacing_in_mm,
                                              recon_mode, apodization)
    adapter = SignedDelayMultiplyAndSumAdapter(settings)
    return adapter.reconstruction_algorithm(time_series_sensor_data, detection_geometry)
