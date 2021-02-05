# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.utils import Tags
from simpa.core.image_reconstruction import ReconstructionAdapterBase
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import load_hdf5
from simpa.core.device_digital_twins import DEVICE_MAP
import numpy as np
import torch


class PyTorchDASAdapter(ReconstructionAdapterBase):
    def reconstruction_algorithm(self, time_series_sensor_data, settings):
        """
        Applies the Delay and Sum beamforming algorithm [1] to the time series sensor data (2D numpy array where the first dimension corresponds to the sensor elements
        and the second to the recorded time steps) with the given beamforming settings (dictionary).
        A reconstructed image (2D numpy array) is returned.
        This implementation uses PyTorch Tensors to perform computations and is able to run on GPUs.

        [1] T. Kirchner et al. 2018, "Signed Real-Time Delay Multiply and Sum Beamformingfor Multispectral Photoacoustic Imaging", https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi3hZjA48jtAhUM6OAKHWK-BuAQFjAAegQIBxAC&url=https%3A%2F%2Fwww.mdpi.com%2F2313-433X%2F4%2F10%2F121%2Fpdf&usg=AOvVaw3CCZEt7L_xoUbWvlW1Ljx5
        """

        ### INPUT CHECKING AND VALIDATION ###

        # check settings dictionary for elements and read them in
        if Tags.WAVELENGTH in settings and settings[Tags.WAVELENGTH]:
            acoustic_data_path = generate_dict_path(settings, Tags.PROPERTY_SPEED_OF_SOUND,
                                                    wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)
            sound_speed_m = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], acoustic_data_path)[
                Tags.PROPERTY_SPEED_OF_SOUND]
            speed_of_sound_in_m_per_s = np.mean(sound_speed_m)
        else:
            raise AttributeError("Please specify a value for WAVELENGTH to obtain the average speed of sound")

        if Tags.K_WAVE_SPECIFIC_DT in settings and settings[Tags.K_WAVE_SPECIFIC_DT]:
            time_spacing_in_ms = settings[Tags.K_WAVE_SPECIFIC_DT] * 1000
        else:
            raise AttributeError("Please specify a value for K_WAVE_SPECIFIC_DT")

        if Tags.SPACING_MM in settings and settings[Tags.SPACING_MM]:
            sensor_spacing_in_mm = settings[Tags.SPACING_MM]
        else:
            raise AttributeError("Please specify a value for SPACING_MM")

        # get device specific sensor positions
        device = DEVICE_MAP[settings[Tags.DIGITAL_DEVICE]]
        device.check_settings_prerequisites(settings)
        device.adjust_simulation_volume_and_settings(settings)

        sensor_positions = device.get_detector_element_positions_accounting_for_device_position_mm(settings)
        sensor_positions = np.round(sensor_positions / sensor_spacing_in_mm).astype(int)
        sensor_positions = np.array(sensor_positions[:, [0, 2]])  # only use x and y positions and ignore z

        # time series sensor data must be numpy array
        if isinstance(sensor_positions, np.ndarray):
            sensor_positions = torch.from_numpy(sensor_positions)
        if isinstance(time_series_sensor_data, np.ndarray):
            time_series_sensor_data = torch.from_numpy(time_series_sensor_data)
        assert isinstance(time_series_sensor_data,
                          torch.Tensor), 'The time series sensor data must have been converted to a tensor'

        # move tensors to GPU if available, otherwise use CPU
        if Tags.GPU not in settings:
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        else:
            dev = "cuda" if settings[Tags.GPU] else "cpu"

        device = torch.device(dev)
        sensor_positions = sensor_positions.to(device)
        time_series_sensor_data = time_series_sensor_data.to(device)

        # array must be of correct dimension
        assert time_series_sensor_data.ndim == 2, 'Samples must have exactly 2 dimensions. ' \
                                                  'Apply beamforming per wavelength if you have a 3D array. '

        ### ALGORITHM ITSELF ###

        ## compute size of beamformed image ##
        xdim = (max(sensor_positions[:, 0]) - min(sensor_positions[:, 0]))
        xdim = int(xdim) + 1  # correction due to subtraction of indices starting at 0
        ydim = float(time_series_sensor_data.shape[1] * time_spacing_in_ms * speed_of_sound_in_m_per_s) / sensor_spacing_in_mm
        ydim = int(round(ydim))
        n_sensor_elements = time_series_sensor_data.shape[0]

        print(f'Number of pixels in X dimension: {xdim}, Y dimension: {ydim}, sensor elements: {n_sensor_elements}')

        # construct output image
        output = torch.zeros((xdim, ydim), dtype=torch.float32, device=device)

        xx, yy, jj = torch.meshgrid(torch.arange(xdim, device=device),
                                    torch.arange(ydim, device=device),
                                    torch.arange(n_sensor_elements, device=device))

        delays = torch.sqrt(((yy - sensor_positions[:, 1][jj]) * sensor_spacing_in_mm) ** 2 +
                            ((xx - torch.abs(sensor_positions[:, 0][jj])) * sensor_spacing_in_mm) ** 2) \
                            / (speed_of_sound_in_m_per_s * time_spacing_in_ms)

        delays = torch.round(delays).long()

        # perform index validation
        invalid_indices = torch.where(torch.logical_or(delays < 0, delays >= float(time_series_sensor_data.shape[1])))
        delays[invalid_indices] = 0

        values = time_series_sensor_data[jj, delays]

        # set values of invalid indices to 0 so that they don't influence the result
        values[invalid_indices] = 0
        sum = torch.sum(values, dim=2)
        counter = torch.count_nonzero(values, dim=2)
        torch.divide(sum, counter, out=output)

        return output.cpu().numpy()
