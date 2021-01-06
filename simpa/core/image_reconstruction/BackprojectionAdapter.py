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

from simpa.core.image_reconstruction import ReconstructionAdapterBase
from simpa.utils import Tags
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.io_handling.io_hdf5 import load_hdf5
from simpa.core.device_digital_twins import DEVICE_MAP

import numpy as np
import torch


class BackprojectionAdapter(ReconstructionAdapterBase):

    def reconstruction_algorithm(self, time_series_sensor_data, settings):

        time_series_sensor_data = np.swapaxes(time_series_sensor_data, 0, 1)

        acoustic_data_path = generate_dict_path(settings, Tags.PROPERTY_SPEED_OF_SOUND,
                                                wavelength=settings[Tags.WAVELENGTH], upsampled_data=True)
        sound_speed_m = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], acoustic_data_path)[Tags.PROPERTY_SPEED_OF_SOUND]
        sound_speed_m = np.mean(sound_speed_m)

        target_dim_m = np.asarray([settings[Tags.DIM_VOLUME_X_MM]/1000, settings[Tags.DIM_VOLUME_Y_MM]/1000,
                                  settings[Tags.DIM_VOLUME_Z_MM]/1000])

        resolution_m = settings[Tags.SPACING_MM]/1000

        sampling_frequency = settings[Tags.SENSOR_SAMPLING_RATE_MHZ] * 1000000
        if Tags.RECONSTRUCTION_MODE in settings:
            mode = settings[Tags.RECONSTRUCTION_MODE]
        else:
            mode = Tags.RECONSTRUCTION_MODE_DIFFERENTIAL

        device = DEVICE_MAP[settings[Tags.DIGITAL_DEVICE]]

        sensor_positions = device.get_detector_element_positions_mm(settings) / 1000
        sensor_positions[:, 1] = 0
        sensor_positions[:, [1, 2]] = sensor_positions[:, [2, 1]]

        target_dim_m[[0, 1]] = target_dim_m[[1, 0]]
        print(sound_speed_m)
        print(target_dim_m)
        print(resolution_m)
        print(sampling_frequency)

        return self.backprojectionND_torch(time_series_sensor_data, speed_of_sound_m=sound_speed_m,
                                           target_dim_m=target_dim_m, resolution_m=resolution_m,
                                           sensor_positions_m=sensor_positions, sampling_frequency=sampling_frequency,
                                           mode=mode)


    def backprojectionND_torch(self, time_series_data, speed_of_sound_m, target_dim_m, resolution_m,
                               sensor_positions_m, sampling_frequency, mode=None):
        """ ND packprojection """

        if mode is None:
            mode = Tags.RECONSTRUCTION_MODE_DIFFERENTIAL

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(np.shape(time_series_data))

        time_series_data = torch.from_numpy(time_series_data.copy()).float().to(device)
        target_dim_m = torch.from_numpy(target_dim_m.copy()).float().to(device)
        sensor_positions_m = torch.from_numpy(sensor_positions_m.copy()).float().to(device)

        num_detectors = time_series_data.shape[1]
        num_samples = time_series_data.shape[0]
        time_per_sample_s = 1/sampling_frequency

        time_vector = torch.arange(0, num_samples * time_per_sample_s, time_per_sample_s).to(device)
        target_dim_voxels = torch.round(torch.true_divide(target_dim_m, resolution_m)).int().to(device)

        sizeT = len(time_vector)
        back_projection = torch.zeros(*target_dim_voxels, names=None).float().to(device)

        edges_m = torch.true_divide(target_dim_voxels, 2) * resolution_m
        gridsizes = [torch.linspace(edges_m[i], -edges_m[i], target_dim_voxels[i]) for i in range(len(target_dim_voxels))]

        mesh = torch.zeros([3, len(gridsizes[0]), len(gridsizes[1]), len(gridsizes[2])], names=None).to(device)
        torch.stack(torch.meshgrid(*gridsizes), out=mesh)

        for i in range(num_detectors):

            time_series_pressure_of_detection_element = time_series_data[:, i]
            time_derivative_pressure = (time_series_pressure_of_detection_element[1:] -
                                        time_series_pressure_of_detection_element[0:-1])
            time_derivative_pressure = torch.cat([time_derivative_pressure, torch.zeros([1], names=None).to(device)])
            time_derivative_pressure = torch.mul(time_derivative_pressure, 1 / time_per_sample_s)
            time_derivative_pressure = torch.mul(time_derivative_pressure, time_vector)

            detectors = torch.add(mesh, torch.reshape(-sensor_positions_m[i], (-1, 1, 1, 1)))

            distance_to_meshpoints_m = torch.sqrt(torch.sum(torch.square(detectors), axis=0))
            time_delay_s = (distance_to_meshpoints_m / speed_of_sound_m)
            time_delay_voxels = time_delay_s / time_per_sample_s
            time_delay_voxels_indices = torch.round(time_delay_voxels).int() - 1

            # correct for out of bounds indices
            time_delay_voxels_indices[time_delay_voxels_indices >= sizeT] = sizeT - 1
            time_delay_voxels_indices[time_delay_voxels_indices <= 0] = 0
            time_delay_voxels_indices = time_delay_voxels_indices.long()

            pressure = time_series_pressure_of_detection_element[time_delay_voxels_indices]
            derivative_pressure = time_derivative_pressure[time_delay_voxels_indices]

            if mode == Tags.RECONSTRUCTION_MODE_PRESSURE:
                back_projection = back_projection + pressure
            elif mode == Tags.RECONSTRUCTION_MODE_DIFFERENTIAL:
                back_projection = back_projection - derivative_pressure
            else:
                back_projection = back_projection + pressure - derivative_pressure

        return back_projection.cpu().numpy()