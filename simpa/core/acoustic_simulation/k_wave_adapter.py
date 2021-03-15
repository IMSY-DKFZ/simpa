# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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

import numpy as np
import subprocess
from simpa.utils import Tags, SaveFilePaths
from simpa.io_handling.io_hdf5 import load_hdf5, save_hdf5
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils.settings_generator import Settings
import os
import scipy.io as sio
from simpa.core.device_digital_twins import DEVICE_MAP
from simpa.core.acoustic_simulation import AcousticForwardAdapterBase


class KwaveAcousticForwardModel(AcousticForwardAdapterBase):
    """
    The KwaveAcousticForwardModel adapter enables acoustic simulations to be run with the
    k-wave MATLAB toolbox. k-Wave is a free toolbox (http://www.k-wave.org/) developed by Bradley Treeby
    and Ben Cox (University College London) and Jiri Jaros (Brno University of Technology).

    In order to use this toolbox, MATLAB needs to be installed on your system and the path to the
    MATLAB binary needs to be specified in the settings dictionary.

    In order to use the toolbox from with SIMPA, a number of parameters have to be specified in the
    settings dictionary::

        The initial pressure distribution:
            Tags.OPTICAL_MODEL_INITIAL_PRESSURE
        Acoustic tissue properties:
            Tags.PROPERTY_SPEED_OF_SOUND
            Tags.PROPERTY_DENSITY
            Tags.PROPERTY_ALPHA_COEFF
        The digital twin of the imaging device:
            Tags.DIGITAL_DEVICE
        Other parameters:
            Tags.PERFORM_UPSAMPLING
            Tags.SPACING_MM
            Tags.UPSCALE_FACTOR
            Tags.PROPERTY_ALPHA_POWER
            Tags.GPU
            Tags.PMLInside
            Tags.PMLAlpha
            Tags.PlotPML
            Tags.RECORDMOVIE
            Tags.MOVIENAME
            Tags.ACOUSTIC_LOG_SCALE
            Tags.SENSOR_DIRECTIVITY_PATTERN

    Many of these will be set automatically by SIMPA, but you may use the
    simpa.utils.settings_generator convenience methods to generate settings files that contain
    sensible defaults for these parameters.

    Please also refer to the simpa_examples scripts to see how the settings file can be
    parametrized successfully.

    """

    def forward_model(self, settings) -> np.ndarray:

        optical_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                          wavelength=settings[Tags.WAVELENGTH])

        print("OPTICAL_PATH", optical_path)

        data_dict = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], optical_path)

        tmp_ac_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], SaveFilePaths.SIMULATION_PROPERTIES)

        if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
            axes = (0, 1)
        else:
            axes = (0, 2)
        wavelength = str(settings[Tags.WAVELENGTH])
        data_dict[Tags.PROPERTY_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.PROPERTY_SPEED_OF_SOUND], 3, axes=axes)
        data_dict[Tags.PROPERTY_DENSITY] = np.rot90(tmp_ac_data[Tags.PROPERTY_DENSITY], 3, axes=axes)
        data_dict[Tags.PROPERTY_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.PROPERTY_ALPHA_COEFF], 3, axes=axes)
        data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.flip(
            np.rot90(data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE][wavelength], axes=axes))
        data_dict[Tags.OPTICAL_MODEL_FLUENCE] = np.flip(
            np.rot90(data_dict[Tags.OPTICAL_MODEL_FLUENCE][wavelength], axes=axes))

        PA_device = DEVICE_MAP[settings[Tags.DIGITAL_DEVICE]]
        PA_device.check_settings_prerequisites(settings)
        detector_positions_mm = PA_device.get_detector_element_positions_accounting_for_device_position_mm(settings)
        detector_positions_voxels = np.round(detector_positions_mm / settings[Tags.SPACING_MM]).astype(int)

        print("Number of detector elements:", len(detector_positions_voxels))

        sensor_map = np.zeros(np.shape(data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE]))
        if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
            sensor_map[detector_positions_voxels[:, 2], detector_positions_voxels[:, 0]] = 1
        else:
            half_y_dir_detector_pixels = int(round(0.5*PA_device.detector_element_length_mm/settings[Tags.SPACING_MM]))
            aranged_voxels = np.arange(- half_y_dir_detector_pixels, half_y_dir_detector_pixels, 1)

            if len(aranged_voxels) < 1:
                aranged_voxels = [0]

            for pixel in aranged_voxels:
                sensor_map[detector_positions_voxels[:, 2],
                           detector_positions_voxels[:, 1] + pixel,
                           detector_positions_voxels[:, 0]] = 1

        print("Number of ones in sensor_map:", np.sum(sensor_map))

        data_dict[Tags.PROPERTY_SENSOR_MASK] = sensor_map
        save_hdf5({Tags.PROPERTY_SENSOR_MASK: sensor_map}, settings[Tags.SIMPA_OUTPUT_PATH],
                  generate_dict_path(Tags.PROPERTY_SENSOR_MASK,
                                     wavelength=settings[Tags.WAVELENGTH]))

        try:
            data_dict[Tags.PROPERTY_DIRECTIVITY_ANGLE] = np.rot90(tmp_ac_data[Tags.PROPERTY_DIRECTIVITY_ANGLE], 3,
                                                                  axes=axes)
        except ValueError:
            print("No directivity_angle specified")
        except KeyError:
            print("No directivity_angle specified")

        optical_path = settings[Tags.SIMPA_OUTPUT_PATH] + ".mat"

        possible_k_wave_parameters = [Tags.PERFORM_UPSAMPLING, Tags.SPACING_MM, Tags.UPSCALE_FACTOR,
                                      Tags.PROPERTY_ALPHA_POWER, Tags.GPU, Tags.PMLInside, Tags.PMLAlpha, Tags.PlotPML,
                                      Tags.RECORDMOVIE, Tags.MOVIENAME, Tags.ACOUSTIC_LOG_SCALE,
                                      Tags.SENSOR_DIRECTIVITY_PATTERN]

        k_wave_settings = Settings({
            Tags.SENSOR_NUM_ELEMENTS: PA_device.number_detector_elements,
            Tags.SENSOR_DIRECTIVITY_SIZE_M: PA_device.detector_element_width_mm/1000,
            Tags.SENSOR_CENTER_FREQUENCY_HZ: PA_device.center_frequency_Hz,
            Tags.SENSOR_BANDWIDTH_PERCENT: PA_device.bandwidth_percent
        })

        for parameter in possible_k_wave_parameters:
            if parameter in settings:
                k_wave_settings[parameter] = settings[parameter]

        data_dict["settings"] = k_wave_settings
        sio.savemat(optical_path, data_dict, long_field_names=True)

        if Tags.ACOUSTIC_SIMULATION_3D in settings and settings[Tags.ACOUSTIC_SIMULATION_3D] is True:
            print("Simulating 3D....")
            simulation_script_path = "simulate_3D"
        else:
            print("Simulating 2D....")
            simulation_script_path = "simulate_2D"

        cmd = list()
        cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-automation")
        cmd.append("-wait")
        cmd.append("-r")
        cmd.append("addpath('"+settings[Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION]+"');" +
                   simulation_script_path + "('" + optical_path + "');exit;")
        cur_dir = os.getcwd()
        os.chdir(settings[Tags.SIMULATION_PATH])
        print(cmd)
        subprocess.run(cmd)

        raw_time_series_data = sio.loadmat(optical_path)[Tags.TIME_SERIES_DATA]

        time_grid = sio.loadmat(optical_path + "dt.mat")
        num_time_steps = int(np.round(time_grid["number_time_steps"]))

        # TODO create a flag in the PA device specification if output should be 2D or 3D also returns the axis of the imaging plane
        if (Tags.ACOUSTIC_SIMULATION_3D in settings and settings[Tags.ACOUSTIC_SIMULATION_3D] and
            Tags.DIGITAL_DEVICE in settings and settings[Tags.DIGITAL_DEVICE] == Tags.DIGITAL_DEVICE_MSOT):

            sensor_mask = data_dict[Tags.PROPERTY_SENSOR_MASK]
            num_imaging_plane_sensors = int(np.sum(sensor_mask[:, detector_positions_voxels[0][1], :]))

            raw_time_series_data = np.reshape(raw_time_series_data, [num_imaging_plane_sensors, -1, num_time_steps])

            if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings and settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
                if settings[Tags.RECONSTRUCTION_ALGORITHM] in [Tags.RECONSTRUCTION_ALGORITHM_PYTORCH_DAS,
                                                               Tags.RECONSTRUCTION_ALGORITHM_DAS,
                                                               Tags.RECONSTRUCTION_ALGORITHM_DMAS,
                                                               Tags.RECONSTRUCTION_ALGORITHM_SDMAS]:
                    raw_time_series_data = np.average(raw_time_series_data, axis=1)

        settings[Tags.K_WAVE_SPECIFIC_DT] = float(time_grid["time_step"])
        settings[Tags.K_WAVE_SPECIFIC_NT] = num_time_steps

        save_hdf5(settings, settings[Tags.SIMPA_OUTPUT_PATH], "/settings/")

        os.remove(optical_path)
        os.remove(optical_path + "dt.mat")
        os.chdir(cur_dir)

        return raw_time_series_data
