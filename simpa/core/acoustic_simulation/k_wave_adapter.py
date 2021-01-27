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

import numpy as np
import subprocess
from simpa.utils import Tags, SaveFilePaths
from simpa.io_handling.io_hdf5 import load_hdf5, save_hdf5
from simpa.io_handling.serialization import SIMPAJSONSerializer
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.utils.settings_generator import Settings
import json
import os
import scipy.io as sio
from simpa.core.device_digital_twins.msot_devices import MSOTAcuityEcho


def simulate(settings):

    optical_path = generate_dict_path(settings, Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                      wavelength=settings[Tags.WAVELENGTH],
                                      upsampled_data=True)

    data_dict = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], optical_path)

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            tmp_ac_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                                    SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA,
                                                                               settings[Tags.WAVELENGTH]))
        else:
            tmp_ac_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                                    SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA,
                                                                               settings[Tags.WAVELENGTH]))
    else:
        tmp_ac_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                                SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA,
                                                                           settings[Tags.WAVELENGTH]))

    if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
        axes = (0, 1)
    else:
        axes = (0, 2)
    data_dict[Tags.PROPERTY_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.PROPERTY_SPEED_OF_SOUND], 3, axes=axes)
    data_dict[Tags.PROPERTY_DENSITY] = np.rot90(tmp_ac_data[Tags.PROPERTY_DENSITY], 3, axes=axes)
    data_dict[Tags.PROPERTY_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.PROPERTY_ALPHA_COEFF], 3, axes=axes)
    data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE] = np.flip(np.rot90(data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE],
                                                                      axes=axes))
    data_dict[Tags.OPTICAL_MODEL_FLUENCE] = np.flip(np.rot90(data_dict[Tags.OPTICAL_MODEL_FLUENCE], axes=axes))

    PA_device = MSOTAcuityEcho()
    detector_positions = PA_device.get_detector_element_positions(settings)
    detector_positions = np.round(detector_positions / settings[Tags.SPACING_MM]).astype(int)

    sensor_map = np.zeros(np.shape(data_dict[Tags.OPTICAL_MODEL_INITIAL_PRESSURE]))
    if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
        sensor_map[detector_positions[:, 2], detector_positions[:, 0]] = 1
    else:
        half_y_dir_detector_pixels = int(round(0.5*PA_device.detector_element_length_mm/settings[Tags.SPACING_MM]))
        for pixel in np.arange(- half_y_dir_detector_pixels, half_y_dir_detector_pixels, 1):
            sensor_map[detector_positions[:, 2], detector_positions[:, 1] + pixel, detector_positions[:, 0]] = 1
    data_dict[Tags.PROPERTY_SENSOR_MASK] = sensor_map
    save_hdf5({Tags.PROPERTY_SENSOR_MASK: sensor_map}, settings[Tags.SIMPA_OUTPUT_PATH],
              generate_dict_path(settings, Tags.PROPERTY_SENSOR_MASK, upsampled_data=False,
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
                                  Tags.MEDIUM_ALPHA_POWER, Tags.GPU, Tags.PMLInside, Tags.PMLAlpha, Tags.PlotPML,
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

    json_path, ext = os.path.splitext(settings[Tags.SIMPA_OUTPUT_PATH])
    tmp_json_filename = json_path + ".json"
    # if Tags.SETTINGS_JSON_PATH not in settings:
    #     with open(tmp_json_filename, "w") as json_file:
    #         serializer = SIMPAJSONSerializer()
    #         json.dump(settings, json_file, indent="\t", default=serializer.default)

    if Tags.ACOUSTIC_SIMULATION_3D in settings and settings[Tags.ACOUSTIC_SIMULATION_3D] is True:
        simulation_script_path = "simulate_3D"
    else:
        simulation_script_path = "simulate_2D"

    cmd = list()
    cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-r")
    cmd.append("addpath('"+settings[Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION]+"');" +
               simulation_script_path + "('" + optical_path + "');exit;")
    cur_dir = os.getcwd()
    os.chdir(settings[Tags.SIMULATION_PATH])
    print(cmd)
    subprocess.run(cmd)

    raw_time_series_data = sio.loadmat(optical_path + ".mat")[Tags.TIME_SERIES_DATA]

    if Tags.ACOUSTIC_SIMULATION_3D in settings and settings[Tags.ACOUSTIC_SIMULATION_3D]:

        num_time_steps = np.shape(raw_time_series_data)[1]
        num_imaging_plane_sensors = np.shape(np.argwhere(
            data_dict[Tags.PROPERTY_SENSOR_MASK][:, int(np.shape(data_dict[Tags.PROPERTY_SENSOR_MASK])[1] / 2), :]))[0]
        num_orthogonal_sensors = np.shape(np.argwhere(
            data_dict[Tags.PROPERTY_SENSOR_MASK][:, :, int(np.shape(data_dict[Tags.PROPERTY_SENSOR_MASK])[2] / 2)]))[0]

        if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings and settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
            if settings[Tags.RECONSTRUCTION_ALGORITHM] in [Tags.RECONSTRUCTION_ALGORITHM_DAS,
                                                           Tags.RECONSTRUCTION_ALGORITHM_DMAS,
                                                           Tags.RECONSTRUCTION_ALGORITHM_SDMAS]:
                raw_time_series_data = np.reshape(raw_time_series_data, [num_imaging_plane_sensors,
                                                                         num_orthogonal_sensors, num_time_steps])
                raw_time_series_data = np.sum(raw_time_series_data, axis=1) / num_orthogonal_sensors
        else:
            raw_time_series_data = np.reshape(raw_time_series_data, [num_imaging_plane_sensors,
                                                                     num_orthogonal_sensors, num_time_steps])

    time_grid = sio.loadmat(optical_path + "dt.mat")
    settings["dt_acoustic_sim"] = float(time_grid["time_step"])
    settings["Nt_acoustic_sim"] = float(time_grid["number_time_steps"])

    os.remove(optical_path)
    os.remove(optical_path + ".mat")
    os.remove(optical_path + "dt.mat")
    os.chdir(cur_dir)

    return raw_time_series_data
