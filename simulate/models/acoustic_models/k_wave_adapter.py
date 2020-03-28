# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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
from simulate import Tags, SaveFilePaths
from io_handling.io_hdf5 import load_hdf5
from utils.serialization import IPPAIJSONSerializer
import json
import os
import scipy.io as sio


def simulate(settings, optical_path):

    data_dict = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], optical_path)

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            tmp_ac_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH],
                                    SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH]))
        else:
            tmp_ac_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH],
                                    SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH]))
    else:
        tmp_ac_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH],
                                SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH]))

    data_dict[Tags.PROPERTY_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[Tags.PROPERTY_SPEED_OF_SOUND], 3)
    data_dict[Tags.PROPERTY_DENSITY] = np.rot90(tmp_ac_data[Tags.PROPERTY_DENSITY], 3)
    data_dict[Tags.PROPERTY_ALPHA_COEFF] = np.rot90(tmp_ac_data[Tags.PROPERTY_ALPHA_COEFF], 3)
    data_dict[Tags.PROPERTY_SENSOR_MASK] = np.rot90(tmp_ac_data[Tags.PROPERTY_SENSOR_MASK], 3)
    try:
        data_dict[Tags.PROPERTY_DIRECTIVITY_ANGLE] = np.rot90(tmp_ac_data[Tags.PROPERTY_DIRECTIVITY_ANGLE], 3)
    except ValueError:
        print("No directivity_angle specified")
    except KeyError:
        print("No directivity_angle specified")

    optical_path = settings[Tags.IPPAI_OUTPUT_PATH] + ".mat"
    sio.savemat(optical_path, data_dict)

    json_path, ext = os.path.splitext(settings[Tags.IPPAI_OUTPUT_PATH])
    tmp_json_filename = json_path + ".json"
    if Tags.SETTINGS_JSON_PATH not in settings:
        with open(tmp_json_filename, "w") as json_file:
            serializer = IPPAIJSONSerializer()
            json.dump(settings, json_file, indent="\t", default=serializer.default)

    cmd = list()
    cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-r")
    cmd.append("addpath('"+settings[Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION]+"');" +
               settings[Tags.ACOUSTIC_MODEL_SCRIPT] + "('" + tmp_json_filename +
               "', '" + optical_path + "');exit;")
    cur_dir = os.getcwd()
    os.chdir(settings[Tags.SIMULATION_PATH])

    subprocess.run(cmd)

    raw_time_series_data = sio.loadmat(optical_path + ".mat")["sensor_data_2D"]
    settings["dt_acoustic_sim"] = float(sio.loadmat(optical_path + "dt.mat", variable_names="time_step")["time_step"])

    os.remove(optical_path)
    os.remove(optical_path + ".mat")
    os.remove(optical_path + "dt.mat")
    if Tags.SETTINGS_JSON_PATH not in settings:
        os.remove(tmp_json_filename)
    os.chdir(cur_dir)

    return raw_time_series_data


