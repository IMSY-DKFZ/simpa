import numpy as np
import subprocess
from ippai.simulate import Tags
import json
import os
from shutil import copy


def simulate(settings, optical_path):

    tmp_output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/tmp_output.npy"
    settings["output_file"] = tmp_output_file

    tmp_json_filename = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/test_settings.json"
    with open(tmp_json_filename, "w") as json_file:
        json.dump(settings, json_file, indent="\t")

    cmd = list()
    cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-r")
    cmd.append("addpath('"+settings[Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION]+"');" +
               settings[Tags.ACOUSTIC_MODEL_SCRIPT] + "('" + tmp_json_filename +
               "', '" + optical_path + "');exit;")
    cur_dir = os.getcwd()
    os.chdir(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME])

    subprocess.run(cmd)

    sensor_data = np.load(tmp_output_file)
    os.remove(os.path.join(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME], "fluence.npy"))
    os.remove(os.path.join(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME], "initial_pressure.npy"))
    os.remove(tmp_output_file)
    os.chdir(cur_dir)

    return sensor_data


