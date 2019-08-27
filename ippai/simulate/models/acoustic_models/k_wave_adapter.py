import numpy as np
import subprocess
from ippai.simulate import Tags
import json
import os


def simulate(settings, optical_path):

    tmp_output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "_output.npy"
    settings["output_file"] = tmp_output_file

    tmp_json_filename = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/test_settings.json"
    with open(tmp_json_filename, "w") as json_file:
        json.dump(settings, json_file)

    cmd = list()
    cmd.append("matlab")
    cmd.append("-nodisplay")
    cmd.append("-r")
    cmd.append(settings[Tags.ACOUSTIC_MODEL_SCRIPT] + " ('" + tmp_json_filename + "', '" + optical_path + "');exit;")
    os.chdir(settings[Tags.SIMULATION_PATH])

    subprocess.run(cmd)

    sensor_data = np.load(tmp_output_file)
    os.remove(os.path.join(settings[Tags.SIMULATION_PATH], "fluence.npy"))
    os.remove(os.path.join(settings[Tags.SIMULATION_PATH], "initial_pressure.npy"))
    os.remove(tmp_output_file)

    return sensor_data


