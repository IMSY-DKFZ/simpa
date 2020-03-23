import numpy as np
import subprocess
from ippai.simulate import Tags, SaveFilePaths
from ippai.io_handling.io_hdf5 import load_hdf5, save_hdf5
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

    # plt.imshow(data_dict["sos"])
    # plt.show()

    #pre, ext = os.path.splitext(optical_path)
    optical_path = settings[Tags.IPPAI_OUTPUT_PATH] + ".mat"
    sio.savemat(optical_path, data_dict)

    tmp_output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "_output.npy"
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
    os.chdir(settings[Tags.SIMULATION_PATH])

    subprocess.run(cmd)

    sensor_data = np.load(tmp_output_file)
    settings["dt_acoustic_sim"] = float(sio.loadmat(tmp_output_file + ".mat", variable_names="time_step")["time_step"])

    os.remove(optical_path)
    os.remove(tmp_output_file)
    os.remove(tmp_output_file + ".mat")
    os.chdir(cur_dir)

    return sensor_data


