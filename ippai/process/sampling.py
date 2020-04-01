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
from ippai.utils import Tags
from ippai.simulate import SaveFilePaths
from ippai.process import preprocess_images
from ippai.deep_learning import Architectures, datasets
from ippai.io_handling.io_hdf5 import load_hdf5, save_hdf5
from scipy.ndimage import zoom
import os
import torch
import subprocess
import json


def upsample(settings, optical_path):
    """
    Upsamples all image_data saved in optical path.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param optical_path: (str) Path to the .npz file, where the output of the optical forward model is saved.
    :return: Path to the upsampled image data.
    """

    print("UPSAMPLE IMAGE")

    optical_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], optical_path)

    fluence = np.rot90(
        preprocess_images.preprocess_image(settings, np.rot90(optical_data[Tags.OPTICAL_MODEL_FLUENCE], 3)), 3)
    initial_pressure = np.rot90(
        preprocess_images.preprocess_image(settings, np.rot90(optical_data[Tags.OPTICAL_MODEL_INITIAL_PRESSURE], 3)))

    if Tags.UPSAMPLING_METHOD in settings:
        if settings[Tags.UPSAMPLING_METHOD] == Tags.UPSAMPLING_METHOD_DEEP_LEARNING:
            fluence = dl_upsample(settings, fluence)
            initial_pressure = dl_upsample(settings, initial_pressure)

        if settings[Tags.UPSAMPLING_METHOD] == Tags.UPSAMPLING_METHOD_NEAREST_NEIGHBOUR:
            fluence = nn_upsample(settings, fluence)
            initial_pressure = nn_upsample(settings, initial_pressure)

        if settings[Tags.UPSAMPLING_METHOD] == Tags.UPSAMPLING_METHOD_BILINEAR:
            fluence = bl_upsample(settings, fluence)
            initial_pressure = bl_upsample(settings, initial_pressure)

        if settings[Tags.UPSAMPLING_METHOD] in ["lanczos2", "lanczos3"]:
            fluence = lanczos_upsample(settings, fluence)
            initial_pressure = lanczos_upsample(settings, initial_pressure)

    else:
        fluence = nn_upsample(settings, fluence)
        initial_pressure = nn_upsample(settings, initial_pressure)

    upsampled_optical_output_path = SaveFilePaths.OPTICAL_OUTPUT.\
        format(Tags.UPSAMPLED_DATA, str(settings[Tags.WAVELENGTH]))

    save_hdf5({Tags.OPTICAL_MODEL_FLUENCE: fluence, Tags.OPTICAL_MODEL_INITIAL_PRESSURE: initial_pressure},
              settings[Tags.IPPAI_OUTPUT_PATH],
              upsampled_optical_output_path)

    return upsampled_optical_output_path


def dl_upsample(settings, image_data):
    """
    Upsamples the given image with the deep learning model specified in the settings.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param image_data: (numpy array) Image to be upsampled.
    :param model_path: (string) Path to the file of the deep learning model state_dict.
    :return: Upsampled image.
    """

    def model_prediction(image, upscale_factor, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dl_model = Architectures.ESPCN(upscale_factor=upscale_factor)
        dl_model.to(device)
        dl_model.load_state_dict(torch.load(path))
        dl_model.eval()

        with torch.no_grad():
            normalized_image_data, mx, mn = datasets.normalize_min_max(image)
            normalized_tensor = torch.from_numpy(normalized_image_data).unsqueeze(0).unsqueeze(0).type(torch.float32)

            upsampled_tensor = dl_model(normalized_tensor.to(device)).cpu()

            normalized_upsampled_image = upsampled_tensor.detach().squeeze(0).squeeze(0).numpy()

            upsampled_image = datasets.normalize_min_max_inverse(normalized_upsampled_image, mx=mx, mn=mn)
        return upsampled_image

    low_res = settings[Tags.SPACING_MM]
    high_res = settings[Tags.SPACING_MM]/settings[Tags.UPSCALE_FACTOR]

    if settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
        cut_off_pixel = int(round(42.2/settings[Tags.SPACING_MM]))
        image_data = np.rot90(image_data, 3)
        probe_image = image_data[:cut_off_pixel, :]
        volume_image = image_data[cut_off_pixel:, :]
        upsampled_probe_image = model_prediction(probe_image, settings[Tags.UPSCALE_FACTOR],
                                                 "/home/kris/hard_drive/ippai/data/deep_learning_models/"
                                                 "probe_Upscale_0.34_to_0.17/epoch_199.pt")
        upsampled_volume_image = model_prediction(volume_image, settings[Tags.UPSCALE_FACTOR],
                                                  "/home/kris/hard_drive/ippai/data/deep_learning_models/"
                                                  "volume_Upscale_0.34_to_0.17/epoch_199.pt")

        upsampled_image = np.zeros([image_data.shape[0]*2, image_data.shape[1]*2])
        upsampled_image[:cut_off_pixel*settings[Tags.UPSCALE_FACTOR], :] = upsampled_probe_image
        upsampled_image[cut_off_pixel * settings[Tags.UPSCALE_FACTOR]:, :] = upsampled_volume_image
        upsampled_image = np.rot90(upsampled_image)

    else:
        model_path = settings[Tags.DL_MODEL_PATH]

        if model_path is None:
            model_root = "/home/kris/hard_drive/ippai/data/deep_learning_models"
            for i, model in enumerate(os.listdir(model_root)):
                if str(low_res) in model and str(high_res) in model:
                    model_path = os.path.join(model_root, model)
                    break
                elif i == len(os.listdir(model_root)) - 1:
                    raise FileNotFoundError("Deep Learning model with specified scales doesn't exist.")
                else:
                    continue

            for file in os.listdir(model_path):
                model_path = os.path.join(model_path, file)
                break

        upsampled_image = model_prediction(image_data, settings[Tags.UPSCALE_FACTOR], model_path)

    return upsampled_image


def nn_upsample(settings, image_data):
    """
    Upsamples the given image with the nearest neighbor method.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param image_data: (numpy array) Image to be upsampled.
    :return: Upsampled image.
    """
    upsampled_image = zoom(image_data, settings[Tags.UPSCALE_FACTOR], order=0)

    return upsampled_image


def bl_upsample(settings, image_data):
    """
    Upsamples the given image with the bilinear method.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param image_data: (numpy array) Image to be upsampled.
    :return: Upsampled image.
    """
    upsampled_image = zoom(image_data, settings[Tags.UPSCALE_FACTOR], order=1)

    return upsampled_image


def lanczos_upsample(settings, image_data):
    """
    Upsamples the given image with a given lanczos kernel.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param image_data: (numpy array) Image to be upsampled.
    :return: Upsampled image.
    """

    tmp_output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "_output.npy"
    np.save(tmp_output_file, image_data)
    settings["output_file"] = tmp_output_file

    tmp_json_filename = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/test_settings.json"
    with open(tmp_json_filename, "w") as json_file:
        json.dump(settings, json_file, indent="\t")

    cmd = list()
    cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-r")
    cmd.append("addpath('"+settings[Tags.UPSAMPLING_SCRIPT_LOCATION]+"');" +
               settings[Tags.UPSAMPLING_SCRIPT] + "('" + tmp_json_filename + "');exit;")

    cur_dir = os.getcwd()
    os.chdir(settings[Tags.SIMULATION_PATH])

    subprocess.run(cmd)
    upsampled_image = np.load(tmp_output_file)
    os.remove(tmp_output_file)
    os.chdir(cur_dir)

    return upsampled_image
