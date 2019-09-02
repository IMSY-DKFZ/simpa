import numpy as np
from ippai.simulate import Tags
from ippai.process import preprocess_images
from ippai.deep_learning import datasets
from scipy.ndimage import zoom
import os
import torch


def upsample(settings, optical_path):
    """
    Upsamples all image_data saved in optical path.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param optical_path: (str) Path to the .npz file, where the output of the optical forward model is saved.
    :return: Path to the upsampled image data.
    """

    print("UPSAMPLE IMAGE")
    upsampled_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                     "upsampled_" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "_" + \
                     str(settings[Tags.WAVELENGTH]) + ".npz"

    optical_data = np.load(optical_path)

    fluence = np.rot90(preprocess_images.preprocess_image(settings, np.rot90(optical_data["fluence"], 3)), 3)
    initial_pressure = np.rot90(preprocess_images.preprocess_image(settings, np.rot90(optical_data["initial_pressure"], 3)))

    if settings[Tags.UPSAMPLING_METHOD] == "deep_learning":
        fluence = dl_upsample(settings, fluence)
        initial_pressure = dl_upsample(settings, initial_pressure)

    if settings[Tags.UPSAMPLING_METHOD] == "nearest_neighbor":
        fluence = nn_upsample(settings, fluence)
        initial_pressure = nn_upsample(settings, initial_pressure)

    if settings[Tags.UPSAMPLING_METHOD] == "bilinear":
        fluence = bl_upsample(settings, fluence)
        initial_pressure = bl_upsample(settings, initial_pressure)

    np.savez(upsampled_path, fluence=fluence, initial_pressure=initial_pressure)

    return upsampled_path


def dl_upsample(settings, image_data):
    """
    Upsamples the given image with the deep learning model specified in the settings.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :param image_data: (numpy array) Image to be upsampled.
    :return: Upsampled image.
    """
    low_res = settings[Tags.SPACING_MM]
    high_res = settings[Tags.SPACING_MM]/settings[Tags.UPSCALE_FACTOR]

    model_root = "../deep_learning/models"

    for i, model in enumerate(os.listdir(model_root)):
        if str(low_res) in model and str(high_res) in model:
            model_path = os.path.join(model_root, model)
            break
        elif i == len(os.listdir(model_root)) - 1:
            raise FileNotFoundError("Deep Learning model with specified scales doesn't exist.")
        else:
            continue

    dl_model = torch.load(model_path)
    dl_model.eval()

    return image_data


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
