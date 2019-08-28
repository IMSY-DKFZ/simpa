import numpy as np
from ippai.simulate import Tags
from ippai.process import preprocess_images


def upsample(settings, optical_path):

    upsampled_path = optical_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                   Tags.OPTICAL_MODEL_OUTPUT_NAME + "_" + \
                   str(settings[Tags.WAVELENGTH]) + ".npz"

    optical_data = np.load(optical_path)

    for key in optical_data.keys():
        optical_data[key] = preprocess_images.preprocess_image(settings, optical_data[key])

    if settings[Tags.UPSAMPLING_METHOD] == "deep_learning":
        for key in optical_data.keys():
            optical_data[key] = dl_upsample(settings, optical_path)

    np.savez()

    return upsampled_path


def dl_upsample(settings, optical_path):
    upsampled_path = optical_path
    return upsampled_path