from ippai.simulate import Tags
import numpy as np

from ippai.simulate.models.noise_models import GaussianNoise


def apply_noise_model_to_time_series_data(settings, acoustic_model_result_path):
    """

    :param settings:
    :param acoustic_model_result_path:
    :return:
    """

    if not (Tags.APPLY_NOISE_MODEL in settings and settings[Tags.APPLY_NOISE_MODEL]):
        print("WARN: No noise model was applied.")
        return acoustic_model_result_path

    noise_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                          Tags.NOISE_MODEL_OUTPUT_NAME + "_" + \
                          str(settings[Tags.WAVELENGTH]) + ".npz"

    noise_model = None
    time_series_data = np.load(acoustic_model_result_path)[Tags.TIME_SERIES_DATA]

    if settings[Tags.NOISE_MODEL] == Tags.NOISE_MODEL_GAUSSIAN:
        noise_model = GaussianNoise()

    time_series_data_noise = noise_model.apply_noise_model(time_series_data, settings)

    np.savez(noise_output_path,
             time_series_data=time_series_data_noise)

    return noise_output_path


def apply_noise_model_to_reconstructed_data(settings, reconstructed_data_path):
    """
    TODO
    :param settings:
    :param reconstructed_data_path:
    :return:
    """

    if not (Tags.APPLY_NOISE_MODEL in settings and settings[Tags.APPLY_NOISE_MODEL]):
        print("WARN: No noise model was applied.")
        return reconstructed_data_path

    noise_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                        Tags.NOISE_MODEL_OUTPUT_NAME + "_" + \
                        str(settings[Tags.WAVELENGTH]) + ".npz"

    noise_model = None
    reconstructed_data = np.load(reconstructed_data_path)[Tags.RECONSTRUCTED_DATA]

    if settings[Tags.NOISE_MODEL] == Tags.NOISE_MODEL_GAUSSIAN:
        noise_model = GaussianNoise()

    reconstructed_data_noise = noise_model.apply_noise_model(reconstructed_data, settings)

    np.savez(noise_output_path,
             reconstructed_data_noise=reconstructed_data_noise)

    return noise_output_path
