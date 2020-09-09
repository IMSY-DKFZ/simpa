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

from simpa.utils import Tags, SaveFilePaths
from simpa.core.noise_simulation import GaussianNoiseModel
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5


def apply_noise_model_to_time_series_data(settings, acoustic_model_result_path):
    """

    :param settings:
    :param acoustic_model_result_path:
    :return:
    """

    if not (Tags.APPLY_NOISE_MODEL in settings and settings[Tags.APPLY_NOISE_MODEL]):
        print("WARN: No noise model was applied.")
        return acoustic_model_result_path

    noise_model = None
    time_series_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], acoustic_model_result_path)[Tags.TIME_SERIES_DATA]

    if settings[Tags.NOISE_MODEL] == Tags.NOISE_MODEL_GAUSSIAN:
        noise_model = GaussianNoiseModel()

    time_series_data_noise = noise_model.apply_noise_model(time_series_data, settings)

    noise_output_path = SaveFilePaths.NOISE_ACOUSTIC_OUTPUT.format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH])
    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            noise_output_path = \
                SaveFilePaths.NOISE_ACOUSTIC_OUTPUT.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])
    save_hdf5({Tags.TIME_SERIES_DATA: time_series_data_noise}, settings[Tags.SIMPA_OUTPUT_PATH],
              noise_output_path)

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

    noise_model = None
    reconstructed_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], reconstructed_data_path)[Tags.RECONSTRUCTED_DATA]

    if settings[Tags.NOISE_MODEL] == Tags.NOISE_MODEL_GAUSSIAN:
        noise_model = GaussianNoiseModel()

    reconstructed_data_noise = noise_model.apply_noise_model(reconstructed_data, settings)

    noise_output_path = SaveFilePaths.NOISE_RECONSTRCTION_OUTPUT.format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH])
    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            noise_output_path = \
                SaveFilePaths.NOISE_RECONSTRCTION_OUTPUT.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])
    save_hdf5({Tags.RECONSTRUCTED_DATA_NOISE: reconstructed_data_noise}, settings[Tags.SIMPA_OUTPUT_PATH],
              noise_output_path)

    return noise_output_path
