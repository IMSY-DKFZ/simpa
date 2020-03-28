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

from simulate import Tags
from simulate.volume_creator import create_simulation_volume
from simulate.models.optical_modelling import run_optical_forward_model
from simulate.models.acoustic_modelling import run_acoustic_forward_model
from simulate.models.noise_modelling import apply_noise_model_to_reconstructed_data
from simulate.models.reconstruction import perform_reconstruction
from process.sampling import upsample
from io_handling.io_hdf5 import save_hdf5, load_hdf5
from utils.serialization import IPPAIJSONSerializer
import numpy as np
import os
import json


def simulate(settings):
    """

    :param settings:
    :return:
    """

    ippai_output = dict()
    wavelengths = settings[Tags.WAVELENGTHS]
    volume_output_paths = []
    optical_output_paths = []
    acoustic_output_paths = []
    reconstruction_output_paths = []

    path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    if Tags.IPPAI_OUTPUT_NAME in settings:
        ippai_output_path = path + settings[Tags.IPPAI_OUTPUT_NAME]
    else:
        ippai_output_path = path + "ippai_output"

    serializer = IPPAIJSONSerializer()

    if Tags.SETTINGS_JSON in settings:
        if settings[Tags.SETTINGS_JSON]:
            with open(ippai_output_path + ".json", "w") as json_file:
                json.dump(settings, json_file, indent="\t", default=serializer.default)
            settings[Tags.SETTINGS_JSON_PATH] = ippai_output_path + ".json"

    settings[Tags.IPPAI_OUTPUT_PATH] = ippai_output_path + ".hdf5"

    ippai_output[Tags.SETTINGS] = settings
    save_hdf5(ippai_output, settings[Tags.IPPAI_OUTPUT_PATH])

    for wavelength in wavelengths:

        if settings[Tags.RANDOM_SEED] is not None:
            np.random.seed(settings[Tags.RANDOM_SEED])
        else:
            np.random.seed(None)

        settings[Tags.WAVELENGTH] = wavelength
        volume_output_path = create_simulation_volume(settings)
        volume_output_paths.append(volume_output_path)

        optical_output_path = None
        acoustic_output_path = None

        if settings[Tags.RUN_OPTICAL_MODEL]:
            optical_output_path = run_optical_forward_model(settings, volume_output_path)
            optical_output_paths.append(optical_output_path)

        if Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW in settings:
            if settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW]:
                extract_field_of_view(settings, volume_output_path, optical_output_path, acoustic_output_path)

        if Tags.PERFORM_UPSAMPLING in settings:
            if settings[Tags.PERFORM_UPSAMPLING]:
                optical_output_path = upsample(settings, optical_output_path)
                optical_output_paths.append(optical_output_path)

        if Tags.RUN_ACOUSTIC_MODEL in settings:
            if settings[Tags.RUN_ACOUSTIC_MODEL]:
                acoustic_output_path = run_acoustic_forward_model(settings, optical_output_path)
                acoustic_output_paths.append(acoustic_output_path)

        if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings:
            if settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
                reconstruction_output_path = perform_reconstruction(settings, acoustic_output_path)
                if (Tags.APPLY_NOISE_MODEL in settings) and settings[Tags.APPLY_NOISE_MODEL]:
                    reconstruction_output_path = apply_noise_model_to_reconstructed_data(settings, reconstruction_output_path)
                reconstruction_output_paths.append(reconstruction_output_path)

    return [volume_output_paths, optical_output_paths, acoustic_output_paths, reconstruction_output_paths]


def extract_field_of_view(settings, volume_path, optical_path, acoustic_path):
    if volume_path is not None:
        volume_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], volume_path)
        sizes = np.shape(volume_data[Tags.PROPERTY_ABSORPTION_PER_CM])
        for key, value in volume_data.items():
            if np.shape(value) == sizes:
                volume_data[key] = value[:, int(sizes[1]/2), :]

        save_hdf5(volume_data, settings[Tags.IPPAI_OUTPUT_PATH], volume_path)

    if optical_path is not None:
        optical_data = load_hdf5(settings[Tags.IPPAI_OUTPUT_PATH], optical_path)
        fluence = optical_data['fluence']
        sizes = np.shape(fluence)
        optical_data["fluence"] = fluence[:, int(sizes[1] / 2), :]
        optical_data['initial_pressure'] = optical_data['initial_pressure'][:, int(sizes[1] / 2), :]

        save_hdf5(optical_data, settings[Tags.IPPAI_OUTPUT_PATH], optical_path)


    if acoustic_path is not None:
        acoustic_data = np.load(acoustic_path)
