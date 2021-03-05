# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
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

from simpa.utils import Tags
from simpa.core.volume_creation.volume_creation import run_volume_creation
from simpa.core.optical_simulation.optical_modelling import run_optical_forward_model
from simpa.core.acoustic_simulation.acoustic_modelling import run_acoustic_forward_model
from simpa.core.noise_simulation.noise_modelling import apply_noise_model_to_time_series_data
from simpa.core.image_reconstruction.reconstruction_modelling import perform_reconstruction
from simpa.process.sampling import upsample
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
from simpa.io_handling.serialization import SIMPAJSONSerializer
from simpa.utils.settings_generator import Settings
import numpy as np
import os
import json


def simulate(settings):
    """
    This method constitutes the staring point for the simulation pipeline
    of the SIMPA toolkit. It calls all relevant and wanted simulation modules in the
    following pre-determined order::

        def simulation(settings):
            for wavelength in settings[Tags.WAVELENGTHS]:

                simulation_data = volume_creator.create_simulation_volumes(settings)
                if optical_simulation in settings:
                    optical_model.simulate(simulation_data, settings)
                if acoustic_simulation in settings:
                    acoustic_model.simulate(simulation_data, settings)
                if noise_simulation in settings:
                    noise_model.simulate(simulation_data, settings)
                if image_reconstruction in settings:
                    reconstruction_model.simulate(simulation_data, settings)

                io_handler.save_hdf5(simulation_data, settings)

    :param settings: settings dictionary containing the simulation instructions
    :return: list with the save paths of the simulated data within the HDF5 file.
    """

    if not isinstance(settings, Settings):
        raise TypeError("Use a Settings instance from simpa.utils.settings_generator as simulation input.")

    simpa_output = dict()
    path = settings[Tags.SIMULATION_PATH] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    if Tags.SIMPA_OUTPUT_NAME in settings:
        simpa_output_path = path + settings[Tags.SIMPA_OUTPUT_NAME]
    else:
        simpa_output_path = path + settings[Tags.VOLUME_NAME]

    serializer = SIMPAJSONSerializer()

    if Tags.SETTINGS_JSON in settings:
        if settings[Tags.SETTINGS_JSON]:
            with open(simpa_output_path + ".json", "w") as json_file:
                json.dump(settings, json_file, indent="\t", default=serializer.default)
            settings[Tags.SETTINGS_JSON_PATH] = simpa_output_path + ".json"

    settings[Tags.SIMPA_OUTPUT_PATH] = simpa_output_path + ".hdf5"

    simpa_output[Tags.SETTINGS] = settings
    save_hdf5(simpa_output, settings[Tags.SIMPA_OUTPUT_PATH])

    for wavelength in settings[Tags.WAVELENGTHS]:

        if settings[Tags.RANDOM_SEED] is not None:
            np.random.seed(settings[Tags.RANDOM_SEED])
        else:
            np.random.seed(None)

        settings[Tags.WAVELENGTH] = wavelength
        volume_output_path = run_volume_creation(settings)

        optical_output_path = None
        acoustic_output_path = None

        if Tags.RUN_OPTICAL_MODEL in settings and settings[Tags.RUN_OPTICAL_MODEL]:
            optical_output_path = run_optical_forward_model(settings)

        if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
            if Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW not in settings or settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW]:
                extract_field_of_view(settings, volume_output_path, optical_output_path, acoustic_output_path)

        if Tags.PERFORM_UPSAMPLING in settings:
            if settings[Tags.PERFORM_UPSAMPLING]:
                upsample(settings)

        if Tags.RUN_ACOUSTIC_MODEL in settings:
            if settings[Tags.RUN_ACOUSTIC_MODEL]:
                acoustic_output_path = run_acoustic_forward_model(settings)
                if (Tags.APPLY_NOISE_MODEL in settings) and settings[Tags.APPLY_NOISE_MODEL]:
                    apply_noise_model_to_time_series_data(settings, acoustic_output_path)

        if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings:
            if settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
                perform_reconstruction(settings)

    # Quick and dirty fix:
    all_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH])
    save_hdf5(all_data, settings[Tags.SIMPA_OUTPUT_PATH])


def extract_field_of_view(settings, volume_path, optical_path, acoustic_path):
    wavelength = str(settings[Tags.WAVELENGTH])
    if volume_path is not None:
        volume_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], volume_path)
        sizes = np.shape(volume_data[Tags.PROPERTY_ABSORPTION_PER_CM])
        for key, value in volume_data.items():
            if key in [Tags.PROPERTY_ABSORPTION_PER_CM, Tags.PROPERTY_SCATTERING_PER_CM, Tags.PROPERTY_ANISOTROPY]:
                if np.shape(value[wavelength]) == sizes:
                    volume_data[key][wavelength] = value[:, int(sizes[1] / 2), :]
            else:
                if np.shape(value) == sizes:
                    volume_data[key] = value[:, int(sizes[1] / 2), :]

        save_hdf5(volume_data, settings[Tags.SIMPA_OUTPUT_PATH], volume_path)

    if optical_path is not None:
        optical_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], optical_path)
        fluence = optical_data['fluence'][wavelength]
        sizes = np.shape(fluence)
        optical_data["fluence"][wavelength] = fluence[:, int(sizes[1] / 2), :]
        optical_data['initial_pressure'][wavelength] = optical_data['initial_pressure'][wavelength][:, int(sizes[1] / 2), :]

        save_hdf5(optical_data, settings[Tags.SIMPA_OUTPUT_PATH], optical_path)
