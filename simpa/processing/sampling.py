"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import numpy as np
from simpa.log import Logger
from simpa.utils import Tags, SaveFilePaths
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.processing import preprocess_images
from simpa.io_handling.io_hdf5 import load_hdf5, save_hdf5
from simpa.io_handling.serialization import SIMPAJSONSerializer
from scipy.ndimage import zoom
import os
import inspect
import subprocess
import json

logger = Logger()

def upsample(settings):
    """
    Upsamples all image_data saved in optical path.

    :param settings: (dict) Dictionary that describes all simulation parameters.
    :return: Path to the upsampled image data.
    """

    logger.info("UPSAMPLE IMAGE")

    optical_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME, wavelength=settings[Tags.WAVELENGTH])

    optical_data = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], optical_path)

    fluence = np.rot90(
        preprocess_images.preprocess_image(settings, np.rot90(optical_data[Tags.OPTICAL_MODEL_FLUENCE], 3)))
    initial_pressure = None

    if Tags.UPSAMPLING_METHOD in settings:

        if settings[Tags.UPSAMPLING_METHOD] == Tags.UPSAMPLING_METHOD_NEAREST_NEIGHBOUR:
            fluence = nn_upsample(settings, fluence)
            # initial_pressure = nn_upsample(settings, initial_pressure)
            props = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                              SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA,
                                                                         settings[Tags.WAVELENGTH]))
            mua = props[Tags.PROPERTY_ABSORPTION_PER_CM]

            # mua = np.flip(mua)
            if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
                # Initial pressure should be given in units of Pascale
                conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
                gruneisen_parameter = props[Tags.PROPERTY_GRUNEISEN_PARAMETER]
                initial_pressure = (mua * fluence * gruneisen_parameter *
                                    (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE][str(settings[Tags.WAVELENGTH])] / 1000)
                                    * conversion_factor)
            else:
                initial_pressure = mua * fluence

        if settings[Tags.UPSAMPLING_METHOD] == Tags.UPSAMPLING_METHOD_BILINEAR:
            fluence = bl_upsample(settings, fluence)
            # initial_pressure = bl_upsample(settings, initial_pressure)
            props = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                              SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA,
                                                                         settings[Tags.WAVELENGTH]))
            mua = props[Tags.PROPERTY_ABSORPTION_PER_CM]

            # mua = np.flip(mua)
            if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
                units = Tags.UNITS_PRESSURE
                # Initial pressure should be given in units of Pascale
                conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
                gruneisen_parameter = props[Tags.PROPERTY_GRUNEISEN_PARAMETER]
                initial_pressure = (mua * fluence * gruneisen_parameter *
                                    (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE][str(settings[Tags.WAVELENGTH])] / 1000)
                                    * conversion_factor)
            else:
                initial_pressure = mua * fluence

        if settings[Tags.UPSAMPLING_METHOD] in ["lanczos2", "lanczos3"]:
            fluence = lanczos_upsample(settings, fluence)
            # initial_pressure = lanczos_upsample(settings, initial_pressure)
            props = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                              SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA,
                                                                         settings[Tags.WAVELENGTH]))
            mua = props[Tags.PROPERTY_ABSORPTION_PER_CM]

            # mua = np.flip(mua)
            if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
                units = Tags.UNITS_PRESSURE
                # Initial pressure should be given in units of Pascale
                conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
                gruneisen_parameter = props[Tags.PROPERTY_GRUNEISEN_PARAMETER]
                initial_pressure = (mua * fluence * gruneisen_parameter *
                                    (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE][str(settings[Tags.WAVELENGTH])] / 1000)
                                    * conversion_factor)
            else:
                initial_pressure = mua * fluence

    else:
        fluence = nn_upsample(settings, fluence)
        # initial_pressure = nn_upsample(settings, initial_pressure)
        props = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH],
                          SaveFilePaths.SIMULATION_PROPERTIES.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH]))
        mua = props[Tags.PROPERTY_ABSORPTION_PER_CM]

        # mua = np.flip(mua)
        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
            units = Tags.UNITS_PRESSURE
            # Initial pressure should be given in units of Pascale
            conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
            gruneisen_parameter = props[Tags.PROPERTY_GRUNEISEN_PARAMETER]
            initial_pressure = (mua * fluence * gruneisen_parameter *
                                (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE][str(settings[Tags.WAVELENGTH])] / 1000)
                                * conversion_factor)
        else:
            initial_pressure = mua * fluence
    upsampled_optical_output_path = SaveFilePaths.OPTICAL_OUTPUT.\
        format(Tags.UPSAMPLED_DATA, str(settings[Tags.WAVELENGTH]))

    save_hdf5({Tags.OPTICAL_MODEL_FLUENCE: fluence, Tags.OPTICAL_MODEL_INITIAL_PRESSURE: initial_pressure},
              settings[Tags.SIMPA_OUTPUT_PATH],
              upsampled_optical_output_path)

    return upsampled_optical_output_path


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
        serializer = SIMPAJSONSerializer()
        json.dump(settings, json_file, indent="\t", default=serializer.default)

    base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    cmd = list()
    cmd.append(settings[Tags.ACOUSTIC_MODEL_BINARY_PATH])
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-r")
    cmd.append("addpath('"+base_script_path+"');" +
               "upsampling.m" + "('" + tmp_output_file + "', '" + tmp_json_filename + "');exit;")

    cur_dir = os.getcwd()
    os.chdir(settings[Tags.SIMULATION_PATH])

    subprocess.run(cmd)
    upsampled_image = np.load(tmp_output_file)
    os.remove(tmp_output_file)
    os.chdir(cur_dir)

    return upsampled_image
