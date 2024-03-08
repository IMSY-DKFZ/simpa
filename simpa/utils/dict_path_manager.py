# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.utils.constants import (wavelength_dependent_properties,
                                   wavelength_independent_properties,
                                   wavelength_dependent_image_processing_output,
                                   wavelength_independent_image_processing_output,
                                   simulation_output,
                                   simulation_output_fields,
                                   toolkit_properties)


def generate_dict_path(data_field, wavelength: (int, float) = None) -> str:
    """
    Generates a path within an hdf5 file in the SIMPA convention

    :param data_field: Data field that is supposed to be stored in an hdf5 file.
    :param wavelength: Wavelength of the current simulation.
    :return: String which defines the path to the data_field.
    """

    all_wavelength_independent_properties = wavelength_independent_properties + toolkit_properties

    if data_field in [Tags.SIMULATIONS, Tags.SETTINGS, Tags.DIGITAL_DEVICE, Tags.SIMULATION_PIPELINE]:
        return "/" + data_field + "/"

    if wavelength is not None:
        wl = "/{}/".format(wavelength)

    if data_field in wavelength_dependent_properties:
        if wavelength is not None:
            dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + data_field + wl
        else:
            dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + data_field
    elif data_field in simulation_output:
        if data_field in [Tags.DATA_FIELD_FLUENCE, Tags.DATA_FIELD_INITIAL_PRESSURE, Tags.OPTICAL_MODEL_UNITS,
                          Tags.DATA_FIELD_DIFFUSE_REFLECTANCE, Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                          Tags.DATA_FIELD_PHOTON_EXIT_POS, Tags.DATA_FIELD_PHOTON_EXIT_DIR]:
            if wavelength is not None:
                dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/" + data_field + wl
            else:
                dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/" + data_field
        else:
            if wavelength is not None:
                dict_path = "/" + Tags.SIMULATIONS + "/" + data_field + wl
            else:
                dict_path = "/" + Tags.SIMULATIONS + "/" + data_field

    elif data_field in all_wavelength_independent_properties:
        dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + data_field + "/"
    elif data_field in simulation_output_fields:
        dict_path = "/" + Tags.SIMULATIONS + "/" + data_field + "/"
    elif data_field in wavelength_dependent_image_processing_output:
        if wavelength is not None:
            dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + data_field + wl
        else:
            dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + data_field

    elif data_field in wavelength_independent_image_processing_output:
        dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + data_field + "/"
    else:
        raise ValueError("The requested data_field is not a valid argument. Please specify a valid data_field using "
                         "the Tags from simpa/utils/tags.py!")

    return dict_path


def get_data_field_from_simpa_output(simpa_output: dict, data_field: (tuple, str), wavelength: (int, float) = None):
    """
    Navigates through a dictionary in the standard simpa output format to a specific data field.

    :param simpa_output: Dictionary that is in the standard simpa output format.
    :param data_field: Data field that is contained in simpa_output.
    :param wavelength: Wavelength of the current simulation.
    :return: Queried data_field.
    """

    dict_path = generate_dict_path(data_field, wavelength)
    keys_to_data_field = dict_path.split("/")
    current_dict = simpa_output
    for key in keys_to_data_field:
        if key == "":
            continue
        current_dict = current_dict[key]

    return current_dict
