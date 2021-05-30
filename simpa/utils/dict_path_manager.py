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


def generate_dict_path(data_field, wavelength: (int, float) = None) -> str:
    """
    Generates a path within an hdf5 file in the SIMPA convention

    :param data_field: Data field that is supposed to be stored in an hdf5 file.
    :param wavelength: Wavelength of the current simulation.
    :return: String which defines the path to the data_field.
    """

    if data_field in [Tags.SIMULATIONS, Tags.SETTINGS]:
        return "/" + data_field + "/"

    wavelength_dependent_properties = [Tags.PROPERTY_ABSORPTION_PER_CM,
                                       Tags.PROPERTY_SCATTERING_PER_CM,
                                       Tags.PROPERTY_ANISOTROPY]

    wavelength_independent_properties = [Tags.PROPERTY_OXYGENATION,
                                         Tags.PROPERTY_SEGMENTATION,
                                         Tags.PROPERTY_GRUNEISEN_PARAMETER,
                                         Tags.PROPERTY_SPEED_OF_SOUND,
                                         Tags.PROPERTY_DENSITY,
                                         Tags.PROPERTY_ALPHA_COEFF,
                                         Tags.PROPERTY_SENSOR_MASK,
                                         Tags.PROPERTY_DIRECTIVITY_ANGLE]

    simulation_output = [Tags.OPTICAL_MODEL_FLUENCE,
                         Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
                         Tags.OPTICAL_MODEL_UNITS,
                         Tags.TIME_SERIES_DATA,
                         Tags.TIME_SERIES_DATA_NOISE,
                         Tags.RECONSTRUCTED_DATA,
                         Tags.RECONSTRUCTED_DATA_NOISE]

    simulation_output_fields = [Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                Tags.SIMULATION_PROPERTIES]

    wavelength_dependent_image_processing_output = [Tags.ITERATIVE_qPAI_RESULT]

    wavelength_independent_image_processing_output = [Tags.LINEAR_UNMIXING_RESULT]

    if wavelength is None and ((data_field in wavelength_dependent_properties) or (data_field in simulation_output) or
                               (data_field in wavelength_dependent_image_processing_output)):
        raise ValueError("Please specify the wavelength as integer!")
    else:
        wl = "/{}/".format(wavelength)

    if data_field in wavelength_dependent_properties:
        dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + data_field + wl
    elif data_field in simulation_output:
        if data_field in [Tags.OPTICAL_MODEL_FLUENCE, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, Tags.OPTICAL_MODEL_UNITS]:
            dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/" + data_field + wl
        else:
            dict_path = "/" + Tags.SIMULATIONS + "/" + data_field + wl
    elif data_field in wavelength_independent_properties:
        dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + data_field + "/"
    elif data_field in simulation_output_fields:
        dict_path = "/" + Tags.SIMULATIONS + "/" + data_field + "/"
    elif data_field in wavelength_dependent_image_processing_output:
        dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + data_field + wl
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
