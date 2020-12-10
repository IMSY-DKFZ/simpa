# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
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


def generate_dict_path(settings, data_field, wavelength=None, upsampled_data=None):
    sampled_data = "/"
    wl = "/"

    if Tags.PERFORM_UPSAMPLING in settings and settings[Tags.PERFORM_UPSAMPLING] is True:
        if upsampled_data is None:
            raise ValueError("Please specify if the data is the original data or the upsampled data with "
                             "the parameter 'upsampled_data' using either Tags.ORIGINAL_DATA or Tags.UPSAMPLED_DATA!")
        elif upsampled_data is True:
            sampled_data = "/{}/".format(Tags.UPSAMPLED_DATA)
        elif upsampled_data is False:
            sampled_data = "/{}/".format(Tags.ORIGINAL_DATA)
    elif Tags.PERFORM_UPSAMPLING not in settings or settings[Tags.PERFORM_UPSAMPLING] is False:
        sampled_data = "/{}/".format(Tags.ORIGINAL_DATA)

    if Tags.WAVELENGTH in settings:
        if wavelength is None:
            raise ValueError("Please specify the wavelength as int!")
        else:
            wl = "/{}/".format(wavelength)

    simulation_properties = [Tags.SIMULATION_PROPERTIES,
                             Tags.PROPERTY_ABSORPTION_PER_CM,
                             Tags.PROPERTY_SCATTERING_PER_CM,
                             Tags.PROPERTY_ANISOTROPY,
                             Tags.PROPERTY_OXYGENATION,
                             Tags.PROPERTY_SEGMENTATION,
                             Tags.PROPERTY_GRUNEISEN_PARAMETER,
                             Tags.PROPERTY_SPEED_OF_SOUND,
                             Tags.PROPERTY_DENSITY,
                             Tags.PROPERTY_ALPHA_COEFF,
                             Tags.PROPERTY_SENSOR_MASK,
                             Tags.PROPERTY_DIRECTIVITY_ANGLE]

    simulation_output = [Tags.OPTICAL_MODEL_OUTPUT_NAME,
                         Tags.OPTICAL_MODEL_FLUENCE,
                         Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
                         Tags.TIME_SERIES_DATA,
                         Tags.TIME_SERIES_DATA_NOISE,
                         Tags.RECONSTRUCTED_DATA,
                         Tags.RECONSTRUCTED_DATA_NOISE]

    if data_field in simulation_properties:
        dict_path = "/" + Tags.SIMULATIONS + sampled_data + Tags.SIMULATION_PROPERTIES + wl
    elif data_field in simulation_output:
        if data_field in [Tags.OPTICAL_MODEL_FLUENCE, Tags.OPTICAL_MODEL_INITIAL_PRESSURE]:
            dict_path = "/" + Tags.SIMULATIONS + sampled_data + Tags.OPTICAL_MODEL_OUTPUT_NAME + wl
        else:
            dict_path = "/" + Tags.SIMULATIONS + sampled_data + data_field + wl
    else:
        raise ValueError("The requested data_field is not a valid argument. Please specify a valid data_field using "
                         "the Tags from simpa/utils/tags.py!")

    return dict_path
