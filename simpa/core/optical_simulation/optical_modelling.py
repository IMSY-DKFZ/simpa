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
from simpa.core.optical_simulation.mcx_adapter import McxAdapter
from simpa.core.optical_simulation.mcxyz_adapter import McxyzAdapter
from simpa.core.optical_simulation.test_optical_adapter import TestOpticalAdapter
from simpa.io_handling.io_hdf5 import save_hdf5, load_hdf5
from simpa.utils.dict_path_manager import generate_dict_path


def run_optical_forward_model(settings):
    """
    This method is the main entry point for running optical forward simulations with the SIMPA toolkit.
    It is important, that the Tags.OPTICAL_MODEL tag is set in the settings dictionary, as well as any
    tags that have to be present for the specific model.
    """

    if Tags.OPTICAL_MODEL not in settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings. Skipping optical modelling.")

    model = settings[Tags.OPTICAL_MODEL]
    forward_model_implementation = None

    if model == Tags.OPTICAL_MODEL_MCXYZ:
        forward_model_implementation = McxyzAdapter()
    elif model == Tags.OPTICAL_MODEL_MCX:
        forward_model_implementation = McxAdapter()
    elif model == Tags.OPTICAL_MODEL_TEST:
        forward_model_implementation = TestOpticalAdapter()

    optical_properties_path = generate_dict_path(Tags.SIMULATION_PROPERTIES, wavelength=settings[Tags.WAVELENGTH])

    fluence = forward_model_implementation.simulate(optical_properties_path, settings)

    optical_properties = load_hdf5(settings[Tags.SIMPA_OUTPUT_PATH], optical_properties_path)
    absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM][str(settings[Tags.WAVELENGTH])]
    gruneisen_parameter = optical_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER]
    initial_pressure = absorption * fluence

    if Tags.PERFORM_UPSAMPLING not in settings or settings[Tags.PERFORM_UPSAMPLING] is False:
        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
            units = Tags.UNITS_PRESSURE
            # Initial pressure should be given in units of Pascale
            conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
            initial_pressure = (absorption * fluence * gruneisen_parameter *
                                (settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] / 1000)
                                * conversion_factor)
        else:
            units = Tags.UNITS_ARBITRARY
            initial_pressure = absorption * fluence
    else:
        units = Tags.UNITS_ARBITRARY

    optical_output_path = generate_dict_path(Tags.OPTICAL_MODEL_OUTPUT_NAME)

    optical_output = {
        Tags.OPTICAL_MODEL_FLUENCE: {settings[Tags.WAVELENGTH]: fluence},
        Tags.OPTICAL_MODEL_INITIAL_PRESSURE: {settings[Tags.WAVELENGTH]: initial_pressure},
        Tags.OPTICAL_MODEL_UNITS: {settings[Tags.WAVELENGTH]: units}
    }

    save_hdf5(optical_output, settings[Tags.SIMPA_OUTPUT_PATH], optical_output_path)

    return optical_output_path
